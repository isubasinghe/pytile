use melior::{
    ir::{operation::OperationLike, BlockLike, Module},
    pass::PassManager,
    utility::{parse_pass_pipeline, register_all_llvm_translations, register_all_passes},
    ExecutionEngine,
};
use pytile::{frontend::parse_source, runtime::unit_from_expression};
use rustpython_parser::ast;

fn tensor_type(shape: &[usize]) -> String {
    format!(
        "tensor<{}xf32>",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn literal(shape: &[usize], values: &[f32]) -> String {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    if shape.len() == 1 {
        format!(
            "[{}]",
            values
                .iter()
                .map(|x| format!("0x{:08X}", x.to_bits()))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        let stride = shape[1..].iter().product::<usize>();
        format!(
            "[{}]",
            values
                .chunks(stride)
                .map(|row| literal(&shape[1..], row))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Run generated tensor IR through LLVM on the CPU, independently of GPU work.
fn execute(source: &str, shapes: &[&[usize]], values: &[Vec<f32>], output: [usize; 2]) -> Vec<f32> {
    let context = pytile::ir::create_context();
    register_all_llvm_translations(&context);
    register_all_passes();
    let ast = parse_source(source, "test.py").unwrap();
    let function = ast
        .iter()
        .find_map(|stmt| match stmt {
            ast::Stmt::FunctionDef(f) => Some(f),
            _ => None,
        })
        .unwrap();
    let args = format!(
        "({},)",
        shapes
            .iter()
            .map(|shape| format!("np.zeros({shape:?}, dtype=np.float32)"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut unit = unit_from_expression(function, &args).unwrap();
    let mut module = unit.compile_module(&context, &ast).unwrap();
    let types = shapes
        .iter()
        .map(|shape| tensor_type(shape))
        .collect::<Vec<_>>();
    let constants = shapes
        .iter()
        .zip(values)
        .enumerate()
        .map(|(i, (shape, data))| {
            format!(
                "%arg{i} = arith.constant dense<{}> : {}",
                literal(shape, data),
                types[i]
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let operands = (0..shapes.len())
        .map(|i| format!("%arg{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let output_type = tensor_type(&output);
    let harness = Module::parse(&context, &format!(r#"
        module {{
            func.func @check(%row: index, %col: index) -> f32 attributes {{llvm.emit_c_interface}} {{
                {constants}
                %out = func.call @{name}({operands}) : ({types}) -> {output_type}
                %value = tensor.extract %out[%row, %col] : {output_type}
                return %value : f32
            }}
            func.func private @{name}({types}) -> {output_type}
        }}
    "#, name=function.name, types=types.join(", "))).unwrap();
    module
        .body()
        .append_operation((*harness.body().first_operation().unwrap()).clone());
    assert!(module.as_operation().verify());
    let passes = PassManager::new(&context);
    parse_pass_pipeline(
        passes.as_operation_pass_manager(),
        "builtin.module(arith-expand,convert-elementwise-to-linalg,
            one-shot-bufferize{bufferize-function-boundaries},
            func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,
            convert-to-llvm,reconcile-unrealized-casts)",
    )
    .unwrap();
    passes.run(&mut module).unwrap();
    let engine = ExecutionEngine::new(&module, 2, &[], false);
    let mut results = Vec::new();
    for row in 0..output[0] {
        for col in 0..output[1] {
            let mut row = row as i64;
            let mut col = col as i64;
            let mut result = f32::NAN;
            // The verified C wrapper takes two 64-bit indices and returns f32.
            unsafe {
                engine
                    .invoke_packed(
                        "check",
                        &mut [
                            &mut row as *mut i64 as *mut (),
                            &mut col as *mut i64 as *mut (),
                            &mut result as *mut f32 as *mut (),
                        ],
                    )
                    .unwrap();
            }
            results.push(result);
        }
    }
    results
}

#[test]
fn full_mnist_forward_matches_numpy_logits() {
    fn data(count: usize, modulus: i32, offset: i32, divisor: f32) -> Vec<f32> {
        (0..count)
            .map(|i| ((i as i32 % modulus) - offset) as f32 / divisor)
            .collect()
    }
    let values = [
        data(2 * 784, 13, 6, 16.0),
        data(784 * 32, 11, 5, 32.0),
        data(32, 5, 2, 8.0),
        data(32 * 10, 7, 3, 16.0),
        data(10, 3, 1, 4.0),
    ];
    // Generated independently by NumPy's mnist_forward.forward with the same
    // deterministic float32 inputs. The two rows exercise different activations.
    let expected = [
        -0.24157715,
        -0.037231445,
        0.23461914,
        -0.21789551,
        0.032592773,
        0.22155762,
        -0.24206543,
        0.008422852,
        0.21276855,
        -0.26538086,
        -0.28112793,
        0.0031738281,
        0.18579102,
        -0.19702148,
        0.0018310547,
        0.23913574,
        -0.20178223,
        -0.03112793,
        0.25317383,
        -0.31420898,
    ];
    let actual = execute(
        include_str!("../examples/mnist_forward.py"),
        &[&[2, 784], &[784, 32], &[32], &[32, 10], &[10]],
        &values,
        [2, 10],
    );
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn broadcasting_and_reversed_scalar_maximum_execute() {
    let result = execute(
        "def compute(x, y): return np.maximum(0.0, x + y) * 0.5\n",
        &[&[2, 1], &[1, 3]],
        &[vec![-2.0, 2.0], vec![-1.0, 0.0, 3.0]],
        [2, 3],
    );
    assert_eq!(result, [0.0, 0.0, 0.5, 0.5, 1.0, 2.5]);
    let result = execute(
        "def compute(x, y): return np.maximum(x, y)\n",
        &[&[2, 1], &[1, 3]],
        &[vec![-2.0, 2.0], vec![-1.0, 0.0, 3.0]],
        [2, 3],
    );
    assert_eq!(result, [-1.0, 0.0, 3.0, 2.0, 2.0, 3.0]);
}

#[test]
fn maximum_preserves_nan_and_handles_negative_thresholds() {
    let result = execute(
        "def compute(x): return np.maximum(x, -0.5)\n",
        &[&[2, 3]],
        &[vec![
            -2.0,
            -0.25,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
        ]],
        [2, 3],
    );
    assert_eq!(&result[..2], &[-0.5, -0.25]);
    assert!(result[2].is_nan());
    assert_eq!(&result[3..], &[f32::INFINITY, -0.5, 1.0]);
}

#[test]
fn unsupported_calls_and_bad_shapes_report_errors() {
    let context = pytile::ir::create_context();
    for (expression, message) in [
        ("np.maximum(x)", "exactly two positional"),
        ("np.maximum(x, 0.0, out=x)", "no keywords"),
        ("np.minimum(x, 0.0)", "Only np.maximum"),
        ("np.maximum(x, y)", "Cannot broadcast"),
    ] {
        let source = format!("def compute(x, y): return {expression}\n");
        let ast = parse_source(&source, "test.py").unwrap();
        let ast::Stmt::FunctionDef(function) = &ast[0] else {
            panic!("expected function")
        };
        let mut unit = unit_from_expression(
            function,
            "(np.zeros((2, 3), dtype=np.float32), np.zeros(4, dtype=np.float32))",
        )
        .unwrap();
        let error = unit.compile_module(&context, &ast).err().unwrap();
        assert!(error.to_string().contains(message), "{error}");
    }
}
