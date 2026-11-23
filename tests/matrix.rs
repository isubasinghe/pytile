use melior::{
    ir::{operation::OperationLike, BlockLike, Module},
    pass::PassManager,
    utility::{parse_pass_pipeline, register_all_llvm_translations, register_all_passes},
    ExecutionEngine,
};
use pytile::{frontend::parse_source, runtime::unit_from_expression};
use rustpython_parser::ast;

#[test]
fn matrix_scores_execute_with_expected_values() {
    let context = pytile::ir::create_context();
    register_all_llvm_translations(&context);
    register_all_passes();
    let ast = parse_source(
        include_str!("../examples/matrix_scores.py"),
        "matrix_scores.py",
    )
    .unwrap();
    let function = ast
        .iter()
        .find_map(|stmt| match stmt {
            ast::Stmt::FunctionDef(function) => Some(function),
            _ => None,
        })
        .unwrap();
    let mut unit = unit_from_expression(function,
        "(np.zeros((3, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))"
    ).unwrap();
    let mut module = unit.compile_module(&context, &ast).unwrap();
    let harness = Module::parse(&context, r#"
        module {
            func.func @check(%row: index, %col: index) -> f32 attributes {llvm.emit_c_interface} {
                %x = arith.constant dense<[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf32>
                %w = arith.constant dense<[[2.0, -1.0], [-1.0, 2.0]]> : tensor<2x2xf32>
                %b = arith.constant dense<[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]> : tensor<3x2xf32>
                %scores = func.call @compute(%x, %w, %b) : (tensor<3x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
                %value = tensor.extract %scores[%row, %col] : tensor<3x2xf32>
                return %value : f32
            }
            func.func private @compute(tensor<3x2xf32>, tensor<2x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
        }
    "#).unwrap();
    let check = harness.body().first_operation().unwrap();
    module.body().append_operation((*check).clone());
    assert!(module.as_operation().verify());

    // Execute the generated operations on the CPU to test their numerical
    // semantics independently of the future tensor-to-GPU lowering.
    let passes = PassManager::new(&context);
    parse_pass_pipeline(
        passes.as_operation_pass_manager(),
        "builtin.module(
            arith-expand,
            convert-elementwise-to-linalg,
            one-shot-bufferize{bufferize-function-boundaries},
            func.func(convert-linalg-to-loops),
            lower-affine,
            convert-scf-to-cf,
            convert-to-llvm,
            reconcile-unrealized-casts
        )",
    )
    .unwrap();
    passes.run(&mut module).unwrap();
    assert!(module.as_operation().verify());
    let engine = ExecutionEngine::new(&module, 2, &[], false);
    for (row, expected) in [[3.0_f32, -1.0], [0.0, 2.0], [2.0, 1.0]].iter().enumerate() {
        for (col, &expected) in expected.iter().enumerate() {
            let mut row = row as i64;
            let mut col = col as i64;
            let mut result = f32::NAN;
            // The C wrapper takes two 64-bit index arguments and returns f32.
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
            assert_eq!(result, expected);
        }
    }
}
