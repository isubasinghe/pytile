use pytile::frontend::{parse_source, CompileUnit, Typ};
use rumpy::DType;
use rustpython_parser::ast;
use rustpython_vm::{builtins::PyTuple, compiler::Mode, Interpreter, VirtualMachine};

fn with_vm(check: impl FnOnce(&VirtualMachine)) {
    let builder = Interpreter::builder(Default::default());
    let numpy = rumpy::module_def(&builder.ctx);
    let interpreter = builder.add_native_module(numpy).build();
    interpreter.enter(check);
    assert_eq!(interpreter.finalize(None), 0);
}

fn function(source: &str) -> ast::StmtFunctionDef {
    parse_source(source, "test.py")
        .unwrap()
        .into_iter()
        .find_map(|stmt| match stmt {
            ast::Stmt::FunctionDef(function) => Some(function),
            _ => None,
        })
        .unwrap()
}

fn bind<'c, 'a>(
    vm: &VirtualMachine,
    function: &ast::StmtFunctionDef,
    inputs: &str,
) -> anyhow::Result<CompileUnit<'c, 'a>> {
    let scope = vm.new_scope_with_builtins();
    let code = vm.compile(inputs, Mode::Exec, "<inputs>".into()).unwrap();
    vm.run_code_obj(code, scope.clone())
        .unwrap_or_else(|error| {
            let mut message = String::new();
            vm.write_exception(&mut message, &error).unwrap();
            panic!("{message}")
        });
    let args = scope.globals.get_item("args", vm).unwrap();
    let args = args.downcast_ref::<PyTuple>().unwrap();
    CompileUnit::from_runtime_args(function, args.as_slice(), vm)
}

#[test]
fn binds_scalars_and_positional_only_names_without_executing_body() {
    with_vm(|vm| {
        let f = function("def f(flag, /, count, rate):\n    raise RuntimeError('must not run')\n");
        let unit = bind(vm, &f, "args = (True, 2**100, 0.25)").unwrap();
        assert_eq!(unit.variable_type("flag"), Some(&Typ::Boolean));
        assert_eq!(unit.variable_type("count"), Some(&Typ::Integral));
        assert_eq!(unit.variable_type("rate"), Some(&Typ::Fractional));
        assert_eq!(unit.variable_type("missing"), None);
    });
}

#[test]
fn preserves_tensor_dtype_rank_and_empty_dimensions() {
    with_vm(|vm| {
        let f = function("def f(x, y):\n    return x + y\n");
        let unit = bind(vm, &f, "import numpy as np\nargs = (np.zeros((0, 3), dtype=np.int16), np.array(2.0, dtype=np.float32))").unwrap();
        assert_eq!(
            unit.variable_type("x"),
            Some(&Typ::Tensor {
                dtype: DType::I16,
                shape: vec![0, 3]
            })
        );
        assert_eq!(
            unit.variable_type("y"),
            Some(&Typ::Tensor {
                dtype: DType::F32,
                shape: vec![]
            })
        );
    });
}

#[test]
fn seeds_mnist_parameters_and_keeps_them_through_lowering_entry() {
    with_vm(|vm| {
        let context = pytile::ir::create_context();
        let source = include_str!("../examples/mnist_forward.py");
        let f = function(source);
        let mut unit = bind(vm, &f, "import numpy as np\nargs = tuple(np.zeros(shape, dtype=np.float32) for shape in [(4, 784), (784, 32), (32,), (32, 10), (10,)])").unwrap();
        for (name, shape) in [
            ("x", vec![4, 784]),
            ("w1", vec![784, 32]),
            ("b1", vec![32]),
            ("w2", vec![32, 10]),
            ("b2", vec![10]),
        ] {
            assert_eq!(
                unit.variable_type(name),
                Some(&Typ::Tensor {
                    dtype: DType::F32,
                    shape
                })
            );
        }
        let before = unit.variables().clone();
        let module = parse_source(source, "mnist_forward.py").unwrap();
        let error = unit.compile_module(&context, &module).err().unwrap();
        // Tensor arguments now lower, but the np.maximum call is still unsupported.
        assert!(error
            .to_string()
            .contains("this expression isn't supported at the moment"));
        assert_eq!(unit.variables(), &before);
    });
}

#[test]
fn each_call_gets_its_own_types_and_shapes() {
    with_vm(|vm| {
        let f = function(include_str!("../examples/add.py"));
        let first = bind(
            vm,
            &f,
            "import numpy as np\nargs = (np.zeros((2, 3), dtype=np.float32), 1)",
        )
        .unwrap();
        let second = bind(
            vm,
            &f,
            "import numpy as np\nargs = (np.zeros((7, 3), dtype=np.float64), 1.0)",
        )
        .unwrap();
        assert_eq!(
            first.variable_type("x"),
            Some(&Typ::Tensor {
                dtype: DType::F32,
                shape: vec![2, 3]
            })
        );
        assert_eq!(
            second.variable_type("x"),
            Some(&Typ::Tensor {
                dtype: DType::F64,
                shape: vec![7, 3]
            })
        );
        assert_eq!(first.variable_type("y"), Some(&Typ::Integral));
        assert_eq!(second.variable_type("y"), Some(&Typ::Fractional));
    });
}

#[test]
fn rejects_bad_calls_instead_of_guessing_types() {
    with_vm(|vm| {
        let f = function("def f(x, y=1):\n    return x\n");
        for inputs in ["args = (1,)", "args = (1, 2, 3)"] {
            assert!(bind(vm, &f, inputs)
                .unwrap_err()
                .to_string()
                .contains("expected 2"));
        }
        assert!(bind(vm, &f, "args = (1, 2)").is_ok());
        let error = bind(vm, &f, "args = ('unsupported', 2)").unwrap_err();
        assert!(format!("{error:#}").contains("argument 'x'"));
        assert!(format!("{error:#}").contains("Unsupported runtime argument"));
        let error = bind(vm, &f, "import numpy as np\nargs = (np.array(['text']), 2)").unwrap_err();
        assert!(format!("{error:#}").contains("Unsupported tensor dtype"));
        for source in ["def f(*x): pass", "def f(**x): pass", "def f(*, x): pass"] {
            assert!(bind(vm, &function(source), "args = ()")
                .unwrap_err()
                .to_string()
                .contains("not supported"));
        }
    });
}
