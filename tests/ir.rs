use melior::ir::{operation::OperationLike, Module};

#[test]
fn high_level_reference_parses_and_verifies_in_mlir() {
    let context = pytile::ir::create_context();
    let module = Module::parse(&context, include_str!("../examples/high_level.mlir")).unwrap();
    assert!(module.as_operation().verify());
}

#[test]
fn mlir_rejects_invalid_return_type() {
    let context = pytile::ir::create_context();
    let module = Module::parse(
        &context,
        "module { func.func @bad(%x: f32) -> i32 { func.return %x : f32 } }",
    );
    assert!(module.is_none());
}
