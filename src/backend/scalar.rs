//! Adapt one scalar i64 function to a single-thread CUDA kernel.

use anyhow::{ensure, Context, Result};
use melior::ir::{
    attribute::{StringAttribute, TypeAttribute},
    operation::OperationLike,
    r#type::{FunctionType, IntegerType},
    BlockLike, Module,
};

/// Keep the original function as device code and add `<name>_kernel` with an
/// extra output pointer. The caller must launch exactly one block and thread.
pub(super) fn wrap(source: &str) -> Result<String> {
    let context = crate::ir::create_context();
    let module = Module::parse(&context, source).context("Cannot parse scalar MLIR module")?;
    ensure!(module.as_operation().verify(), "Invalid scalar MLIR module");
    let function = module
        .body()
        .first_operation()
        .context("Empty MLIR module")?;
    ensure!(
        function.name().as_string_ref().as_str()? == "func.func"
            && function.next_in_block().is_none(),
        "Expected exactly one scalar func.func or a standalone gpu.module"
    );
    let signature = FunctionType::try_from(
        TypeAttribute::try_from(function.attribute("function_type")?)?.value(),
    )?;
    let i64_type = IntegerType::new(&context, 64).into();
    ensure!(
        signature.result_count() == 1 && signature.result(0)? == i64_type,
        "Scalar GPU lowering requires exactly one i64 result"
    );
    let mut parameters = Vec::new();
    let mut operands = Vec::new();
    let mut types = Vec::new();
    for index in 0..signature.input_count() {
        ensure!(
            signature.input(index)? == i64_type,
            "Scalar GPU lowering requires i64 arguments"
        );
        parameters.push(format!("%arg{index}: i64"));
        operands.push(format!("%arg{index}"));
        types.push("i64");
    }
    parameters.push("%out: memref<1xi64>".into());
    let name = StringAttribute::try_from(function.attribute("sym_name")?)?;
    let kernel = StringAttribute::new(&context, &format!("{}_kernel", name.value()));
    // Print symbol names as MLIR string attributes so quoted names remain valid.
    // The function body comes from MLIR's printer, without textual rewriting.
    let gpu = format!(
        "gpu.module @kernels {{
  {function}
  gpu.func @{kernel}({parameters}) kernel {{
    %result = func.call @{name}({operands}) : ({types}) -> i64
    %zero = arith.constant 0 : index
    memref.store %result, %out[%zero] : memref<1xi64>
    gpu.return
  }}
}}\n",
        parameters = parameters.join(", "),
        operands = operands.join(", "),
        types = types.join(", "),
    );
    Ok(gpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_wrapper_verifies_and_preserves_computation() {
        let context = crate::ir::create_context();
        let source = "module { func.func @compute(%x: i64, %y: i64) -> i64 {
            %sum = arith.addi %x, %y : i64
            return %sum : i64
        } }";
        let gpu = wrap(source).unwrap();
        let module = Module::parse(&context, &format!("module {{ {gpu} }}")).unwrap();
        assert!(module.as_operation().verify());
        assert!(gpu.contains("compute_kernel"));
        assert!(gpu.contains("arith.addi"));
        assert!(gpu.contains("memref.store %result, %out[%zero]"));
    }

    #[test]
    fn unsupported_signatures_are_rejected() {
        for source in [
            "module {}",
            "module { func.func @f() { return } }",
            "module { func.func @f(%x: f32) -> f32 { return %x : f32 } }",
            "module { func.func @f() { return } func.func @g() { return } }",
        ] {
            assert!(wrap(source).is_err());
        }
    }
}
