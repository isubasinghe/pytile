//! Construct real MLIR objects through Melior, without implementing Python lowering.
use anyhow::{ensure, Result};
use melior::{
    dialect::{arith, func},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType},
        Block, BlockLike, Location, Module, Region, RegionLike, Type,
    },
    Context,
};

fn build_module(context: &Context) -> Result<Module<'_>> {
    let location = Location::unknown(context);
    let module = Module::new(location);
    let tensor: Type = RankedTensorType::new(&[128], Type::float32(context), None).into();
    let block = Block::new(&[(tensor, location), (tensor, location)]);
    let sum = block.append_operation(arith::addf(
        block.argument(0)?.into(),
        block.argument(1)?.into(),
        location,
    ));
    let square = block.append_operation(arith::mulf(
        sum.result(0)?.into(),
        sum.result(0)?.into(),
        location,
    ));
    block.append_operation(func::r#return(&[square.result(0)?.into()], location));
    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "compute"),
        TypeAttribute::new(FunctionType::new(context, &[tensor, tensor], &[tensor]).into()),
        region,
        &[],
        location,
    ));
    ensure!(module.as_operation().verify(), "MLIR verification failed");
    Ok(module)
}

fn main() -> Result<()> {
    let context = pytile::ir::create_context();
    let module = build_module(&context)?;
    print!("{}", module.as_operation());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructed_module_verifies_and_round_trips() {
        let context = pytile::ir::create_context();
        let module = build_module(&context).unwrap();
        let text = module.as_operation().to_string();
        let reparsed = Module::parse(&context, &text).expect("MLIR printer output must parse");
        assert!(reparsed.as_operation().verify());
        assert!(text.contains("arith.addf"));
        assert!(text.contains("arith.mulf"));
        assert!(text.contains("tensor<128xf32>"));
    }
}
