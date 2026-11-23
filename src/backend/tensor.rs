//! Static float32 tensor operations -> separate, ordered CUDA kernels.
//!
//! One thread owns each output element. Matmul uses a serial dot product per
//! thread; launch boundaries synchronize dependent operations. Every tensor
//! result has separate storage, preserving SSA values without in-place aliasing.

use anyhow::{bail, ensure, Context as _, Result};
use melior::{
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType, ShapedTypeLike},
        Attribute, BlockLike, Module, OperationRef, RegionLike, Type, TypeLike, Value, ValueLike,
    },
    Context,
};
use serde::Serialize;

#[derive(Serialize)]
pub(super) struct Manifest {
    /// Detect accidentally pairing a manifest with a different PTX artifact.
    pub ptx_fnv1a64: String,
    version: u32,
    function: String,
    dtype: &'static str,
    /// Buffer IDs of explicit function arguments, in source order.
    inputs: Vec<usize>,
    /// Shapes of all allocations (arguments and intermediate results).
    buffers: Vec<Vec<usize>>,
    output: usize,
    kernels: Vec<Kernel>,
}

#[derive(Serialize)]
struct Kernel {
    name: String,
    buffers: Vec<usize>,
    grid: [usize; 3],
    block: [usize; 3],
}

pub(super) struct Program {
    pub gpu: String,
    pub manifest: Manifest,
}

#[derive(Clone)]
enum Binding {
    Buffer(usize),
    Empty,
    Scalar(String),
}

fn shape(context: &Context, ty: Type<'_>) -> Result<Vec<usize>> {
    let tensor = RankedTensorType::try_from(ty)?;
    ensure!(
        tensor.element() == Type::float32(context),
        "GPU tensors must have dtype float32"
    );
    let dims = (0..tensor.rank())
        .map(|i| tensor.dim_size(i))
        .collect::<Result<Vec<_>, _>>()?;
    ensure!(
        !dims.is_empty() && dims.iter().all(|&d| d > 0 && d <= i64::MAX as usize),
        "GPU tensors currently require positive static dimensions, got {dims:?}"
    );
    let elements = dims
        .iter()
        .try_fold(1_usize, |n, &d| n.checked_mul(d))
        .context("Tensor size overflow")?;
    ensure!(elements <= i64::MAX as usize / 4, "Tensor is too large");
    Ok(dims)
}

fn memref(shape: &[usize]) -> String {
    format!(
        "memref<{}xf32>",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn binding<'c, 'a>(bindings: &[(Value<'c, 'a>, Binding)], value: Value<'c, 'a>) -> Result<Binding> {
    bindings
        .iter()
        .find(|(key, _)| *key == value)
        .map(|(_, value)| value.clone())
        .context("Unsupported tensor dataflow")
}

fn scalar<'c, 'a>(bindings: &[(Value<'c, 'a>, Binding)], value: Value<'c, 'a>) -> Result<String> {
    match binding(bindings, value)? {
        Binding::Scalar(value) => Ok(value),
        _ => bail!("Expected a scalar operand"),
    }
}

fn buffer<'c, 'a>(bindings: &[(Value<'c, 'a>, Binding)], value: Value<'c, 'a>) -> Result<usize> {
    match binding(bindings, value)? {
        Binding::Buffer(id) => Ok(id),
        _ => bail!("GPU operation reads an uninitialized or unsupported tensor"),
    }
}

/// Copy supported scalar arithmetic into a kernel without changing its operands.
fn scalar_operation<'c, 'a>(
    op: OperationRef<'c, 'a>,
    bindings: &mut Vec<(Value<'c, 'a>, Binding)>,
    code: &mut String,
    next: &mut usize,
) -> Result<()> {
    ensure!(
        op.result_count() == 1 && op.region_count() == 0,
        "Unsupported scalar GPU operation"
    );
    let ty = op.result(0)?.r#type().to_string();
    ensure!(
        ty == "f32" || ty == "f64",
        "GPU scalar expressions must be floats"
    );
    let name = op.name().as_string_ref().as_str()?.to_owned();
    let value = format!("%s{next}");
    *next += 1;
    let instruction = match name.as_str() {
        "arith.constant" => format!("arith.constant {}", op.attribute("value")?),
        "arith.truncf" | "arith.extf" => format!(
            "{name} {} : {} to {ty}",
            scalar(bindings, op.operand(0)?)?,
            op.operand(0)?.r#type()
        ),
        "arith.negf" => format!("{name} {} : {ty}", scalar(bindings, op.operand(0)?)?),
        "arith.addf" | "arith.mulf" | "arith.maximumf" => format!(
            "{name} {}, {} : {ty}",
            scalar(bindings, op.operand(0)?)?,
            scalar(bindings, op.operand(1)?)?
        ),
        _ => bail!("Unsupported scalar GPU operation: {name}"),
    };
    code.push_str(&format!("{value} = {instruction}\n"));
    bindings.push((op.result(0)?.into(), Binding::Scalar(value)));
    Ok(())
}

fn add_buffer(ids: &mut Vec<usize>, id: usize) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

fn load(id: usize, indices: &[String], shapes: &[Vec<usize>]) -> String {
    format!(
        "memref.load %b{id}[{}] : {}",
        indices.join(", "),
        memref(&shapes[id])
    )
}

/// Returns None for scalar functions, which use the existing scalar adapter.
pub(super) fn lower(source: &str) -> Result<Option<Program>> {
    let context = crate::ir::create_context();
    let module = Module::parse(&context, source).context("Cannot parse tensor MLIR")?;
    let function = module
        .body()
        .first_operation()
        .context("Empty MLIR module")?;
    ensure!(
        function.name().as_string_ref().as_str()? == "func.func"
            && function.next_in_block().is_none(),
        "Expected exactly one function or a standalone gpu.module"
    );
    let signature = FunctionType::try_from(
        TypeAttribute::try_from(function.attribute("function_type")?)?.value(),
    )?;
    if signature.result_count() != 1 || !signature.result(0)?.is_ranked_tensor() {
        return Ok(None);
    }
    ensure!(module.as_operation().verify(), "Invalid tensor MLIR module");
    let function_name = StringAttribute::try_from(function.attribute("sym_name")?)?
        .value()
        .to_owned();
    let body = function
        .region(0)?
        .first_block()
        .context("Missing function body")?;
    ensure!(
        body.next_in_region().is_none(),
        "Tensor GPU lowering requires one function block"
    );
    let mut manifest = Manifest {
        ptx_fnv1a64: String::new(),
        version: 1,
        function: function_name,
        dtype: "float32",
        inputs: vec![],
        buffers: vec![],
        output: 0,
        kernels: vec![],
    };
    let mut bindings = Vec::new();
    for i in 0..signature.input_count() {
        manifest.buffers.push(shape(&context, signature.input(i)?)?);
        manifest.inputs.push(i);
        bindings.push((body.argument(i)?.into(), Binding::Buffer(i)));
    }
    let mut gpu = String::from("gpu.module @kernels {\n");
    let mut scalars = String::new();
    let mut next_scalar = 0;
    let mut operation = body.first_operation();
    let mut returned = false;
    while let Some(op) = operation {
        operation = op.next_in_block();
        let name = op.name().as_string_ref().as_str()?.to_owned();
        if name == "func.return" {
            ensure!(op.operand_count() == 1, "Expected one tensor return value");
            manifest.output = buffer(&bindings, op.operand(0)?)?;
            returned = true;
            continue;
        }
        if name == "tensor.empty" {
            shape(&context, op.result(0)?.r#type())?;
            bindings.push((op.result(0)?.into(), Binding::Empty));
            continue;
        }
        if name.starts_with("arith.") {
            scalar_operation(op, &mut bindings, &mut scalars, &mut next_scalar)?;
            continue;
        }
        ensure!(
            op.result_count() == 1,
            "GPU operation must produce one tensor"
        );
        let output_shape = shape(&context, op.result(0)?.r#type())?;
        let output = manifest.buffers.len();
        manifest.buffers.push(output_shape.clone());
        let rank = output_shape.len();
        let indices = (0..rank).map(|i| format!("%i{i}")).collect::<Vec<_>>();
        let mut ids = Vec::new();
        let mut code = String::new();
        match name.as_str() {
            "linalg.fill" => {
                ensure!(op.operand_count() == 2, "Unsupported fill operands");
                let value = scalar(&bindings, op.operand(0)?)?;
                code.push_str(&format!(
                    "memref.store {value}, %b{output}[{}] : {}\n",
                    indices.join(", "),
                    memref(&output_shape)
                ));
            }
            "linalg.matmul" => {
                let a = buffer(&bindings, op.operand(0)?)?;
                let b = buffer(&bindings, op.operand(1)?)?;
                let init = buffer(&bindings, op.operand(2)?)?;
                ensure!(
                    rank == 2 && manifest.buffers[a].len() == 2 && manifest.buffers[b].len() == 2,
                    "GPU matmul requires rank-2 inputs and output"
                );
                let expected_maps = Attribute::parse(&context, "[affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>, affine_map<(m, n, k) -> (m, n)>]").unwrap();
                ensure!(
                    op.attribute("indexing_maps")? == expected_maps,
                    "Only standard matmul indexing is supported"
                );
                for id in [a, b, init] {
                    add_buffer(&mut ids, id);
                }
                let k = manifest.buffers[a][1];
                code.push_str(&format!("%initial = {}\n%k_end = arith.constant {k} : index\n%dot = scf.for %k = %zero to %k_end step %one iter_args(%acc = %initial) -> (f32) {{\n%a = {}\n%b = {}\n%product = arith.mulf %a, %b : f32\n%sum = arith.addf %acc, %product : f32\nscf.yield %sum : f32\n}}\nmemref.store %dot, %b{output}[%i0, %i1] : {}\n",
                    load(init, &indices, &manifest.buffers),
                    load(a, &["%i0".into(), "%k".into()], &manifest.buffers),
                    load(b, &["%k".into(), "%i1".into()], &manifest.buffers), memref(&output_shape)));
            }
            "linalg.generic" => {
                let region = op.region(0)?;
                let region_body = region.first_block().context("Missing generic body")?;
                ensure!(
                    region_body.next_in_region().is_none()
                        && region_body.argument_count() == op.operand_count(),
                    "Unsupported generic region"
                );
                let dims = (0..rank)
                    .map(|i| format!("d{i}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut maps = Vec::new();
                let mut local = bindings.clone();
                for i in 0..op.operand_count() {
                    let operand = op.operand(i)?;
                    let is_output = i + 1 == op.operand_count();
                    let shape = if operand.r#type().is_ranked_tensor() {
                        shape(&context, operand.r#type())?
                    } else {
                        vec![]
                    };
                    ensure!(shape.len() <= rank, "Unsupported generic indexing rank");
                    let offset = rank - shape.len();
                    let mut exprs = Vec::new();
                    let mut loads = Vec::new();
                    for (axis, &size) in shape.iter().enumerate() {
                        ensure!(
                            size == 1 || size == output_shape[offset + axis],
                            "Invalid broadcast shape"
                        );
                        if size == 1 && !is_output {
                            exprs.push("0".to_owned());
                            loads.push("%zero".to_owned());
                        } else {
                            exprs.push(format!("d{}", offset + axis));
                            loads.push(format!("%i{}", offset + axis));
                        }
                    }
                    maps.push(format!("affine_map<({dims}) -> ({})>", exprs.join(", ")));
                    let arg: Value = region_body.argument(i)?.into();
                    match binding(&bindings, operand)? {
                        Binding::Scalar(value) => local.push((arg, Binding::Scalar(value))),
                        Binding::Buffer(id) => {
                            add_buffer(&mut ids, id);
                            let value = format!("%in{i}");
                            code.push_str(&format!(
                                "{value} = {}\n",
                                load(id, &loads, &manifest.buffers)
                            ));
                            local.push((arg, Binding::Scalar(value)));
                        }
                        Binding::Empty => {} // Using this argument below produces an error.
                    }
                }
                let maps = Attribute::parse(&context, &format!("[{}]", maps.join(", "))).unwrap();
                let iterators = Attribute::parse(
                    &context,
                    &format!(
                        "[{}]",
                        vec!["#linalg.iterator_type<parallel>"; rank].join(", ")
                    ),
                )
                .unwrap();
                ensure!(
                    op.attribute("indexing_maps")? == maps
                        && op.attribute("iterator_types")? == iterators,
                    "GPU generic lowering supports pointwise broadcasting only"
                );
                let mut inner = region_body.first_operation();
                while let Some(instruction) = inner {
                    inner = instruction.next_in_block();
                    if instruction.name().as_string_ref().as_str()? == "linalg.yield" {
                        let value = scalar(&local, instruction.operand(0)?)?;
                        code.push_str(&format!(
                            "memref.store {value}, %b{output}[{}] : {}\n",
                            indices.join(", "),
                            memref(&output_shape)
                        ));
                    } else {
                        scalar_operation(instruction, &mut local, &mut code, &mut next_scalar)?;
                    }
                }
            }
            _ => bail!("Unsupported tensor GPU operation: {name}"),
        }
        add_buffer(&mut ids, output);
        let elements = output_shape.iter().product::<usize>();
        let blocks = elements.div_ceil(128);
        ensure!(blocks <= i32::MAX as usize, "GPU launch grid is too large");
        let kernel_name = format!("pytile_kernel_{}", manifest.kernels.len());
        let parameters = ids
            .iter()
            .map(|&id| format!("%b{id}: {}", memref(&manifest.buffers[id])))
            .collect::<Vec<_>>()
            .join(", ");
        gpu.push_str(&format!("gpu.func @{kernel_name}({parameters}) kernel {{\n{scalars}\n%zero = arith.constant 0 : index\n%one = arith.constant 1 : index\n%bid = gpu.block_id x\n%bdim = gpu.block_dim x\n%tid = gpu.thread_id x\n%base = arith.muli %bid, %bdim : index\n%linear = arith.addi %base, %tid : index\n%size = arith.constant {elements} : index\n%valid = arith.cmpi ult, %linear, %size : index\nscf.if %valid {{\n"));
        for (axis, &size) in output_shape.iter().enumerate() {
            let stride = output_shape[axis + 1..].iter().product::<usize>();
            gpu.push_str(&format!("%stride{axis} = arith.constant {stride} : index\n%extent{axis} = arith.constant {size} : index\n%q{axis} = arith.divui %linear, %stride{axis} : index\n%i{axis} = arith.remui %q{axis}, %extent{axis} : index\n"));
        }
        gpu.push_str(&format!("{code}\n}}\ngpu.return\n}}\n"));
        manifest.kernels.push(Kernel {
            name: kernel_name,
            buffers: ids,
            grid: [blocks, 1, 1],
            block: [128, 1, 1],
        });
        bindings.push((op.result(0)?.into(), Binding::Buffer(output)));
    }
    ensure!(
        returned && !manifest.kernels.is_empty(),
        "Tensor GPU lowering requires a computed tensor return"
    );
    gpu.push_str("}\n");
    let verified = Module::parse(&context, &format!("module {{ {gpu} }}"))
        .context("Generated GPU module failed verification")?;
    ensure!(
        verified.as_operation().verify(),
        "Invalid generated GPU module"
    );
    Ok(Some(Program { gpu, manifest }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend::parse_source, runtime::unit_from_expression};
    use rustpython_parser::ast;

    #[test]
    fn branching_tensor_values_keep_separate_buffers() {
        let context = crate::ir::create_context();
        let source =
            "def compute(x, b):\n    y = x + b\n    z = np.maximum(y, 0.0)\n    return y + z\n";
        let ast = parse_source(source, "test.py").unwrap();
        let ast::Stmt::FunctionDef(function) = &ast[0] else {
            panic!("expected function")
        };
        let mut unit = unit_from_expression(
            function,
            "(np.zeros((3, 2), dtype=np.float32), np.zeros(2, dtype=np.float32))",
        )
        .unwrap();
        let module = unit.compile_module(&context, &ast).unwrap();
        let program = lower(&module.as_operation().to_string()).unwrap().unwrap();
        assert_eq!(program.manifest.inputs, [0, 1]);
        assert_eq!(program.manifest.buffers.len(), 5);
        assert_eq!(program.manifest.kernels.len(), 3);
        // y must remain available after computing z, with independent storage.
        assert_eq!(program.manifest.kernels[2].buffers, [2, 3, 4]);
        assert_eq!(program.manifest.output, 4);
        assert!(program.gpu.contains("arith.cmpi ult"));
    }

    #[test]
    fn unsupported_tensor_shapes_and_dtypes_fail_before_emission() {
        for (ty, error) in [
            ("tensor<?xf32>", "positive static dimensions"),
            ("tensor<0xf32>", "positive static dimensions"),
            ("tensor<2xf64>", "dtype float32"),
        ] {
            let source =
                format!("module {{ func.func @f(%x: {ty}) -> {ty} {{ return %x : {ty} }} }}");
            let message = match lower(&source) {
                Err(error) => error.to_string(),
                Ok(_) => panic!("expected a diagnostic"),
            };
            assert!(message.contains(error), "{message}");
        }
    }
}
