//! MLIR context setup. All IR objects and builders come directly from Melior.
use melior::{dialect::DialectRegistry, utility::register_all_dialects, Context};

pub fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context
}
