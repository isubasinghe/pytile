"""Start here: validation, symbol tracking, and lowering are your exercises."""

import ast


def parse_source(source: str, filename: str = "<input>") -> ast.Module:
    """Parse without executing user code."""
    return ast.parse(source, filename=filename)


def compile_module(module: ast.Module) -> str:
    """Return textual CUDA Tile MLIR.

    First milestone: two array arguments and `return x + y`.
    Then add local assignments and multiplication. Reject unsupported nodes.
    Use fixed 128-element float32 tiles; defer partial tiles and broadcasting.
    """
    raise NotImplementedError(
        "Implement validation and Tile IR emission in pytile/frontend.py. "
        "Use --dump-ast to inspect your input first."
    )
