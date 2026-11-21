import argparse
import ast
import sys
from pathlib import Path

from .frontend import compile_module, parse_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a small Python subset to CUDA Tile IR")
    parser.add_argument("source", type=Path)
    parser.add_argument("--dump-ast", action="store_true", help="inspect the Python syntax tree")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        module = parse_source(args.source.read_text(), str(args.source))
        output = ast.dump(module, indent=2) if args.dump_ast else compile_module(module)
        if args.output:
            args.output.write_text(output + "\n")
        else:
            print(output)
    except (OSError, SyntaxError, NotImplementedError) as error:
        print(f"pytile: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
