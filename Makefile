.PHONY: help test lint ast doctor toolchain smoke

help:
	@echo 'Enter nix develop, then: make ast | test | lint | doctor | smoke'
	@echo 'make toolchain builds the pinned CUDA Tile tools (first build takes time).'

test:
	python3 -m pytest -q

lint:
	ruff check .

ast:
	python3 -m pytile examples/square_sum.py --dump-ast

doctor:
	python3 scripts/doctor.py

toolchain:
	nix build --cores 8 .

smoke:
	mkdir -p build
	cuda-tile-opt examples/smoke.mlir -o build/smoke.verified.mlir
	cuda-tile-translate examples/smoke.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o build/smoke.tilebc
	test -s build/smoke.tilebc
