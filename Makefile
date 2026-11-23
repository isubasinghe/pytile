.PHONY: help test test-backend lint ast ir verify-ir doctor toolchain smoke gpu-smoke

help:
	@echo 'nix develop; make ast | ir | verify-ir | test | test-backend | lint | gpu-smoke'
	@echo 'Compiler: Rust. Input language: Python. Target: NVIDIA sm_75.'

test:
	cargo test --locked --all-targets

test-backend:
	cargo test --locked --test backend -- --ignored

lint:
	cargo fmt --check
	cargo clippy --locked --all-targets -- -D warnings
	ruff check scripts examples

ast:
	cargo run --locked --quiet -- ast examples/square_sum.py

ir:
	mkdir -p build
	cargo run --locked --quiet --example ir_builder > build/high_level.mlir

verify-ir: ir
	mlir-opt build/high_level.mlir -o build/high_level.verified.mlir

doctor:
	python3 scripts/doctor.py

toolchain:
	nix build .

smoke:
	cargo run --locked --quiet -- ptx examples/smoke.mlir -o build/smoke.ptx

gpu-smoke: smoke
	python3 scripts/run_gpu.py build/smoke.ptx

.PHONY: matrix-demo mnist-demo mnist-forward mnist-test

matrix-demo:
	python3 examples/matrix_scores.py

mnist-demo:
	python3 examples/run_mnist.py

mnist-forward:
	python3 examples/run_mnist.py --forward-only --batch-size 32

mnist-test:
	python3 -m unittest discover -s examples -p test_mnist_mlp.py
