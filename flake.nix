{
  description = "PyTile: a small Python frontend for CUDA Tile IR";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/f68407657ad294471eb4b2e65d193df26edf729f";
    cuda-tile = { url = "github:NVIDIA/cuda-tile"; flake = false; };
    llvm = { url = "github:llvm/llvm-project/d6a4b2ad0b90fad43565a25f1cbcbac6e6857dde"; flake = false; };
  };

  outputs = { self, nixpkgs, cuda-tile, llvm }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3.withPackages (p: [ p.numpy p.pytest ]);
      tileTools = pkgs.stdenv.mkDerivation {
        pname = "cuda-tile-tools";
        version = "${cuda-tile.shortRev or "dev"}";
        src = cuda-tile;
        nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
        buildInputs = [ pkgs.zlib ];
        cmakeFlags = [
          "-DCMAKE_BUILD_TYPE=Release"
          "-DCUDA_TILE_USE_LLVM_SOURCE_DIR=${llvm}"
          "-DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF"
          "-DCUDA_TILE_ENABLE_TESTING=OFF"
          "-DCUDA_TILE_ENABLE_CAPI=OFF"
          "-DLLVM_ENABLE_ASSERTIONS=OFF"
          "-DLLVM_ENABLE_TERMINFO=OFF"
          "-DLLVM_ENABLE_LIBXML2=OFF"
          "-DLLVM_ENABLE_ZSTD=OFF"
          "-DLLVM_PARALLEL_LINK_JOBS=1"
        ];
        ninjaFlags = [ "cuda-tile-opt" "cuda-tile-translate" ];
        enableParallelBuilding = true;
        installPhase = ''
          runHook preInstall
          mkdir -p $out/bin
          cp bin/cuda-tile-opt bin/cuda-tile-translate $out/bin/
          runHook postInstall
        '';
      };
      basePackages = [ python pkgs.ruff pkgs.git pkgs.gnumake pkgs.cmake pkgs.ninja ];
    in {
      packages.${system}.default = tileTools;
      devShells.${system} = {
        frontend = pkgs.mkShell { packages = basePackages; };
        default = pkgs.mkShell { packages = basePackages ++ [ tileTools ]; };
      };
    };
}
