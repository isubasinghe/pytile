{
  description = "PyTile: a Rust compiler from Python to MLIR and NVIDIA PTX";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/f68407657ad294471eb4b2e65d193df26edf729f";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3.withPackages (p: [ p.numpy ]);
      llvm = pkgs.llvmPackages_21;
      tools = pkgs.symlinkJoin {
        name = "pytile-llvm-tools";
        paths = [ llvm.mlir llvm.llvm ];
      };
      # Nix splits LLVM headers, libraries and executables into separate outputs.
      # mlir-sys and tblgen expect a single LLVM/MLIR installation prefix.
      sdk = pkgs.symlinkJoin {
        name = "pytile-mlir-sdk";
        paths = [ llvm.mlir llvm.mlir.dev llvm.llvm.dev llvm.llvm.lib llvm.llvm ];
        postBuild = ''
          unlink $out/bin/llvm-config
          cat > $out/bin/llvm-config <<EOF
          #!${pkgs.runtimeShell}
          for arg in "\$@"; do
            case "\$arg" in
              --libdir) echo "$out/lib"; exit 0 ;;
              --includedir) echo "$out/include"; exit 0 ;;
            esac
          done
          exec ${llvm.llvm.dev}/bin/llvm-config "\$@"
          EOF
          chmod +x $out/bin/llvm-config
        '';
      };
      basePackages = [
        pkgs.rustc pkgs.cargo pkgs.rustfmt pkgs.clippy pkgs.rust-analyzer
        pkgs.git pkgs.gnumake
      ];
      # Use Mold through Nix's compiler wrapper so Cargo and native build scripts
      # share the same linker without overriding the user's RUSTFLAGS.
      shell = extras: (pkgs.mkShell.override {
        stdenv = pkgs.stdenvAdapters.useMoldLinker pkgs.stdenv;
      }) {
        packages = basePackages ++ [ tools sdk ] ++ extras;
        nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook ];
        buildInputs = [ pkgs.zlib pkgs.libxml2 pkgs.libffi ];
        MLIR_SYS_210_PREFIX = "${sdk}";
        TABLEGEN_210_PREFIX = "${sdk}";
      };
    in {
      packages.${system}.default = tools;
      devShells.${system} = {
        frontend = shell [];
        default = shell [ python pkgs.ruff ];
      };
    };
}
