{
  description = "A development environment for Deep Learning from Scratch (DLFS)";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Define Python and its required packages
        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.numpy
          ps.pytest
          ps.black
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];

          shellHook = ''
            echo "========================================================"
            echo "  Welcome to Deep Learning from Scratch (DLFS)!   "
            echo "  Python $(python --version).             "
            echo "========================================================"
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';
        };
      });
}
