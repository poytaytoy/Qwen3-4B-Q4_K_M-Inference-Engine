{
  description = "CUDA development shell for my Inference Engine";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cuda = pkgs.cudaPackages_12;
    in {
      devShells.${system}.default = (pkgs.mkShell.override {
        stdenv = cuda.backendStdenv;
      }) {
        packages = with pkgs; [
          cmake ninja git curl cacert pkg-config
          zip unzip gnutar gnumake patch
          autoconf automake libtool m4 bison flex
          python3 perl patchelf
          cuda.cudatoolkit
        ];

        CUDAToolkit_ROOT = "${cuda.cudatoolkit}";
        CUDA_PATH = "${cuda.cudatoolkit}";
        CUDACXX = "${cuda.cudatoolkit}/bin/nvcc";
        CUDAHOSTCXX = "${cuda.backendStdenv.cc}/bin/g++";
        # Use the shell's tools rather than vcpkg's downloaded executables.
        VCPKG_FORCE_SYSTEM_BINARIES = "1";

        shellHook = ''
          # Use the host NVIDIA driver; do not bundle a different driver version.
          for driver_dir in /run/opengl-driver/lib /usr/lib/wsl/lib; do
            if [ -d "$driver_dir" ]; then
              export LD_LIBRARY_PATH="$driver_dir''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
          done
        '';
      };
    };
}
