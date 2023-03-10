{
  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in 
  {
    devShells.x86_64-linux.default = pkgs.mkShell rec {

      buildInputs = [
        pkgs.pipenv
        pkgs.zlib
      ];

      shellHook = ''
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
      '';

    };
  };
}
