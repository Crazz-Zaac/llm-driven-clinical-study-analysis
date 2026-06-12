{ pkgs ? import <nixpkgs> {} }:

let
  # Define a custom Poetry package with the export plugin
  poetryWithPlugins = pkgs.poetry.withPlugins (p: [
    p.poetry-plugin-export
  ]);
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    poetryWithPlugins  # Use the custom package instead of standard poetry
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

    # Auto-create venv if missing
    if [ ! -d ".venv" ]; then
      poetry config virtualenvs.in-project true
      poetry install
    fi

    # Auto-activate
    source .venv/bin/activate
  '';
}   