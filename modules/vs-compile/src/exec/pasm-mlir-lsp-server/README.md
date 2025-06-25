# How to use the MLIR PASM LSP

1. Build vesyla as usual

   ```bash
   cd vesyla
   make
   ```

2. Rename and move the lsp server in your PATH

   ```bash
   mkdir -p ~/local/bin
   cp build/bin/pasm-mlir-lsp-server ~/local/bin/mlir-lsp-server
   export PATH=~/local/bin:$PATH
   ```
