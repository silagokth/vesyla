# vesyla &emsp; [![Build Status]][actions] [![Rustc Version 1.82+]][rustc]

[Build Status]: https://github.com/silagokth/vesyla/actions/workflows/ci-weekly-build.yml/badge.svg
[actions]: https://github.com/silagokth/vesyla/actions/workflows/ci-weekly-build.yml
[Rustc Version 1.82+]: https://img.shields.io/badge/rustc-1.82+-lightgray.svg?e&logo=rust&logoColor=white
[rustc]: https://blog.rust-lang.org/2024/10/17/Rust-1.82.0/

Synthesis and compilation tool suite for DRRA hardware accelerator platform.

## Requirements

- [minizinc](https://www.minizinc.org/) (used by `vesyla compile`)

## Download

Built packages are available for download in the [releases](https://github.com/silagokth/vesyla/releases).

## [Modules](./modules/)

- `testcase`: Test case infrastructure and build-in test cases for DRRA-2
- `component`: Assemble the components of vesyla,
  it requires you to specify the location of DRRA libary as enviroment variable `VESYLA_SUITE_PATH_COMPONENTS`.
  Check the repo [drra-components](https://github.com/silagokth/drra-components).
- `compile`: Compiler for DRRA-2 instruction set architecture (ISA).
  It compiles the proto-assembly code (.pasm) into binary code.
  The compiler uses the [minizinc](https://www.minizinc.org/) solver.

## Compile and Install

1. Install dependencies

   ```bash
   sh ./scripts/install_dependencies.sh
   ```

2. Build Vesyla

   ```bash
   make
   ```

3. Install Vesyla

   ```bash
   sudo make install
   ```

## Usage

### `vesyla`

```shell
Usage: vesyla [command and options]
Commands:
        compile     Compile the source code
        component   Assemble the system
        testcase    Test the system
Options:
        -h, --help     Show this help message
        -V, --version  Show version information
```

### `vesyla testcase`

```shell
Usage: vesyla testcase <COMMAND>

Commands:
  init      Initialize testcase directory
  run       Run testcase
  generate  Generate testcase scripts
  export    Export testcase
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

### `vesyla component`

```shell
Usage: vesyla component <COMMAND>

Commands:
  create         Create a new component
  assemble       Assemble the system
  validate_json  Validate JSON file
  clean          Clean the build directory
  help           Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

### `vesyla compile`

```shell
Usage: vesyla compile --arch FILE --isa FILE --pasm FILE [--output DIR]
Or
vesyla compile --arch FILE --isa FILE --cpp FILE [--output DIR]
```

## MLIR LSP Server for PASM

When compiling Vesyla, an MLIR LSP server for MLIR-PASM (Proto-Assembly MLIR dialect) files is also built.
It is located in `build/bin/pasm-mlir-lsp-server`.
You can use it with the MLIR plugin for VSCode or
any other editor that supports LSP.
To set it up in VSCode, you can use the MLIR
extension and configure the server path in the settings:

```json
{
  "settings": {
    "mlir.server_path": "build/bin/pasm-mlir-lsp-server"
  }
}
```
