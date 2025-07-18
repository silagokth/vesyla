# vesyla &emsp; [![Build Status]][actions] [![Rustc Version 1.82+]][rustc] [![Python Version 3.6+]][python]

[Build Status]: https://github.com/silagokth/vesyla/actions/workflows/ci-weekly-build.yml/badge.svg
[actions]: https://github.com/silagokth/vesyla/actions/workflows/ci-pr.yml?query=branch%3Amaster
[Rustc Version 1.82+]: https://img.shields.io/badge/rustc-1.82+-lightgray.svg?e&logo=rust&logoColor=white
[rustc]: https://blog.rust-lang.org/2024/10/17/Rust-1.82.0/
[Python Version 3.6+]: https://img.shields.io/badge/python-3.6+-lightgray.svg?e&logo=python&logoColor=white
[python]: https://www.python.org/downloads/release/python-360/

Tool suite for DRRA-2 hardware accelerator platform.

## Projects

- vs-testcase: Test case infrastructure and build-in test cases for DRRA-2
- vs-component: Assemble the components of vesyla,
  it requires you to specify the location of DRRA libary as enviroment variable `VESYLA_SUITE_PATH_COMPONENTS`.
  Check the repo [drra-components](https://github.com/silagokth/drra-components).
- vs-compile: Compiler for DRRA-2 instruction set architecture (ISA).
  It compiles the proto-assembly code (.pasm) into binary code.
  The compiler uses the [minizinc](https://www.minizinc.org/) solver.

- vs-manas (**deprecated**): Low level assembler for DRRA-2 ISA
  This project is deprecated, it will be integrated into [vs-compile](./modules/vs-compile).
- vs-schedule (**deprecated**): Scheduler for DRRA-2 instruction.
  This project is deprecated, it will be integrated into [vs-compile](./modules/vs-compile).

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

### Requirements

- [minizinc](https://www.minizinc.org/)
- [drra-components](https://github.com/silagokth/drra-components): v2.6.0 or later

### Commands

#### `vesyla`

```shell
Usage: vesyla [command and options]
Commands:
        compile     Compile the source code
        component   Assemble the system
        manas       Validate JSON file
        schedule    Clean the build directory
        testcase    Test the system
Options:
        -h, --help     Show this help message
        -V, --version  Show version information
```

#### `vs-testcase`

```shell
Usage: vs-testcase <COMMAND>

Commands:
  init      Initialize testcase directory
  run       Run testcase
  generate  Generate testcase scripts
  export    Export testcase
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

#### `vs-component`

```shell
Usage: vs-component <COMMAND>

Commands:
  assemble       Assemble the system
  validate_json  Validate JSON file
  clean          Clean the build directory
  help           Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

#### `vs-compile`

```shell
Usage: vs-compile --arch FILE --isa FILE --pasm FILE [--output DIR]
Or
vs-compile --arch FILE --isa FILE --cpp FILE [--output DIR]
```

## MLIR LSP Server for PASM

Vesyla also compile a MLIR LSP server for PASM (Proto-Assembly) files. It is located
in `build/bin/pasm-mlir-lsp-server`. You can use it with the MLIR plugin for VSCode or
any other editor that supports LSP. To set it up in VSCode, you can use the MLIR
extension and configure the server path in the settings:

```json
{
  "settings": {
    "mlir.server_path": "build/bin/pasm-mlir-lsp-server"
  }
}
```
