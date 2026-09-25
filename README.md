# vesyla &emsp; [![Build Status]][actions] [![Rustc Version 1.82+]][rustc]

[Build Status]: https://github.com/silagokth/vesyla/actions/workflows/ci-weekly-build.yml/badge.svg
[actions]: https://github.com/silagokth/vesyla/actions/workflows/ci-weekly-build.yml
[Rustc Version 1.82+]: https://img.shields.io/badge/rustc-1.82+-lightgray.svg?e&logo=rust&logoColor=white
[rustc]: https://blog.rust-lang.org/2024/10/17/Rust-1.82.0/

Synthesis and compilation tool suite for DRRA hardware accelerator platform.

## Installation

### Requirements

- [bender](https://github.com/pulp-platform/bender) (used by `vesyla component`)
- [minizinc](https://www.minizinc.org/) (used by `vesyla compile`)
- `g++` (used by `vesyla testcase`)
- `xdg-open` (from `xdg-utils`) (used by `vesyla show` to open the selected image)
- [graphviz](https://graphviz.org/) (`dot`) (used by `vesyla compile` to render the constraint and interconnect graphs that `vesyla show` displays)
- [gtkwave](https://gtkwave.sourceforge.net/) (used by `vesyla show wave` to open the simulation waveform `trace.vcd`)

### Download

Built packages are available for download in the [releases](https://github.com/silagokth/vesyla/releases).

### Install

- For AppImage or generic tar.gz package,
extract and copy to a location in your PATH.
Make sure to give execution permissions to the binary.

- For Debian-based systems:

   ```bash
   sudo dpkg -i pkg/vesyla-*.deb
   ```

- For RedHat-based systems:

   ```bash
   sudo rpm -i pkg/vesyla-*.rpm
   ```

## Compilation

### Dependencies

- CMake >= 3.22.1
- Clang >= 5 or GCC >= 9.0 (C++17 support)
- Flex and Bison (tested on 2.6.4 and 3.8.2, respectively)
- MLIR (from upstream [llvm/llvm-project](https://github.com/llvm/llvm-project), tested on release `22.1.8`)
  - Download the prebuilt release and point `LLVM_HOME` at it:
    ```bash
    curl -fL -o LLVM-22.1.8-Linux-X64.tar.xz \
      https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.8/LLVM-22.1.8-Linux-X64.tar.xz
    mkdir -p <path>/llvm-install
    tar -xf LLVM-22.1.8-Linux-X64.tar.xz -C <path>/llvm-install --strip-components=1
    export LLVM_HOME=<path>/llvm-install
    ```

### Build

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build . -- -j$(nproc)
   ```

### Create and install package

Packages will be generated in the `build/pkg` directory.

- For Debian-based systems:

   ```bash
   cpack -G DEB
   sudo dpkg -i pkg/vesyla-*.deb
   ```

- For RedHat-based systems:

   ```bash
   cpack -G RPM
   sudo rpm -i pkg/vesyla-*.rpm
   ```

- For generic tar.gz package:

   ```bash
   cpack -G TGZ
   ```

   To install, extract the tarball and copy to a location in your PATH.

## [Modules](./modules/)

- `testcase`: Test case infrastructure and build-in test cases for DRRA-2
- `component`: Assemble the components of vesyla,
  it requires you to specify the location of DRRA libary as enviroment variable `VESYLA_SUITE_PATH_COMPONENTS`.
  Check the repo [drra-components](https://github.com/silagokth/drra-components).
- `compile`: Compiler for DRRA-2 instruction set architecture (ISA).
  It compiles the proto-assembly code (.pasm) into binary code.
  The compiler uses the [minizinc](https://www.minizinc.org/) solver.

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
Usage: vesyla compile --arch FILE --isa FILE --pasm FILE [--output DIR] [--allow-unsafe]
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
