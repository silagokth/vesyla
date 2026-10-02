# vs-fabric: Vesyla fabric elaboration, SST and RTL generation

![Simplified overview of the VS-Fabric RTL generation tool](https://github.com/user-attachments/assets/b84fc75f-9b29-4cf8-9c78-9b41133d6ee1)

## Introduction

- Brief overview of the project
  This tool can be used to generate RTL code from a high-level description of a DRRA fabric.
  The input JSON file for this high-level description has to follow a specific format, described in [docs/json_input_format.md](docs/json_input_format.md).

- Purpose and goals
  The purpose of this tool is to simplify the process of composing a DRRA fabric by using pre-defined components from a library.
  This library is extensible, the extension protocol is described in [../vs-component/docs/library_extension_protocol.md](../vs-component/docs/library_extension_protocol.md).

## Installation

- Prerequisites
  - Having installed the [Vesyla](https://github.com/silagokth/vesyla/tree/develop?tab=readme-ov-file#compile-and-install) or compiled this Rust project with `cargo build`.
  - A library of components. Example components are provided in the [components/](components/) directory.
- Step-by-step installation guide

## Usage

- Basic usage instructions: `vs-fabric <elaborate|sst|rtl> --help`

```shell
Usage: vs-fabric elaborate [OPTIONS] --arch-json <ARCH_JSON> --output <OUTPUT>
Usage: vs-fabric sst [OPTIONS] --arch-json <ARCH_JSON> --output <OUTPUT>
Usage: vs-fabric rtl [OPTIONS] --arch-json <ARCH_JSON> --output <OUTPUT>

Options:
  -a, --arch-json <ARCH_JSON>  Architecture JSON file path
  -o, --output <OUTPUT>        Output directory path
  -d, --debug
  -h, --help
```

- `elaborate` takes the source arch.json and writes `arch/arch.json` (the
  elaborated fabric) and `isa/` into the output directory.
- `sst` and `rtl` take an elaborated arch.json, or a sized one -- the copy the
  compiler writes with the parameters it set for one program -- and write `sst/`
  and `rtl/`. Any parameter a component does not give falls back to its library
  default, then to its cell's parameters, then to the fabric's.

- Testcases can be found in the [drra-tests](https://github.com/silagokth/drra-tests) repository.
