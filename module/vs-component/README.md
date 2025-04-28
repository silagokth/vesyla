# vs-component: Vesyla RTL generation and extension protocol

![Simplified overviex of the VS-Component RTL generation tool](https://github.com/user-attachments/assets/b84fc75f-9b29-4cf8-9c78-9b41133d6ee1)

## Introduction

- Brief overview of the project
  This tool can be used to generate RTL code from a high-level description of a DDRA2 fabric.
  The input JSON file for this high-level description has to follow a specific format, described in [docs/json_input_format.md](docs/json_input_format.md).

- Purpose and goals
  The purpose of this tool is to simplify the process of composing a DRRA2 fabric by using pre-defined components from a library.
  This library is extensible, the extension protocol is described in [docs/library_extension_protocol.md](docs/library_extension_protocol.md).

## Installation

- Prerequisites
  - Having installed the [Vesyla Suite](https://github.com/silagokth/vesyla/tree/develop?tab=readme-ov-file#compile-and-install) or compiled this Rust project with `cargo build`.
  - A library of components. Example components are provided in the [components/](components/) directory.
- Step-by-step installation guide

## Usage

- Basic usage instructions: `vs-component assemble --help`

```shell
Usage: vs-component assemble [OPTIONS] --arch <ARCH> --output <OUTPUT>

Options:
  -a, --arch <ARCH>      Architecture JSON file path
  -o, --output <OUTPUT>  Output directory path
  -d, --debug
  -h, --help
```

- Testcases can be found in the [DRRA Testcase](https://github.com/silagokth/drra-testcase) repository.
