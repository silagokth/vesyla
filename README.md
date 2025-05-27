# vesyla &emsp; [![Build Status]][actions] [![Rustc Version 1.82+]][rustc] [![Python Version 3.6+]][python]

[Build Status]: https://img.shields.io/github/actions/workflow/status/silagokth/vesyla/ci-draft-release.yml
[actions]: https://github.com/serde-rs/json/actions?query=branch%3Amaster
[Rustc Version 1.82+]: https://img.shields.io/badge/rustc-1.82+-lightgray.svg?e&logo=rust&logoColor=white
[rustc]: https://blog.rust-lang.org/2024/10/17/Rust-1.82.0/
[Python Version 3.6+]: https://img.shields.io/badge/python-3.6+-lightgray.svg?e&logo=python&logoColor=white
[python]: https://www.python.org/downloads/release/python-360/

Tool suite for DRRA-2 hardware accelerator platform.

## Projects

- vs-manas: Low level assembler for DRRA-2 ISA
- vs-testcase: Test case infrastructure and build-in test cases for DRRA-2
- vs-entry: Entry point for vesyla
- vs-schedule: Scheduler for DRRA-2 instruction. It will be integrated into the future compiler
- vs-component: Assemble the components of vesyla, it requires you to specify the location of DRRA libary as enviroment variable `VESYLA_SUITE_PATH_COMPONENTS`. Check the repo [drra-components](https://github.com/silagokth/drra-components).

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
