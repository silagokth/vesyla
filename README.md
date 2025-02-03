# vesyla-suite

Tool suite for DRRA-2 hardware accelerator platform.

## Projects:

- vs-manas: Low level assembler for DRRA-2 ISA
- vs-alimpsim: Simulator for DRRA-2
- vs-testcase: Test case infrastructure and build-in test cases for DRRA-2
- vs-entry: Entry point for vesyla-suite
- vs-schedule: Scheduler for DRRA-2 instruction. It will be integrated into the future compiler
- vs-archvis: Visualization of architecture
- vs-component: Assemble the components of vesyla-suite, it requires you to specify the location of DRRA libary as enviroment variable `VESYLA_SUITE_PATH_COMPONENTS`. Check the repo [drra-component-library](https://github.com/silagokth/drra-component-library).

## Compile and Install

- run _install_dependency.sh_ to install tool, library and python packages according to requirements.txt
- run _make_appimage.sh_, and wait it generate the image called "vesyla-suite"
- copy it to any directory that is in your _PATH_ environment.
- now you can use the _vesyla-suite_ command in any working directory.
