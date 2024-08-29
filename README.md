# vesyla-suite

Tool suite for DRRA-2 hardware accelerator platform.

## Projects:

- vs-manas: Low level assembler for DRRA-2 ISA
- vs-alimpsim: Simulator for DRRA-2
- vs-testcase: Test case infrastructure and build-in test cases for DRRA-2
- vs-entry: Entry point for vesyla-suite
- vs-schedule: Scheduler for DRRA-2 instruction. It will be integrated into the future compiler
- vs-archvis: Visualization of architecture

## Compile and Install

- install tool, library and python packages according to requirements.txt
- run *make_appimage.sh*, and wait it generate the image called "vesyla-suite"
- copy it to any directory that is in your *PATH* environment.
- now you can use the *vesyla-suite* command in any working directory.
