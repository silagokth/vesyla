# DRRA-2 Programming Steps

## Define Architecture

Modify the `arch.json` file to define the architecture of the system. Use predefined resources and controllers to define new cells and compose the fabric using the defined cells.

## Create Verification Environment

Modify the `main.cpp` file to create the verification environment. You need to create three functions:

1. `init()`: Generate the stimuli as input buffer.
2. `model_l0`: Describe the behavior of the algorithm in free-style C++ format. This function will create the reference output buffer once being called.
3. `model_l1`: Describe the behavior of the algorithm by calling the simulator with specific code segment. The code segments are defined in folder `asm` as assembly language files. Their names are purely natural numbers starting from 0. You can also use free-style C++ code such as for-loops to wrap the execution of assembly code segments. For now, the free-style C++ code will not be executed on the fabric. It will just be emulated by the computer simulator.

## Write Assembly Code Segments

Put the assembly code segments in the `pasm` folder. The file names should be purely natural numbers starting from 0. Each code segment should be self contained in terms of timing. Meaning that it cannot have any part that relies on event-driving mechanism. After execution of each code segment, the global controller on the fabric is fully synchronized and ready for the next code segment.

## Vesyla template example

The following example is a simple vector multiplication of two vectors with 32 elements.
This example is implemented by DRRA fabric with 1x1 computation fabric.
The architecture is defined in `arch.json`.

The verification environment is defined in [`./run.sh`](./run.sh).
The assembly code segments are defined in the [`pasm`](./pasm/) folder.
The reference output buffer is created by the `model_l0` function defined in [`main.cpp`](./model_0/main.cpp).

The subsequent models' output buffers are compared to model 0's for verification.
