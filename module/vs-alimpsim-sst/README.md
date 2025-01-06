# SiLago SST sim
Dependencies
---
- `cmake`
- `make`
- [SST framework](http://sst-simulator.org)

How to install SiLago SST sim?
---

### Build from scratch
```shell
cd vesyla-suite-4/module/vs-alimpsim-sst/
mkdir -p build
cd build
cmake ..
make
```

### Clean rebuild
```shell
cd vesyla-suite-4/module/vs-alimpsim-sst/
rm -rf build/*
cd build
cmake ..
make
```

Verify install:
---
Running the following command should list the different components registered during installation of the SiLago SST sim.
```shell
sst-info drra
```

Test install:
---
To run the test for the DRRA components:
```shell
sst-test-elements -w "drra*"
```