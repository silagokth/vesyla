# SiLago SST sim
Dependencies
---
- `cmake`
- `make`
- [SST framework](http://sst-simulator.org)

How to install SiLago SST sim?
---

```shell
cd vesyla-suite-4/module/vs-alimpsim-sst/
mkdir -p build
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
