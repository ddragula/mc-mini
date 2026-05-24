# mc-mini

Minimal neutron Monte Carlo transport prototype in C++ and CUDA.

The project compares three execution models:

* **single-thread history-based CPU transport**,
* **multi-thread history-based CPU transport**,
* **event-based CUDA transport** with particle queues.

It uses processed **ACE cross-section data** *(not included in the repository)* and currently focuses on a simple homogeneous box benchmark.

## Features

* single-thread CPU history-based neutron transport
* multi-thread CPU history-based neutron transport
* CUDA event-based transport loop
* ASCII ACE loader
* energy-dependent cross-section lookup
* elastic scattering with recoil energy loss
* absorption and leakage handling
* energy balance tally
* track-length flux tally

## Current benchmark

Default `config.json` configuration:

```text
material: natural carbon ACE data, e.g. C0.ACE / 6000.03c
density: 2.26 g/cm3
geometry: cube, x/y/z in [-5 cm, 5 cm]
source: isotropic point source at origin
source energy: 2 MeV
volume: 1000 cm3
histories: 1e6
```

Example transport performance:

```text
10M histories:
CPU single-thread:  ~2.14 s transport, ~4.7e6 histories/s
CPU multi-thread:   ~0.21 s transport, ~4.7e7 histories/s
CUDA event-based:   ~0.10 s transport, ~9.8e7 histories/s

100M histories:
CPU single-thread:  ~21.5 s transport, ~4.7e6 histories/s
CUDA event-based:   ~0.93 s transport, ~1.1e8 histories/s
```

The CPU and CUDA versions currently agree within expected Monte Carlo statistical variation for leakage, absorbed particles, mean escaped energy, collision count, energy balance and track-length flux.

Performance numbers are hardware-dependent. `histories/s` is also geometry-dependent because larger boxes usually cause more collisions per history.

## Requirements

* CMake 3.25+
* Ninja
* C++20 compiler
* CUDA Toolkit for `mc_cuda`

On Arch Linux / Arch WSL:

```bash
sudo pacman -Syu cuda
```

Useful environment setup if CUDA is installed under `/opt/cuda`:

```bash
export PATH="/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
```

Check CUDA:

```bash
nvcc --version
nvidia-smi
```

## Build

```bash
cmake --preset release
cmake --build --preset release
```

## Configuration

Benchmark inputs are read from `config.json` by default:

```json
{
  "particle_count": 1e6,
  "box_side_length": 10,
  "material_file": "data/ace/C0.ACE",
  "mass_density": 2.26
}
```

Use another config file by passing it as the first program argument, or through
the makefile `CONFIG` variable:

```bash
CONFIG=config.small.json make runh
CONFIG=config.small.json make runhmt
CONFIG=config.small.json make runc
```

Run single-thread CPU version:

```bash
./build/release/mc_history
```

Run multi-thread CPU version:

```bash
./build/release/mc_history_mt
```

Run CUDA version:

```bash
./build/release/mc_cuda
```

Makefile shortcuts:

```bash
make runh     # build and run single-thread CPU
make runhmt   # build and run multi-thread CPU
make runc     # build and run CUDA

make hst      # run existing single-thread CPU binary
make hmt      # run existing multi-thread CPU binary
make cuda     # run existing CUDA binary
```

The multi-thread CPU version uses `std::thread::hardware_concurrency()` by
default. Override it with:

```bash
MC_MINI_THREADS=8 make hmt
```

## Data files

ACE nuclear data files are not included in this repository.

Expected local path example:

```text
data/ace/C0.ACE
```

Processed nuclear data may have redistribution restrictions. Provide your own ACE files locally.

## Repository layout

```text
include/mc_mini/      shared CPU-side headers and ACE loader
src/history/          single-thread and multi-thread history-based CPU transport
src/cuda/             event-based CUDA transport prototype
```

## Limitations

This is a research prototype, not a production reactor-physics code.


## License

This project is licensed under the Apache License 2.0.

Copyright 2026 Dawid Draguła.

ACE nuclear data files are not included in this repository and are not covered
by this license. Users must provide their own ACE files and comply with the
terms of the data source they use.

If you use this project in research, benchmarks, presentations, articles,
reports, or derived public prototypes, please mention the original author and
repository. I also appreciate being contacted about substantial updates,
forks, comparisons, or publications based on this work.
