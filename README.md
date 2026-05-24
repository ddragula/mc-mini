# mc-mini

Minimal neutron Monte Carlo transport prototype in C++ and CUDA.

The project compares two execution models:

* **history-based CPU transport**,
* **event-based CUDA transport** with particle queues.

It uses processed **ACE cross-section data** *(not included in the repository)* and currently focuses on a simple homogeneous box benchmark.

## Features

* CPU history-based neutron transport
* CUDA event-based transport loop
* ASCII ACE loader
* energy-dependent cross-section lookup
* elastic scattering with recoil energy loss
* absorption and leakage handling
* energy balance tally
* track-length flux tally

## Current benchmark

Configuration:

```text
material: natural carbon ACE data, e.g. C0.ACE / 6000.03c
density: 2.26 g/cm3
geometry: cube, x/y/z in [-5 cm, 5 cm]
source: isotropic point source at origin
source energy: 2 MeV
volume: 1000 cm3
```

Example transport performance:

```text
10M histories:
CPU history-based:  ~2.16 s transport, ~4.6e6 histories/s
CUDA event-based:   ~0.10 s transport, ~9.8e7 histories/s
speedup:            ~21x

100M histories:
CPU history-based:  ~21.5 s transport, ~4.7e6 histories/s
CUDA event-based:   ~0.93 s transport, ~1.1e8 histories/s
speedup:            ~23x
```

The CPU and CUDA versions currently agree within expected Monte Carlo statistical variation for leakage, absorbed particles, mean escaped energy, collision count, energy balance and track-length flux.

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

Run CPU version:

```bash
./build/release/mc_history
```

Run CUDA version:

```bash
./build/release/mc_cuda
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
src/history/          history-based CPU transport
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
