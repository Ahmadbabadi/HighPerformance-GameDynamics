# HighPerformance-GameDynamics


## HPC Motivation

| Technique             | Goal                        |
| --------------------- | --------------------------- |
| Vectorization         | Remove Python loops         |
| JIT Compilation       | Compile hot code paths      |
| Multi-threading       | Utilize multiple CPU cores  |
| GPU Computing         | Exploit massive parallelism |
| Distributed Computing | Scale across machines       |


## Implementations

| Implementation   | Language | CPU | GPU | Parallel | Status       |
| ---------------- | -------- | --- | --- | -------- | ------------ |
| Pure Python      | Python   | ✓   | ✗   | ✗        | Stable       |
| NumPy            | Python   | ✓   | ✗   | ✗        | Stable       |
| Vectorized NumPy | Python   | ✓   | ✗   | ✗        | Stable       |
| Numba CPU        | Python   | ✓   | ✗   | ✓        | Stable       |
| Numba CUDA       | Python   | ✗   | ✓   | ✓        | Stable       |
| CuPy             | Python   | ✗   | ✓   | ✓        | Stable       |
| PyTorch CPU      | Python   | ✓   | ✗   | ✓        | Stable       |
| PyTorch CUDA     | Python   | ✗   | ✓   | ✓        | Stable       |
| Julia            | Julia    | ✓   | ✗   | ✗        | Stable       |
| Julia Parallel   | Julia    | ✓   | ✗   | ✓        | Stable       |
| Julia CUDA       | Julia    | ✗   | ✓   | ✓        | Stable       |


## Installation

### Python

```bash
pip install numpy numba cupy torch pandas dask distributed
```

### Julia

```julia
using Pkg

Pkg.add([
    "CUDA",
    "BenchmarkTools",
    "CSV",
    "DataFrames",
    "Distributed"
])
```

---
## Benchmark Results

### Hardware

| Component | Specification |
| --------- | ------------- |
| CPU       | Ryzen 5 7640hs|
| Cores     | 6             |
| RAM       | 32 Gb         |
| GPU       | L4            |
| Python    | 3.13.13       |
| Julia     | 1.12.6        |

### Runtime Comparison for lattice size = 128

| Implementation   | Runtime (s) | Speedup |
| ---------------- | ----------- | ------- |
| Python           | 72.56       | 1×      |
| NumPy            | 300.87      | 0.2×    |
| Vectorized NumPy | 2.63        | 27.6×   |
| Vectorized PyTorch| 1.2        | 60.6×   |
| Numba            | 1.35        | 53.7×   |
| Julia            | 1.59        | 45.6×   |
| Julia Parallel   | 0.46        | 159.4×  |
| CuPy             | 4.56        | 15.9×   |
| V PyTorch CUDA   | 2.59        | 28.0×   |
| Numba CUDA       | 1.12        | 64.5×   |
| Julia CUDA       | 0.16        | 457.3×  |


### Julia parallel implemention report

| Threads | Runtime (s) | Speedup | efficiency |
| ------- | ----------- | ------- | ---------- |
| 1       | 77.60       | 1×      | 1.0        |
| 2       | 44.97       | 1.7×    | 0.86       |
| 4       | 27.14       | 2.8×    | 0.71       |
| 8       | 20.45       | 3.8×    | 0.47       |

---


<p align="center">
  <img src="results/figures/size_time.png" alt="Alt text" width="500"/>
</p>

<p align="center">
  <img src="results/figures/log_size_time.png" alt="Alt text" width="500"/>
</p>