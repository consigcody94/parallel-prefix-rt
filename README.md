# Parallel Prefix Scan for Atmospheric Radiative Transfer

**First-ever parallelization of the complete two-stream adding method for atmospheric radiation.**

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

---

## Verified Benchmark Results (RTX 3060 12GB, CUDA 13.2)

| Solver | Speedup | Max Rel Error | Test Points | Status |
|:---|:---:|:---:|:---:|:---:|
| **Full flux solver** (albedo + source + flux_dn + flux_up) | **3.10x** | 6.3e-07 | 132,096 | 15/15 stress tests PASS |
| **Albedo-only scan** | **3.97x** | 7.76e-07 | 132,096 | Verified |
| FP64 → FP32 (memory bandwidth) | **7.96x** | — | — | Verified |
| Fast exp() (CPU only) | **2.55x** | 3.13e-16 | 10,000,000 | Verified |

### Stress Test Coverage

All 15 regimes tested, all passed (0 NaN, 0 Inf, 0 negative fluxes):

| Regime | Description | Max Error |
|:---|:---|:---:|
| Clear-sky | R<0.1, T>0.8 | 4.7e-07 |
| Thick clouds | R<0.7, T>0.1 | 6.4e-07 |
| Near-conservative | R~0.47, T~0.47 | 1.3e-06 |
| Transparent | R~0, T~1 | 6.0e-07 |
| Shortwave (inc_flux=340 W/m²) | Non-zero TOA flux | 1.1e-06 |
| Zero surface albedo | Edge case | 3.1e-07 |
| High surface albedo (0.9) | Ice/snow | 3.5e-07 |
| Perfect reflector (1.0) | Extreme edge | 4.4e-07 |
| 4 layers | Minimum size | 1.1e-07 |
| 16 layers | Small | 1.9e-07 |
| 32 layers | Medium | 2.5e-07 |
| 64 layers | Standard | 3.2e-07 |
| 256 layers | Large | 5.4e-07 |
| Mixed cloud/clear + SW | Realistic profile | 5.4e-07 |
| Single opaque layer (R=0.8, T=0.05) | Extreme embedded | 5.3e-07 |

---

## What Is This?

The "adding method" in atmospheric radiation solvers computes fluxes through a vertical stack of atmospheric layers. Every weather model (GFS, ECMWF IFS, ICON, etc.) has this as a sequential bottleneck — each layer depends on the one below/above it.

We show this recurrence is parallelizable using three associative scans:

### The Three Scans

| Pass | Recurrence | Associative Operator | Scan Direction |
|:---|:---|:---|:---|
| **1. Albedo** | `alb[i] = R + T²·alb[i+1]/(1-R·alb[i+1])` | 2×2 Möbius matrix multiply | Bottom-up (suffix) |
| **2. Source** | `src[i] = A·src[i+1] + B` (affine in src) | Tuple compose: `(a₂·a₁, a₂·b₁+b₂)` | Bottom-up (suffix) |
| **3. Flux_dn** | `fdn[i+1] = C·fdn[i] + D` (affine in flux) | Same tuple compose | Top-down (prefix) |
| **4. Flux_up** | `fup[i] = alb[i]·fdn[i] + src[i]` | Pointwise (trivially parallel) | — |

**Sequential depth: O(3 log₂ N) instead of O(3N).** For 128 layers: 21 steps instead of 384.

### Mathematical Lineage

| Who | What | Year |
|:---|:---|:---:|
| Grant & Hunt | Proved RT layer operators form a semigroup (associative) | 1969 |
| Blelloch | Parallel prefix scan for any associative operation | 1990 |
| Martin & Cundy | Parallelized affine recurrences on GPU (9x speedup) | 2018 |
| Gu & Dao (Mamba) | Same scan deployed at scale in language models | 2023 |
| **This work** | **First application to atmospheric radiative transfer** | **2026** |

---

## Build & Run

```bash
# Requires NVIDIA GPU + CUDA toolkit
cd benchmarks/cuda

# Full flux solver benchmark
nvcc -O3 -arch=sm_86 full_flux_parallel_scan.cu -o full_flux_scan
./full_flux_scan

# Stress test (15 edge-case regimes)
nvcc -O3 -arch=sm_86 stress_test_full_scan.cu -o stress_test
./stress_test
```

Adjust `-arch=sm_86` for your GPU (sm_75 for Turing, sm_80 for Ampere A100, sm_89 for Ada, sm_90 for Hopper).

---

## What's NOT Claimed

- The tensor compression approach (Tucker decomposition of k-tables) was tested and **failed flux validation**. The Frobenius norm error was low but exp(-τ) amplifies errors beyond operational tolerance. See [issue #394](https://github.com/earth-system-radiation/rte-rrtmgp/issues/394) for the correction.
- Fast exp() provides **no speedup on GPU** — the hardware SFU already handles it. It's a CPU-only optimization.
- The WW3 DIA GPU kernel was tested with simplified index arrays, not full WW3 spectral addressing. The 16.8x is an upper bound.
- Integration testing within a full coupled model (GFS, ICON, etc.) has not been done. The upstream rte-rrtmgp unit tests all pass with zero regressions.

---

## References

1. Grant, I.P. and Hunt, G.E. (1969). "Discrete space theory of radiative transfer." *Proc. R. Soc. A*, 313, 183-197.
2. Blelloch, G.E. (1990). "Prefix Sums and Their Applications." CMU-CS-90-190.
3. Pincus, R., Mlawer, E.J., and Delamere, J.S. (2019). "Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models." *JAMES*, 11, 3085-3098.
4. Martin, E. and Cundy, C. (2018). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length." ICLR.
5. Gu, A. and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752.
6. Ukkonen, P. and Hogan, R.J. (2024). "Twelve Times Faster yet Accurate: A New State-Of-The-Art in Radiation Schemes." *JAMES*.

## License

BSD-3-Clause (same as rte-rrtmgp)
