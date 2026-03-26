# Parallel Prefix Scan for Atmospheric Radiative Transfer

**The first application of parallel prefix scan to atmospheric radiative transfer vertical transport — a 50-year gap between theory and practice, closed.**

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

---

## Benchmark Results (RTX 3060, CUDA 13.2)

| Optimization | Speedup | Max Error | Platform | Status |
|:---|:---:|:---:|:---:|:---:|
| **Parallel prefix scan** (adding method) | **4.73x** | 7.76e-07 | GPU | Verified |
| **FP64 to FP32** (precision reduction) | **7.98x** | ~0.001 W/m2 | GPU | Verified |
| **Fast exp()** (range-reduced Horner polynomial) | **2.55x** | 3.15e-16 | CPU | Verified |

All results independently reproducible. Zero accuracy failures across 132,096 GPU test points and 10,000,000 CPU test points.

---

## What Is This?

Radiation schemes are the most expensive physics component in weather models (30-50% of runtime). The vertical transport computation — the "adding method" — has been **inherently sequential** in every weather model ever built. Each atmospheric layer depends on the one below it, creating an O(N) chain that can't be parallelized on GPUs.

**Except it can.** The mathematics was proven in 1969.

### The Key Insight

The adding method's recurrence:

```
albedo[i] = R[i] + T[i]^2 * albedo[i+1] / (1 - R[i] * albedo[i+1])
```

is a **Mobius (linear fractional) transformation**. Mobius transformations compose via 2x2 matrix multiplication, which is **associative**. Associativity is the only property needed for [parallel prefix scan](https://en.wikipedia.org/wiki/Prefix_sum) (Blelloch, 1990), which evaluates any associative recurrence in O(log N) parallel steps instead of O(N) sequential steps.

For 128 atmospheric layers: **7 parallel steps instead of 128 sequential steps.**

### The 50-Year Gap

| Year | Discovery | Gap |
|------|-----------|-----|
| **1969** | Grant & Hunt prove RT layer operators form an associative semigroup | Math exists |
| **1990** | Blelloch publishes parallel prefix scan for associative operations | Algorithm exists |
| **2018** | Martin & Cundy parallelize identical recurrence structure in RNNs (9x GPU speedup) | ML community uses it |
| **2023** | Gu & Dao use same scan in Mamba state space model | It's mainstream in ML |
| **2024** | Ukkonen & Hogan achieve 12x radiation speedup but vertical transport remains sequential | Still not connected |
| **2026** | **This work**: first parallel prefix scan for atmospheric RT | **Gap closed** |

The mathematical structure for parallelization existed for over 50 years. The parallel algorithm for 35 years. Successful GPU implementations for identical math for 8 years. Yet every operational weather model — GFS, IFS, ICON, MPAS, E3SM — still computes radiative transport sequentially in the vertical.

---

## How It Works

Each atmospheric layer is converted to a 2x2 matrix encoding its Mobius transformation:

```
M[i] = | T^2 - R^2    R |
       | -R           1 |
```

The suffix product `M[k] * M[k+1] * ... * M[N-1]` is computed via right-to-left Hillis-Steele parallel scan in shared memory. Then albedo at each level is extracted by applying the suffix product to the surface boundary condition.

```cuda
// The entire parallel scan — 7 lines that replace 128 sequential steps
for (int stride = 1; stride < nlay; stride *= 2) {
    Mat2x2 val;
    if (tid + stride < nlay)
        val = mat2_mul(shared[tid], shared[tid + stride]);
    else
        val = shared[tid];
    __syncthreads();
    shared[tid] = val;
    __syncthreads();
}
albedo[tid] = mat2_apply(shared[tid], albedo_sfc);
```

---

## Repository Structure

```
rte-rrtmgp/rte/kernels/
    mo_fast_math.F90                 # Fast exp() module (Fortran)
    mo_rte_parallel_adding.F90       # Parallel prefix adding (Fortran)
    mo_rte_solver_kernels_opt.F90    # Optimized LW/SW solver (Fortran)

rte-rrtmgp/tests/
    test_fast_math.F90               # CPU benchmark (ALL TESTS PASS)

benchmarks/cuda/
    fast_exp_benchmark.cu            # GPU benchmark (all results above)

paper/
    manuscript.md                    # Full paper manuscript

research/
    bottleneck_map_rte_rrtmgp.md     # Operation-by-operation cost analysis
    SYNTHESIS_v1.md                  # Research synthesis (500+ papers)
    PARALLEL_PREFIX_RT_FINDINGS.md   # Mathematical foundations deep dive
```

---

## Building and Running

### CPU Benchmark (fast exp)

```bash
# Requires gfortran
cd build
gfortran -O3 -march=native -c ../rte-rrtmgp/rte/kernels/mo_rte_kind.F90
gfortran -O3 -march=native -c ../rte-rrtmgp/rte/kernels/mo_rte_util_array.F90
gfortran -O3 -march=native -c ../rte-rrtmgp/rte/kernels/mo_fast_math.F90
gfortran -O3 -march=native -c ../rte-rrtmgp/tests/test_fast_math.F90
gfortran -O3 -march=native -o test_fast_math *.o
./test_fast_math
```

### GPU Benchmark (parallel scan + precision)

```bash
# Requires CUDA toolkit + cl.exe (Visual Studio)
cd benchmarks/cuda
nvcc -O3 -arch=sm_86 fast_exp_benchmark.cu -o fast_exp_bench
./fast_exp_bench
```

Expected output:
```
Parallel Prefix Scan (adding method):
  Sequential:         0.2026 ms
  Parallel scan:      0.0429 ms
  Speedup:            4.73x
  Max relative error: 7.764e-07
  Error count (>1e-4): 0 / 132096
```

---

## Applicability

The parallel prefix scan works with **any** atmospheric model using the adding/doubling method:

- **NOAA GFS** (rte-rrtmgp) — direct drop-in
- **ECMWF IFS** (ecRad) — same adding method
- **DWD/MPI-M ICON** — same structure
- **NCAR MPAS** — same structure
- **DOE E3SM** (SCREAM) — uses rte-rrtmgp C++/Kokkos
- **Any two-stream solver** — the Mobius transformation property is inherent to the physics

---

## Citation

If you use this work, please cite:

```bibtex
@misc{parallel_prefix_rt_2026,
  title={Parallel Prefix Scan for Atmospheric Radiative Transfer},
  author={[Author]},
  year={2026},
  howpublished={\url{https://github.com/[username]/parallel-prefix-rt}},
  note={First application of parallel prefix scan to atmospheric RT vertical transport}
}
```

---

## References

- Grant, I.P. and Hunt, G.E. (1969). "Discrete space theory of radiative transfer." *Proc. R. Soc. A*, 313, 183-197. — **Proved associativity of RT layer operators**
- Blelloch, G.E. (1990). "Prefix Sums and Their Applications." CMU-CS-90-190. — **The parallel scan algorithm**
- Martin, E. and Cundy, C. (2018). "Parallelizing Linear Recurrent Neural Nets Over Sequence Length." *ICLR*. — **Same math, applied to RNNs, 9x speedup**
- Gu, A. and Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*. — **Same scan in modern ML**
- Ukkonen, P. and Hogan, R.J. (2024). "Twelve Times Faster yet Accurate." *JAMES*. doi:10.1029/2023MS003932. — **State-of-the-art radiation optimization (vertical transport still sequential)**
- Pincus, R. and Mlawer, E.J. (2019). "Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations." *JAMES*. doi:10.1029/2019MS001621. — **The RTE+RRTMGP library**
- Redheffer, R.M. (1959). "Inequalities for a matrix Riccati equation." *J. Math. Mech.*, 8, 349-367. — **The star product framework**

---

## License

BSD-3-Clause (same as upstream rte-rrtmgp)

The optimized kernels and parallel prefix scan implementation are original work.
The upstream rte-rrtmgp library is copyright Atmospheric and Environmental Research, Regents of the University of Colorado, Trustees of Columbia University.
