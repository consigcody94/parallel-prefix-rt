# FV3 fv_tp_2d Optimization: Research Findings

## THE SINGLE MOST ACTIONABLE FINDING
Pace/GT4Py has ALREADY demonstrated 3.92x speedup on FV3 fv_tp_2d-class kernels (Ben-Nun et al., SC22).
Combined with temporal blocking + mixed precision could yield 8-15x.

## Tier 1: Highest Impact (address 40% load/store bottleneck)

### 1. Temporal Blocking / Kernel Fusion
- AN5D Framework (Matsumura et al., CGO 2020, 55 citations)
- Fuse PPM reconstruction + flux computation + limiting + update into single GPU kernel
- Keep intermediates in shared memory/registers
- Expected: **2-4x reduction in memory traffic**

### 2. Mixed Precision Transport
- Kashi et al. (J. Supercomputing, March 2026) — BRAND NEW from ORNL
- FP32 intermediate fluxes, FP64 accumulation
- Halving data width ≈ doubles effective bandwidth
- Expected: **1.5-2x throughput**

### 3. Data Layout Transformation
- Column-major Fortran → GPU-coalesced layout
- GT4Py/Pace handles this automatically
- Expected: **1.3-2x memory bandwidth utilization**

## Tier 2: Algorithmic Improvements

### 4. Multi-Moment Reconstruction (Chen et al. 2020)
- Replace PPM with compact multi-moment scheme
- 4th-order accuracy, narrower stencil
- Reduces both memory reads AND halo width

### 5. Widened Halos + Overlapped Communication
- Compute interior while communicating halos
- Critical for cubed-sphere panel boundaries

### 6. Space-Filling Curve Traversal
- Hilbert curve ordering for 2D stencil cache locality
- Weinzierl (ACM TOMS 2019, 50 citations)

### 7. Horner-Form Polynomial Evaluation
- PPM: a + b*x + c*x² → a + x*(b + c*x)
- 33% fewer multiplies, better numerical stability
- Enables safer reduced precision

## Key References
| Paper | Year | Speedup | Relevance |
|-------|------|---------|-----------|
| Ben-Nun et al., SC22 | 2022 | 3.92x | Directly optimized FV3 via GT4Py |
| Dahm et al., GMD | 2023 | 3.5-4x | Full Python rewrite of FV3 transport |
| Matsumura et al., CGO | 2020 | 2-4x | AN5D temporal blocking for GPU |
| Kashi et al., J. Supercomputing | 2026 | up to 8x | Mixed precision survey (ORNL) |
| H-AMR, ApJS | 2022 | 100-1000x | PPM on GPUs with AMR |
| Fuhrer et al., GMD | 2018 | large | COSMO on 4888 GPUs via DSL |
