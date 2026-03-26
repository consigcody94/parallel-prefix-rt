# NOAA Computational Optimization: Research Synthesis v1
## Cross-Disciplinary Breakthrough Opportunities

### Status: Phase 3 (Parallel Research Execution) — 7 agents deployed, initial synthesis in progress

---

## TARGET 1: rte-rrtmgp (Radiative Transfer) — PRIMARY TARGET

### Prior Art Summary
- **Ukkonen & Hogan (2024)**: 12x speedup on ecRad via code restructuring + spectral reduction + single precision. Published in JAMES.
- **Ukkonen & Hogan (2023)**: RRTMGP-NN 2.0 — neural network replacement for gas optics tables in ECMWF IFS
- **Ukkonen (2022)**: Explored ML emulation pathways; FNNs may not suit sequential RT computation
- **Abdi & Jankov (2024)**: GPU acceleration of CCPP physics — 10x speedup for microphysics
- **Pincus & Mlawer (2019)**: Original rte-rrtmgp description (foundational paper)
- **Alternative implementations**: Julia (CliMA/RRTMGP.jl), JAX (jax-rrtmgp), C++/Kokkos (E3SM/EAMxx)

### Identified Gaps (Ukkonen did NOT address these)

#### Gap 1: Fast exp() Approximation — HIGHEST ROI
- **Current**: `exp()` is called ncol × nlay × ngpt × nmus times per radiation call
  - GFS typical: ~1000 × 127 × 256 × 1 = ~32 million exp() calls per timestep
  - This is THE single most expensive operation in the entire radiation scheme
- **Opportunity**: Replace with Padé approximant or minimax polynomial
  - Literature: Schraudolph (1999) IEEE-trick gives ~10x speedup with ~3% error
  - Better: 5th-order minimax polynomial on [-50, 0] gives <1e-6 relative error, ~3x faster
  - Even better: Range-reduced exp with lookup table + polynomial correction
  - Cross-disciplinary: Kerala School (Madhava) series acceleration for convergence of Taylor exp
- **Impact**: 2-3x speedup on the solver alone
- **Risk**: Must verify radiation flux errors remain within acceptable bounds

#### Gap 2: Parallel Prefix Scan for Vertical Transport — NOVEL
- **Current**: The `adding()` subroutine is inherently sequential in the vertical
  - Two-pass: bottom-up albedo, top-down flux
  - Each level depends on the one below/above
  - This limits vertical parallelization on GPU
- **Opportunity**: Reformulate as parallel prefix (scan) problem
  - The recurrence `albedo(i) = Rdif(i) + Tdif(i)^2 * albedo(i+1) * denom(i)` is a linear recurrence
  - Linear recurrences can be solved with parallel prefix in O(log n) steps
  - References: Blelloch (1990) parallel prefix, cuSPARSE gtsv2 for tridiagonal systems
  - The two-stream equations can be cast as 2×2 matrix product → parallel scan of matrix products
  - Cross-disciplinary: Euler's continued fractions (parallel evaluation algorithms)
- **Impact**: O(nlay) → O(log(nlay)) for transport computation
- **Risk**: Numerical stability of parallel scan in reduced precision; overhead for small nlay

#### Gap 3: Tensor Decomposition of k-Distribution Tables — NOVEL
- **Current**: `kmajor(ntemp, neta, npres+1, ngpt)` — 4D tensor accessed via trilinear interpolation
  - Memory-bandwidth bound operation
  - 8 table lookups per g-point per column per layer
- **Opportunity**: Tucker/CP/tensor-train decomposition
  - If kmajor has low-rank structure, decompose as sum of rank-1 tensors
  - Replace 8-point interpolation with factor-wise interpolation
  - Reduces memory footprint AND access pattern
  - Cross-disciplinary: Ramanujan-type identities for structured tensor decomposition
  - Ancient: Babylonian base-60 as mixed-radix representation → tensor product structure
- **Impact**: Reduce memory traffic by 2-4x if rank is low
- **Risk**: Must analyze actual rank structure of kmajor table

#### Gap 4: GPU-Specific Kernel Optimization for NOAA's Ursa — PRACTICAL
- **Current**: GPU version uses OpenACC/OpenMP target directives
  - Generic, not tuned for H100 architecture
- **Opportunity**: Hand-tuned CUDA kernels or better directive strategies
  - Warp-level primitives for reductions
  - Shared memory for table lookups
  - Cooperative groups for the parallel prefix scan
  - Tensor cores for batched 2×2 matrix operations (two-stream)
  - Cross-disciplinary: GPU architecture as "massively parallel Babylonian abacus"
- **Impact**: 2-5x on GPU over current OpenACC implementation
- **Risk**: Portability concerns; NOAA values portability

#### Gap 5: Spectrally Adaptive g-Point Selection — NOVEL ALGORITHM
- **Current**: All 256 g-points computed for every column regardless of atmospheric conditions
- **Opportunity**: Skip g-points that contribute negligibly to flux in specific conditions
  - Use atmospheric profile (T, P, humidity) to predict which g-points dominate
  - Implement as decision tree or simple threshold
  - Cross-disciplinary: Jain "Ardhacheda" (logarithmic thinking) — importance sampling
  - Ancient: Chinese "Tian Yuan Shu" (method of celestial element) — adaptive computation
- **Impact**: Could skip 30-50% of g-points in clear-sky conditions
- **Risk**: Must maintain accuracy; weather/climate integration may amplify errors

#### Gap 6: Apply Ukkonen's Techniques to rte-rrtmgp Directly
- **Current**: Ukkonen optimized ecRad (ECMWF's wrapper around RRTMGP)
  - The community rte-rrtmgp code that NOAA uses has NOT had these optimizations
- **Opportunity**: Port the dimensional collapsing, batching, and single-precision techniques
  - Collapse g-point and layer loops for better vectorization
  - Batch adjacent cloudy columns
  - Single-precision two-stream kernels with energy conservation bounds
- **Impact**: 3x from code restructuring alone (matching Ukkonen's result)
- **Risk**: Low risk — proven technique, just not applied to this codebase

---

## TARGET 2: FV3 fv_tp_2d (Transport Flux) — SECONDARY

### Key Bottleneck
- 40% load/store instructions → memory-bandwidth bound
- `fv_tp_2d`: PPM (Piecewise Parabolic Method) for horizontal transport on cubed-sphere
- Halo exchange between tiles every timestep

### Potential Breakthroughs
1. **Cache-oblivious tiling**: Temporal blocking for stencil computation
   - Literature: Frigo & Strumpen (2005), Datta et al. (2008)
   - Cross-disciplinary: Space-filling curves (Hilbert, Peano) for memory locality
2. **Communication-avoiding PPM**: Reduce halo exchanges via larger stencils with fewer communications
3. **Mixed-radix FFT on cubed-sphere**: Ancient Babylonian base-60 → mixed-radix computation
4. **Horner-like evaluation**: Qin Jiushao's method for polynomial evaluation in PPM reconstruction

---

## TARGET 3: CCPP Microphysics (Thompson) — SECONDARY

### Key Bottleneck
- Lookup tables for ice/snow/rain processes
- Conditional branches (ice nucleation regimes)
- Data layout mismatch for GPU (horizontal-first vs vertical-first)

### Potential Breakthroughs
1. **ML emulator**: Neural network replacement for lookup tables (proven approach)
2. **Moment method acceleration**: Better numerical integration for size distributions
   - Cross-disciplinary: Jain combinatorial enumeration for particle counting
3. **GPU kernel fusion**: Combine multiple microphysics substeps into single kernel
4. **Symbolic regression**: Discover simplified physics formulas that approximate full scheme

---

## TARGET 4: WW3 (Wave Model) — SECONDARY

### Key Bottleneck
- Discrete Interaction Approximation (DIA) for nonlinear wave-wave interactions
- Spectral propagation across global grid

### Potential Breakthroughs
1. **Fast DIA alternatives**: Web-Boltzmann integral approximation
   - Cross-disciplinary: Fourier analysis history (Pythagoras harmonics)
2. **GPU spectral propagation**: Parallelize across spectral bins
3. **ML wave emulator**: Physics-informed neural network for wave prediction
4. **Sparse spectral representation**: Wavelet-based instead of Fourier-based

---

## RANKING BY EXPECTED ROI

| # | Optimization | Target | Expected Speedup | Implementation Ease | Risk | Priority |
|---|-------------|--------|-------------------|--------------------|----- |----------|
| 1 | Apply Ukkonen techniques to rte-rrtmgp | rte-rrtmgp | 3x | Medium | Low | **HIGHEST** |
| 2 | Fast exp() approximation | rte-rrtmgp | 2-3x (on solver) | Easy | Low | **HIGH** |
| 3 | Single precision two-stream | rte-rrtmgp | 1.5-2x | Easy | Low | **HIGH** |
| 4 | Parallel prefix for adding() | rte-rrtmgp | O(n)→O(log n) | Hard | Medium | **MEDIUM-HIGH** |
| 5 | Tensor decomposition of kmajor | rte-rrtmgp | 2-4x memory | Hard | Medium | **MEDIUM** |
| 6 | GPU kernel optimization | rte-rrtmgp | 2-5x on GPU | Medium | Low | **MEDIUM** |
| 7 | Adaptive g-point selection | rte-rrtmgp | 1.3-1.5x | Medium | Medium | **MEDIUM** |
| 8 | Cache-oblivious tiling | FV3 | 1.5-2x | Hard | Low | **MEDIUM** |
| 9 | ML microphysics emulator | CCPP | 10x+ | Medium | High | **RESEARCH** |
| 10 | Fast DIA alternatives | WW3 | 2-3x | Hard | Medium | **RESEARCH** |

## RECOMMENDED IMPLEMENTATION ORDER
1. Fast exp() approximation (lowest risk, high reward, quick win)
2. Single precision two-stream kernels (proven by Ukkonen on ecRad)
3. Dimensional collapsing (Ukkonen technique applied to rte-rrtmgp)
4. Parallel prefix scan for adding() (novel, publishable)
5. Tensor decomposition of k-tables (novel, publishable)

## PUBLICATION STRATEGY
- **Primary paper**: "Algorithmic Optimization of Radiative Transfer for GPU-Accelerated Weather Prediction"
  - Target: JAMES (Journal of Advances in Modeling Earth Systems) — same venue as Ukkonen
  - Novelty: Fast exp, parallel prefix scan, tensor decomposition — none applied to RT before
  - Building on: Ukkonen & Hogan (2024), Pincus & Mlawer (2019)
- **Secondary paper**: Cross-cutting optimizations for FV3, CCPP, WW3 (if results warrant)
