# Bottleneck Map: rte-rrtmgp (Radiative Transfer)

## Executive Summary
rte-rrtmgp computes radiative fluxes in planetary atmospheres using correlated k-distribution
for gas optics and two-stream/adding methods for radiative transfer. It is the single most
expensive physics component in NOAA's GFS weather model.

## Hot Path Architecture

### 1. Gas Optics (RRTMGP) — `compute_tau_absorption`

#### 1a. `interpolation()` — Compute interpolation coefficients
- **Inner loop**: ncol x nlay x nflav x 2 (temperature levels)
- **Operations**: log(pressure), division, floor, clamp, eta computation
- **Key cost**: `log()` call per column/layer, divisions for eta
- **GPU status**: OpenACC/OpenMP parallelized (collapse(2) on icol,ilay)

#### 1b. `gas_optical_depths_major()` — Major species optical depth
- **Inner loop**: nbnd x nlay x ncol
- **Core operation**: `interpolate3D_byflav()` — 8-point trilinear interpolation
  - 8 table lookups from `kmajor(ntemp, neta, npres+1, ngpt)`
  - 8 multiply-adds per g-point
- **Bottleneck**: Memory bandwidth (table access pattern is scattered)

#### 1c. `gas_optical_depths_minor()` — Minor species optical depth
- **Similar to major** but with additional density/complement scaling
- **Additional cost**: Division for VMR computation, conditional branches

#### 1d. `compute_Planck_source()` — Planck function fractions
- **Same interpolation pattern** as gas optics
- **Additional**: Total Planck function lookup from pre-computed table

### 2. RTE Solver — Longwave No-Scattering

#### 2a. `lw_solver_noscat_oneangle()` — Core LW solver
- **Outer loop**: igpt = 1 to ngpt (~256 g-points)
- **Per g-point operations**:
  1. `tau_loc = tau * D` (multiply)
  2. **`trans = exp(-tau_loc)`** — MOST EXPENSIVE OPERATION
  3. `lw_source_noscat()` — linear-in-tau source function
  4. `lw_transport_noscat_dn()` — sequential layer-by-layer (CANNOT parallelize vertically)
  5. `lw_transport_noscat_up()` — sequential upward
  6. `flux *= pi * weight` (broadband integration)

#### Key Numerical Observation:
- The `exp()` function is called `ncol × nlay × ngpt × nmus` times
- For typical GFS: ncol=~thousands, nlay=~127, ngpt=256, nmus=1-4
- This is potentially BILLIONS of exp() calls per radiation timestep

### 3. RTE Solver — Two-Stream (LW and SW)

#### 3a. `lw_two_stream()` — Meador-Weaver coefficients
- gamma1 = LW_diff_sec * (1 - 0.5*w0*(1+g))  [Fu et al. 1997]
- gamma2 = LW_diff_sec * 0.5*w0*(1-g)
- k = sqrt(max((g1-g2)*(g1+g2), 1e-12))  — **sqrt per col/lay**
- exp_minusktau = exp(-tau*k)  — **exp per col/lay**
- RT_term = 1/(k*(1+exp2) + g1*(1-exp2))  — **division per col/lay**
- Rdif, Tdif from RT_term

#### 3b. `sw_dif_and_source()` — Shortwave (PIFM method)
- Zdunkowski PIFM: gamma1 = (8-w0*(5+3g))/4, gamma2 = 3*w0*(1-g)/4
- Same k, exp, RT_term pattern as LW
- PLUS: Tnoscat = exp(-tau/mu0) — additional exp
- Direct beam: Rdir, Tdir with complex formulas (Eqs 14-15)
- Energy conservation clamp: Rdir = max(0, min(Rdir, 1-Tnoscat))

#### 3c. `adding()` — Shonk & Hogan 2008 transport
- **TWO-PASS algorithm** (inherently sequential in vertical):
  - Pass 1 (bottom→top): Compute albedo and source at each level
    - denom = 1/(1 - Rdif*albedo_below)  — division per level
    - albedo = Rdif + Tdif²*albedo_below*denom
    - src = src_up + Tdif*denom*(src_below + albedo_below*src_dn)
  - Pass 2 (top→bottom): Compute fluxes
    - flux_dn = (Tdif*flux_above + Rdif*src + src_dn)*denom
    - flux_up = flux_dn*albedo + src

### 4. Interpolation Kernels

#### `interpolate2D_byflav()` — Minor species
- 4-point bilinear interpolation (temp × eta)
- Pure: `res = f(1,1)*k(j1,e1) + f(2,1)*k(j1,e2) + f(1,2)*k(j2,e1) + f(2,2)*k(j2,e2)`

#### `interpolate3D_byflav()` — Major species
- 8-point trilinear interpolation (temp × eta × pressure)
- Two temperature levels, each with 4-point (eta × pressure)
- Scaled by col_mix

## Operation Cost Hierarchy

| Operation | Count (per radiation call) | Relative Cost |
|-----------|---------------------------|---------------|
| `exp()` | ncol × nlay × ngpt × nmus | **HIGHEST** |
| Table lookup (k-dist) | ncol × nlay × nbnd × ~16gpt | High (memory-bound) |
| `sqrt()` | ncol × nlay × ngpt (2-stream) | High |
| Division | ncol × nlay × ngpt × ~3 | Medium-High |
| `log()` | ncol × nlay (interpolation) | Medium |
| Multiply-add | Everywhere | Base |

## Optimization Opportunities Identified

### A. Fast exp() Approximation (HIGHEST IMPACT)
- tau*D typically in range [0, ~50]
- Padé approximant or minimax polynomial could be 2-5x faster
- Error bounds manageable for weather/climate (not spectroscopy)
- Already done in some ML inference — transfer to Fortran

### B. Fused exp-related Operations
- Pattern: trans = exp(-t); source ~ (1-trans)/t is numerically problematic for small t
- Already uses linear-in-tau; could use higher-order

### C. k-Distribution Table Compression
- kmajor: ntemp × neta × (npres+1) × ngpt — potentially compressible
- Low-rank (SVD/Tucker) decomposition could reduce memory traffic
- Tensor train decomposition for multi-dimensional tables

### D. Better Quadrature for Angular Integration
- Currently: user-supplied Gaussian quadrature (nmus = 1-4)
- LW_diff_sec = 1.66 (fixed diffusivity factor for 1-angle)
- Clenshaw-Curtis, tanh-sinh, or Gauss-Jacobi might need fewer angles

### E. Parallel Prefix for Sequential Transport
- The adding() method is O(n) sequential in vertical
- Parallel prefix (scan) algorithms could make it O(log n) parallel
- References: Tridiagonal solvers on GPU, cyclic reduction

### F. Two-Stream Reformulation
- Current: Meador-Weaver (1980) / Zdunkowski PIFM (1980)
- Modern alternatives: Ukkonen & Hogan (2024) optimized two-stream
- Matrix formulation enabling batch linear algebra

### G. Mixed Precision Strategy
- Use single precision for intermediate calculations
- Only accumulate in double precision for broadband sums
- The energy conservation clamp in SW suggests sensitivity — needs analysis

## Existing Alternative Implementations
- **Julia**: CliMA/RRTMGP.jl
- **JAX**: climate-analytics-lab/jax-rrtmgp
- **C++/Kokkos**: E3SM-Project (EAMxx)
- **GPU Fortran**: accel/ directory uses OpenACC + OpenMP target
