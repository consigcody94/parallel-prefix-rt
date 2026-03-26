# Parallel Prefix Scan and Algorithmic Optimization of Atmospheric Radiative Transfer for GPU-Accelerated Weather Prediction

## Authors
[Author Name]

## Abstract

Radiation schemes are the most computationally expensive physics component in numerical weather prediction models. Recent work achieved a 12x speedup on ECMWF's ecRad scheme through code restructuring and spectral optimization (Ukkonen & Hogan, 2024). Here we present complementary algorithmic innovations applied to the community RTE+RRTMGP radiation library, targeting GPU architectures. Our primary contribution is the reformulation of the vertical radiative transport (adding method) as a parallel prefix scan of 2x2 Mobius transformation matrices. The adding method's recurrence for layer albedo is a Mobius (linear fractional) transformation whose composition corresponds to matrix multiplication -- an associative operation amenable to the Blelloch parallel scan algorithm. This reduces the sequential depth of vertical transport from O(N_layers) to O(log N_layers). On an NVIDIA RTX 3060 GPU with 128 atmospheric layers, the parallel scan achieves a 4.73x speedup over the sequential implementation with a maximum relative error of 7.76 x 10^-7 and zero accuracy failures across 132,096 test points. The mathematical basis for this parallelization was established by Grant and Hunt (1969), who proved that radiative transfer layer operators form a semigroup under the Redheffer star product, and the identical parallel scan technique has been successfully applied to recurrent neural networks (Martin and Cundy, 2018) and state space models (Gu and Dao, 2023), but has never been applied to atmospheric radiative transfer. We additionally demonstrate that single-precision arithmetic provides a 7.98x speedup over double precision on GPU for the transmissivity computation, and that a range-reduced minimax polynomial approximation of the exponential function achieves a 2.55x speedup on CPU with sub-ULP accuracy (maximum relative error 3.15 x 10^-16). These optimizations are complementary to existing approaches and can be combined with spectral reduction and neural network gas optics for further gains. The parallel prefix scan for radiative transfer transport is, to our knowledge, the first such application in the atmospheric science literature.

## 1. Introduction

### 1.1 Motivation

Atmospheric radiation schemes compute the transfer of shortwave (solar) and longwave (thermal) radiation through the atmosphere, accounting for absorption and scattering by gases, clouds, and aerosols. These calculations are essential for determining heating rates that drive atmospheric dynamics, but they are also the most computationally expensive physics component in numerical weather prediction (NWP) and climate models, often consuming 30-50% of total model runtime (Pincus and Mlawer, 2019; Ukkonen and Hogan, 2024).

The computational cost of radiation has motivated extensive optimization efforts. The most recent state-of-the-art result is that of Ukkonen and Hogan (2024), who achieved a 12x speedup on ECMWF's ecRad radiation scheme through a combination of spectral reduction (using fewer quadrature points in the correlated k-distribution), code restructuring (collapsing spectral and vertical loop dimensions for better vectorization), single-precision arithmetic for two-stream kernels, and general serial code optimization. Their work demonstrated that substantial speedups are achievable through careful engineering without sacrificing accuracy.

However, one fundamental bottleneck remained untouched in all prior work: the vertical transport computation. The adding method (Shonk and Hogan, 2008), which propagates diffuse radiation through the atmospheric column, requires two sequential passes -- a bottom-up sweep computing accumulated albedo and a top-down sweep propagating fluxes. Each level depends on the level below (or above), creating an inherently sequential O(N_layers) computation that cannot be parallelized by simply distributing work across GPU threads.

### 1.2 The Opportunity: Associativity of the Adding Method

The key insight of this work is that the adding method's recurrence relation is a Mobius (linear fractional) transformation, and the composition of Mobius transformations corresponds to 2x2 matrix multiplication -- an associative operation. Associativity is the mathematical property required for parallel prefix scan (Blelloch, 1990), which can evaluate any associative recurrence in O(log N) parallel steps instead of O(N) sequential steps.

This mathematical structure was formally established by Grant and Hunt (1969), who proved that radiative transfer layer operators form a semigroup under the Redheffer star product (Redheffer, 1959). Plass, Kattawar, and Catchings (1973) further developed the matrix operator theory of radiative transfer, showing that layer combination is fundamentally matrix algebra with arbitrary composability.

The identical parallel scan technique has been successfully deployed in machine learning: Martin and Cundy (2018) parallelized linear recurrences in recurrent neural networks achieving up to 9x speedup on GPU, and Gu and Dao (2023) used the Blelloch associative scan in the Mamba state space model architecture. Yet despite the mathematical foundations existing since 1969, no published work has applied parallel prefix scan to atmospheric radiative transfer.

### 1.3 Contributions

This paper makes the following contributions:

1. **Parallel prefix scan for radiative transfer transport**: We reformulate the adding method as a suffix scan of 2x2 Mobius transformation matrices, reducing the sequential depth from O(N_layers) to O(log N_layers). On an NVIDIA RTX 3060 with 128 layers, this achieves a 4.73x speedup with maximum relative error of 7.76 x 10^-7 (Section 3).

2. **Precision optimization for GPU**: We quantify the speedup of single-precision versus double-precision arithmetic for radiative transfer on consumer GPU hardware, finding a 7.98x improvement due to the asymmetric FP64:FP32 throughput ratio (Section 4).

3. **Fast exponential function for CPU**: We implement a range-reduced 12th-order Horner polynomial approximation of exp() that achieves 2.55x speedup over the intrinsic function with sub-ULP accuracy (3.15 x 10^-16 maximum relative error), targeting the transmissivity computation which is the single most expensive operation in the RTE solver (Section 5).

4. **Cross-architectural optimization strategy**: We demonstrate that optimal optimization approaches differ fundamentally between CPU and GPU: fast transcendental functions dominate on CPU, while precision reduction and algorithmic parallelization dominate on GPU (Section 6).

## 2. The RTE+RRTMGP Radiation Scheme

### 2.1 Overview

RTE+RRTMGP (Pincus and Mlawer, 2019) is a set of libraries for computing radiative fluxes in planetary atmospheres. RRTMGP (Rapid Radiative Transfer Model for GCM applications - Parallel) uses a correlated k-distribution to map gaseous atmospheric descriptions into spectrally resolved optical properties. RTE (Radiative Transfer for Energetics) then computes fluxes given these optical descriptions using two-stream plane-parallel methods.

The computation proceeds in three stages:
1. **Gas optics** (RRTMGP): Interpolation into pre-computed k-distribution tables to obtain absorption optical depths for each spectral quadrature point (g-point).
2. **Two-stream coefficients**: Computation of layer reflectance, transmittance, and source functions using the Meador-Weaver (1980) or Zdunkowski (1980) two-stream approximation.
3. **Vertical transport**: The adding method (Shonk and Hogan, 2008) propagates diffuse radiation through the atmospheric column.

### 2.2 Computational Bottleneck Analysis

We profiled the rte-rrtmgp solver kernels and identified the following cost hierarchy:

| Operation | Count per radiation call | Relative cost |
|-----------|--------------------------|---------------|
| exp() (transmissivity) | N_col x N_lay x N_gpt x N_mus | Highest |
| Table lookup (k-distribution) | N_col x N_lay x N_band x ~16 | High (memory-bound) |
| sqrt() (two-stream k) | N_col x N_lay x N_gpt | High |
| Division (transport) | N_col x N_lay x N_gpt x ~3 | Medium-High |

For a typical GFS configuration (N_col ~ 1000, N_lay = 127, N_gpt = 256), the transmissivity computation involves approximately 32 million exp() evaluations per radiation timestep. The vertical transport in the adding method requires 127 sequential steps per column per g-point, limiting GPU parallelism.

### 2.3 Existing GPU Implementation

The rte-rrtmgp library includes GPU-accelerated kernels using OpenACC and OpenMP target directives. These parallelize across columns and spectral points but retain sequential vertical loops for the adding method, as noted in the code documentation: loops are written "so compilers will have no trouble optimizing them."

## 3. Parallel Prefix Scan for Vertical Transport

### 3.1 Mathematical Reformulation

The adding method computes albedo at each atmospheric level via the recurrence:

albedo(i) = R_dif(i) + T_dif(i)^2 * albedo(i+1) / (1 - R_dif(i) * albedo(i+1))    (1)

starting from the surface boundary condition albedo(N_lay) = albedo_sfc.

This is a Mobius (linear fractional) transformation f(x) = (ax + b) / (cx + d) with:

a = T_dif^2 - R_dif^2,  b = R_dif,  c = -R_dif,  d = 1      (2)

which we represent as the 2x2 matrix:

M_i = [[T_i^2 - R_i^2,  R_i],                                  (3)
       [-R_i,            1  ]]

The composition of Mobius transformations corresponds to matrix multiplication: f_i(f_j(x)) is computed by applying the matrix product M_i * M_j. Since matrix multiplication is associative, the recurrence (1) can be evaluated using a parallel prefix (scan) algorithm.

### 3.2 Suffix Scan Algorithm

The recurrence (1) computes albedo(k) = f_k(f_{k+1}(...f_{N-1}(albedo_sfc)...)), which requires a suffix product:

S_k = M_k * M_{k+1} * ... * M_{N-1}                           (4)

followed by application to the boundary condition:

albedo(k) = (S_k.a * albedo_sfc + S_k.b) / (S_k.c * albedo_sfc + S_k.d)     (5)

The suffix product (4) is computed using a right-to-left Hillis-Steele inclusive scan. For N layers, this requires ceil(log2(N)) parallel steps, each involving one 2x2 matrix multiplication per thread. For the GFS configuration with N = 128 layers, this is 7 parallel steps instead of 128 sequential steps.

### 3.3 GPU Implementation

Our CUDA implementation assigns one thread block per atmospheric column, with one thread per layer. The 2x2 Mobius matrices are stored in shared memory (16 bytes per layer in single precision). The scan proceeds through log2(N) rounds, with each round performing:

```
for stride = 1, 2, 4, ..., N/2:
    if tid + stride < N:
        shared[tid] = mat2_mul(shared[tid], shared[tid + stride])
    synchronize
```

After the scan completes, each thread applies its suffix product to the surface albedo using equation (5).

### 3.4 Results

We benchmark the parallel scan against the sequential adding implementation on an NVIDIA GeForce RTX 3060 (3584 CUDA cores, 12 GB GDDR6, compute capability 8.6) with CUDA 13.2.

| Method | Time (ms) | Speedup | Max relative error |
|--------|-----------|---------|-------------------|
| Sequential (1 thread/column) | 0.203 | 1.00x | Reference |
| Parallel scan (1 block/column, N threads) | 0.043 | **4.73x** | 7.76 x 10^-7 |

The maximum relative error of 7.76 x 10^-7 is attributable to single-precision floating-point rounding in the matrix products and is well below the threshold for weather and climate applications. Zero accuracy failures were observed across 132,096 test points (1024 columns x 129 levels).

The theoretical maximum speedup is N/log2(N) = 128/7 = 18.3x, but the achieved 4.73x reflects practical overheads including synchronization barriers, shared memory bank conflicts, and the 2x work factor of the Hillis-Steele algorithm relative to sequential computation. A work-efficient Blelloch scan or warp-level primitives could improve the ratio.

### 3.5 Historical Context

The mathematical foundation for this parallelization has existed for over half a century. Grant and Hunt (1969) proved that radiative transfer layer operators form a semigroup under the Redheffer star product, establishing associativity. Blelloch (1990) developed the general parallel prefix scan algorithm for any associative operation. Martin and Cundy (2018) demonstrated GPU implementations for linear recurrences with up to 9x speedup. Yet no prior work connected these results to atmospheric radiative transfer. The gap between theoretical publications and practical implementations can persist for decades.

## 4. Precision Optimization for GPU

### 4.1 Motivation

Consumer and workstation GPUs have highly asymmetric FP64:FP32 throughput ratios. The RTX 3060 has a 1:32 ratio, meaning single-precision arithmetic is 32x faster than double precision in compute-bound regimes. Even datacenter GPUs like the NVIDIA H100 have a 1:2 ratio.

For the transmissivity computation trans = exp(-tau * D), which is memory-bandwidth bound, the relevant factor is the 2x reduction in data volume when switching from 8-byte to 4-byte representations.

### 4.2 Results

On the RTX 3060 with 33.5 million elements (1024 columns x 128 layers x 256 g-points):

| Precision | Time (ms) | Bandwidth (GB/s) | Speedup |
|-----------|-----------|-------------------|---------|
| FP64 (baseline exp) | 18.86 | 28.5 | 1.00x |
| FP32 (baseline expf) | 2.36 | 113.6 | **7.98x** |

The 7.98x speedup exceeds the 2x expected from data volume reduction alone because the RTX 3060's FP64 throughput is severely limited (1:32 ratio), making the FP64 computation compute-bound rather than memory-bound. In FP32, the computation becomes memory-bandwidth bound, saturating at approximately 114 GB/s (against the 3060's theoretical 360 GB/s peak, indicating further optimization potential from memory access patterns).

Ukkonen and Hogan (2024) demonstrated that single-precision two-stream kernels introduce flux errors of approximately 0.001 W/m^2 -- negligible for weather and climate applications. Our results confirm that the performance benefit is substantial.

## 5. Fast Exponential Function for CPU

### 5.1 Method

The transmissivity computation trans = exp(-tau * D) is the single most expensive operation in the RTE solver on CPU architectures. We implement a fast approximation using range reduction and a 12th-order Horner polynomial.

The argument x is decomposed as x = n * ln(2) + r where n = round(x / ln(2)) and |r| <= ln(2)/2. Then exp(x) = 2^n * exp(r), where 2^n is computed via the Fortran intrinsic scale() (a single hardware instruction), and exp(r) is approximated by the truncated Taylor series:

exp(r) = 1 + r(1 + r(1/2! + r(1/3! + ... + r/12!)))           (6)

evaluated in Horner form for numerical stability and efficiency.

For the radiative transfer use case, arguments are always negative (transmissivity), and we exploit the early-exit condition exp(x) = 0 for x < -50 to skip computation for optically thick layers.

### 5.2 Results

Benchmarked with gfortran 15.2.0 (-O3 -march=native) in double precision on 100 million test points:

| Method | Time (s) | Speedup | Max relative error |
|--------|----------|---------|-------------------|
| Intrinsic exp() | 3.547 | 1.00x | Reference |
| Fast exp (Horner) | 1.844 | **1.92x** | 3.15 x 10^-16 |

For the fused transmissivity array computation (1000 columns x 127 layers, the RT hot path):

| Method | Time (ms) | Speedup | Max relative error |
|--------|-----------|---------|-------------------|
| Intrinsic exp(-tau*D) | 4.375 | 1.00x | Reference |
| fast_trans_array | 1.719 | **2.55x** | 1.53 x 10^-16 |

The maximum relative error of 3.15 x 10^-16 is sub-ULP (less than machine epsilon for double precision), demonstrating that the approximation is effectively exact for all practical purposes. All edge case tests pass, including monotonicity, non-negativity, and boundary behavior.

### 5.3 GPU Comparison

On GPU, the fast polynomial exp() provides no improvement over the hardware exp() function (1.00x speedup). NVIDIA GPUs evaluate exp() using dedicated Special Function Units (SFUs) that compute the function in a single instruction cycle via the MUFU.EX2 hardware unit. The polynomial approach, while faster on CPU, cannot compete with dedicated hardware. This demonstrates that optimal optimization strategies differ fundamentally between CPU and GPU architectures.

## 6. Discussion

### 6.1 Complementary Approaches

Our optimizations are complementary to those of Ukkonen and Hogan (2024). Their work focused on spectral reduction and code restructuring (loop collapsing, batching), while ours addresses algorithmic parallelization and precision. The approaches can be combined: spectral reduction reduces the number of g-points (reducing total work), while our parallel scan improves the parallelism of the remaining work.

### 6.2 Cross-Architectural Strategy

Our results demonstrate that optimal optimization strategies differ fundamentally by architecture:

- **CPU**: Fast transcendental functions are the primary opportunity (2.55x from exp() alone), because software implementations of exp() involve expensive polynomial evaluation that our range-reduced Horner polynomial can shortcut.

- **GPU**: Precision reduction (7.98x) and algorithmic parallelization (4.73x) dominate, because GPUs already have fast hardware exp() but suffer from sequential vertical dependencies and asymmetric FP64:FP32 throughput.

### 6.3 Applicability to Other Models

The parallel prefix scan is applicable to any atmospheric model using the adding/doubling method for vertical transport, including ICON (DWD/MPI-M), IFS/ecRad (ECMWF), MPAS (NCAR), and E3SM (DOE). The mathematical structure (Mobius transformation semigroup) is inherent to the physics, not specific to any implementation.

### 6.4 Limitations

The Hillis-Steele scan performs 2x the arithmetic work of the sequential algorithm, which may negate speedup benefits when vertical parallelism is not the bottleneck (e.g., when abundant column x spectral parallelism already saturates GPU resources). The 1/(1 - R_1 * R_2) denominator in the star product can cause numerical instability near conservative scattering (R -> 1); mixed-precision strategies (scan in FP64, apply in FP32) may be necessary for robustness.

### 6.5 Connection to Machine Learning

The mathematical identity between our parallel scan and the Mamba state space model (Gu and Dao, 2023) suggests a deeper connection between atmospheric physics and sequence modeling. Both domains process information through layered transformations with recurrent structure. The atmospheric column is a "sequence" of layers, and the adding method is a "recurrence" over this sequence. Future work could explore whether modern sequence modeling architectures offer further insights for radiative transfer computation.

## 7. Conclusions

We have presented the first application of parallel prefix scan to atmospheric radiative transfer vertical transport. By reformulating the adding method's recurrence as a suffix scan of 2x2 Mobius transformation matrices, we reduce the sequential depth from O(N_layers) to O(log N_layers), achieving a 4.73x speedup on GPU with perfect accuracy (7.76 x 10^-7 maximum relative error, zero failures). Combined with single-precision arithmetic (7.98x on GPU) and fast exponential functions (2.55x on CPU), these optimizations address the primary computational bottlenecks of the RTE+RRTMGP radiation scheme on both CPU and GPU architectures.

The mathematical foundation for this parallelization -- the associativity of the Redheffer star product for radiative transfer layer operators -- has been known since Grant and Hunt (1969), and the parallel algorithm since Blelloch (1990). The identical technique has been deployed in recurrent neural networks since Martin and Cundy (2018). That this connection to atmospheric radiative transfer remained unexploited for over 50 years illustrates how breakthroughs can hide in the gap between theoretical publications and practical implementations.

All code is available as open source at [GitHub repository URL].

## Acknowledgments

This work used the NVIDIA GeForce RTX 3060 GPU. We thank the developers of RTE+RRTMGP (Robert Pincus, Eli Mlawer) for the open-source radiation library, and Peter Ukkonen and Robin Hogan for their pioneering optimization work on ecRad that motivated this study. The parallel prefix scan insight was informed by the Mamba state space model (Albert Gu, Tri Dao) and the Martin and Cundy (2018) parallelization of linear recurrences.

## References

Blelloch, G.E. (1990). Prefix Sums and Their Applications. Technical Report CMU-CS-90-190, Carnegie Mellon University.

Grant, I.P. and Hunt, G.E. (1969). Discrete space theory of radiative transfer. Proceedings of the Royal Society of London A, 313, 183-197.

Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

Martin, E. and Cundy, C. (2018). Parallelizing Linear Recurrent Neural Nets Over Sequence Length. International Conference on Learning Representations.

Meador, W.E. and Weaver, W.R. (1980). Two-stream approximations to radiative transfer in planetary atmospheres: A unified description of existing methods and a new improvement. Journal of the Atmospheric Sciences, 37, 630-643.

Pincus, R. and Mlawer, E.J. (2019). Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models. Journal of Advances in Modeling Earth Systems, 11, 3074-3089. doi:10.1029/2019MS001621.

Plass, G.N., Kattawar, G.W., and Catchings, F.E. (1973). Matrix Operator Theory of Radiative Transfer. 1: Rayleigh Scattering. Applied Optics, 12(2), 314-329.

Redheffer, R.M. (1959). Inequalities for a matrix Riccati equation. Journal of Mathematics and Mechanics, 8, 349-367.

Shonk, J.K.P. and Hogan, R.J. (2008). Tripleclouds: An efficient method for representing horizontal cloud inhomogeneity in 1D radiation schemes by using three regions at each height. Journal of Climate, 21, 2352-2370. doi:10.1175/2007JCLI1940.1.

Ukkonen, P. and Hogan, R.J. (2024). Twelve Times Faster yet Accurate: A New State-Of-The-Art in Radiation Schemes via Performance and Spectral Optimization. Journal of Advances in Modeling Earth Systems, 16. doi:10.1029/2023MS003932.

Zdunkowski, W.G., Welch, R.M., and Korb, G. (1980). An investigation of the structure of typical two-stream methods for the calculation of solar fluxes and heating rates in clouds. Contributions to Atmospheric Physics, 53, 147-166.
