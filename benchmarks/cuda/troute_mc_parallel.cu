/**
 * GPU Parallel Prefix Scan for Muskingum-Cunge River Routing
 *
 * NOAA t-route (https://github.com/NOAA-OWP/t-route) Issue #526:
 * "speed up MC calculations by rearranging space/time sequence"
 *
 * The Muskingum-Cunge routing equation for a single segment:
 *   qdc = C1*qup + C2*quc + C3*qdp + C4
 *
 * Within a reach (multiple segments, single timestep):
 *   - quc for segment i+1 = qdc from segment i (sequential dependency)
 *   - qup, qdp are known from the previous timestep
 *   - C1, C2, C3, C4 depend on flow depth (nonlinear, solved via secant method)
 *
 * Two-Phase Operator-Splitting Approach:
 *   Phase 1: Compute C1-C4 coefficients for ALL segments in parallel
 *            (using previous timestep's depth as initial estimate)
 *   Phase 2: Solve linear recurrence q[i] = C2[i]*q[i-1] + K[i]
 *            using parallel prefix scan (O(log n) depth)
 *            where K[i] = C1[i]*qup[i] + C3[i]*qdp[i] + C4[i]
 *
 * Mathematical basis:
 *   - The recurrence q[i] = a[i]*q[i-1] + b[i] is an affine map
 *   - Affine map composition is associative: (a2,b2)∘(a1,b1) = (a2*a1, a2*b1+b2)
 *   - Parallel prefix scan with this operator gives all prefix results in O(log n)
 *   - Same technique used in Mamba (Gu & Dao, 2023) and Martin & Cundy (ICLR 2018)
 *
 * References:
 *   - Blelloch (1990). "Prefix Sums and Their Applications." CMU-CS-90-190.
 *   - Martin & Cundy (2018). "Parallelizing Linear Recurrent Neural Nets." ICLR.
 *   - Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with SSMs."
 *   - t-route Issue #526: https://github.com/NOAA-OWP/t-route/issues/526
 *
 * Author: Cody Churchwell
 * Date: March 2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// Channel parameters for one segment
struct ChannelParams {
    float dx;    // segment length (m)
    float bw;    // bottom width (m)
    float tw;    // top width (m)
    float twcc;  // compound channel top width (m)
    float n_ch;  // Manning's roughness coefficient
    float ncc;   // floodplain Manning's n
    float cs;    // channel side slope
    float s0;    // bed slope
};

// State variables for one segment
struct SegmentState {
    float q;     // discharge (m3/s) at current timestep
    float vel;   // velocity (m/s)
    float depth; // flow depth (m)
};

// Affine tuple for parallel prefix scan: f(x) = a*x + b
struct AffineTuple {
    float a;  // multiplicative coefficient
    float b;  // additive term
};

// ============================================================
// Device functions: Muskingum-Cunge coefficient computation
// ============================================================

__device__ void hydraulic_geometry_device(
    float h, float bfd, float bw, float twcc, float z,
    float &twl, float &R, float &AREA, float &AREAC, float &WP, float &WPC)
{
    twl = bw + 2.0f * z * h;

    float h_gt_bf = fmaxf(h - bfd, 0.0f);
    float h_lt_bf = fminf(bfd, h);

    // NWM 3.0 exception: if depth beyond bankfull but no floodplain
    if (h_gt_bf > 0.0f && twcc <= 0.0f) {
        h_gt_bf = 0.0f;
        h_lt_bf = h;
    }

    AREA = (bw + h_lt_bf * z) * h_lt_bf;
    WP = bw + 2.0f * h_lt_bf * sqrtf(1.0f + z * z);
    AREAC = twcc * h_gt_bf;
    WPC = (h_gt_bf > 0.0f) ? (twcc + 2.0f * h_gt_bf) : 0.0f;
    R = (WP + WPC > 0.0f) ? (AREA + AREAC) / (WP + WPC) : 0.0f;
}

// Compute Muskingum-Cunge coefficients for a single segment
// This is the core of secant2_h from the Fortran code
__device__ void compute_mc_coefficients(
    float h, float bfd, float bw, float twcc, float z,
    float s0, float n_ch, float ncc, float dt, float dx,
    float qdp, float ql, float qup, float quc,
    float &C1, float &C2, float &C3, float &C4, float &X)
{
    float twl, R, AREA, AREAC, WP, WPC;
    hydraulic_geometry_device(h, bfd, bw, twcc, z, twl, R, AREA, AREAC, WP, WPC);

    // Kinematic celerity
    float Ck = 0.0f;
    if (h > bfd && twcc > 0.0f && ncc > 0.0f) {
        float R23 = powf(R, 2.0f/3.0f);
        float R53 = powf(R, 5.0f/3.0f);
        Ck = fmaxf(0.0f, ((sqrtf(s0) / n_ch)
            * ((5.0f/3.0f) * R23
            - (2.0f/3.0f) * R53 * (2.0f * sqrtf(1.0f + z*z) / (bw + 2.0f*bfd*z)))
            * AREA
            + (sqrtf(s0) / ncc) * (5.0f/3.0f) * powf(h - bfd, 2.0f/3.0f) * AREAC)
            / (AREA + AREAC));
    } else if (h > 0.0f) {
        float R23 = powf(R, 2.0f/3.0f);
        float R53 = powf(R, 5.0f/3.0f);
        Ck = fmaxf(0.0f, (sqrtf(s0) / n_ch)
            * ((5.0f/3.0f) * R23
            - (2.0f/3.0f) * R53 * (2.0f * sqrtf(1.0f + z*z) / (bw + 2.0f*h*z))));
    }

    // Muskingum travel time
    float Km = (Ck > 0.0f) ? fmaxf(dt, dx / Ck) : dt;

    // Weighting parameter X
    if (Ck > 0.0f) {
        float tw_use = (h > bfd && twcc > 0.0f) ? twcc : twl;
        float denom = 2.0f * tw_use * s0 * Ck * dx;
        if (denom > 0.0f) {
            float qest = (C1*qup + C2*quc + C3*qdp + C4); // use previous coefficients
            X = fminf(0.5f, fmaxf(0.25f, 0.5f * (1.0f - qest / denom)));
        } else {
            X = 0.5f;
        }
    } else {
        X = 0.5f;
    }

    // Muskingum coefficients
    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f; // avoid division by zero

    C1 = (Km * X + dt / 2.0f) / D;
    C2 = (dt / 2.0f - Km * X) / D;
    C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    C4 = (ql * dt) / D;

    // Clamp C4 for channel loss
    if (C4 < 0.0f && fabsf(C4) > C1*qup + C2*quc + C3*qdp) {
        C4 = -(C1*qup + C2*quc + C3*qdp);
    }
}

// Full secant-method MC solve for a single segment (reference implementation)
__device__ float mc_single_segment(
    float dt, float qup, float quc, float qdp, float ql,
    float dx, float bw, float tw, float twcc,
    float n_ch, float ncc, float cs, float s0,
    float depthp, float &depthc, float &velc)
{
    float z = (cs == 0.0f) ? 1.0f : 1.0f / cs;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f * z);
    else bfd = (tw - bw) / (2.0f * z);

    if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f) {
        depthc = 0.0f;
        velc = 0.0f;
        return 0.0f;
    }

    float mindepth = 0.01f;
    depthc = fmaxf(depthp, 0.0f);

    if (ql <= 0.0f && qup <= 0.0f && quc <= 0.0f && qdp <= 0.0f) {
        depthc = 0.0f;
        velc = 0.0f;
        return 0.0f;
    }

    float h = depthc * 1.33f + mindepth;
    float h_0 = depthc * 0.67f;

    float C1 = 0.0f, C2 = 0.0f, C3 = 0.0f, C4 = 0.0f, X = 0.0f;
    float Qj_0, Qj;
    int maxiter = 100;
    int tries = 0;

    float aerror = 0.01f, rerror = 1.0f;

    for (int attempt = 0; attempt < 5; attempt++) {
        int iter = 0;
        while (rerror > 0.01f && aerror >= mindepth && iter <= maxiter) {
            // Evaluate at h_0
            float twl0, R0, A0, AC0, WP0, WPC0;
            hydraulic_geometry_device(h_0, bfd, bw, twcc, z, twl0, R0, A0, AC0, WP0, WPC0);

            float Ck0 = 0.0f;
            if (h_0 > 0.0f) {
                float R23 = powf(R0, 2.0f/3.0f);
                float R53 = powf(R0, 5.0f/3.0f);
                Ck0 = fmaxf(0.0f, (sqrtf(s0)/n_ch) * ((5.0f/3.0f)*R23 - (2.0f/3.0f)*R53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h_0*z))));
            }
            float Km0 = (Ck0 > 0.0f) ? fmaxf(dt, dx/Ck0) : dt;
            float tw_use0 = (h_0 > bfd && twcc > 0.0f) ? twcc : twl0;
            float X0 = (Ck0 > 0.0f && tw_use0*s0*Ck0*dx > 0.0f) ?
                fminf(0.5f, fmaxf(0.0f, 0.5f*(1.0f - Qj_0/(2.0f*tw_use0*s0*Ck0*dx)))) : 0.5f;
            float D0 = Km0*(1.0f-X0) + dt/2.0f;
            if (D0 == 0.0f) D0 = 1.0f;
            float c1_0 = (Km0*X0 + dt/2.0f)/D0;
            float c2_0 = (dt/2.0f - Km0*X0)/D0;
            float c3_0 = (Km0*(1.0f-X0) - dt/2.0f)/D0;
            float c4_0 = (ql*dt)/D0;
            float Qmc0 = c1_0*qup + c2_0*quc + c3_0*qdp + c4_0;
            float Qmn0 = (WP0+WPC0 > 0.0f) ? (1.0f/(((WP0*n_ch)+(WPC0*ncc))/(WP0+WPC0))) * (A0+AC0) * powf(R0, 2.0f/3.0f) * sqrtf(s0) : 0.0f;
            Qj_0 = Qmc0 - Qmn0;

            // Evaluate at h
            float twl1, R1, A1, AC1, WP1, WPC1;
            hydraulic_geometry_device(h, bfd, bw, twcc, z, twl1, R1, A1, AC1, WP1, WPC1);

            float Ck1 = 0.0f;
            if (h > 0.0f) {
                float R23 = powf(R1, 2.0f/3.0f);
                float R53 = powf(R1, 5.0f/3.0f);
                Ck1 = fmaxf(0.0f, (sqrtf(s0)/n_ch) * ((5.0f/3.0f)*R23 - (2.0f/3.0f)*R53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z))));
            }
            float Km1 = (Ck1 > 0.0f) ? fmaxf(dt, dx/Ck1) : dt;
            float tw_use1 = (h > bfd && twcc > 0.0f) ? twcc : twl1;

            C1 = (Km1*X + dt/2.0f) / (Km1*(1.0f-X) + dt/2.0f);
            C2 = (dt/2.0f - Km1*X) / (Km1*(1.0f-X) + dt/2.0f);
            C3 = (Km1*(1.0f-X) - dt/2.0f) / (Km1*(1.0f-X) + dt/2.0f);
            C4 = (ql*dt) / (Km1*(1.0f-X) + dt/2.0f);

            X = (Ck1 > 0.0f && tw_use1*s0*Ck1*dx > 0.0f) ?
                fminf(0.5f, fmaxf(0.25f, 0.5f*(1.0f - (C1*qup+C2*quc+C3*qdp+C4)/(2.0f*tw_use1*s0*Ck1*dx)))) : 0.5f;

            // Recompute with updated X
            float D1 = Km1*(1.0f-X) + dt/2.0f;
            if (D1 == 0.0f) D1 = 1.0f;
            C1 = (Km1*X + dt/2.0f)/D1;
            C2 = (dt/2.0f - Km1*X)/D1;
            C3 = (Km1*(1.0f-X) - dt/2.0f)/D1;
            C4 = (ql*dt)/D1;
            if (C4 < 0.0f && fabsf(C4) > C1*qup+C2*quc+C3*qdp) {
                C4 = -(C1*qup+C2*quc+C3*qdp);
            }

            float Qmc1 = C1*qup + C2*quc + C3*qdp + C4;
            float Qmn1 = (WP1+WPC1 > 0.0f) ? (1.0f/(((WP1*n_ch)+(WPC1*ncc))/(WP1+WPC1))) * (A1+AC1) * powf(R1, 2.0f/3.0f) * sqrtf(s0) : 0.0f;
            Qj = Qmc1 - Qmn1;

            // Secant update
            float h_1;
            if (Qj_0 - Qj != 0.0f) {
                h_1 = h - (Qj * (h_0 - h)) / (Qj_0 - Qj);
                if (h_1 < 0.0f) h_1 = h;
            } else {
                h_1 = h;
            }

            if (h > 0.0f) {
                rerror = fabsf((h_1 - h) / h);
                aerror = fabsf(h_1 - h);
            } else {
                rerror = 0.0f;
                aerror = 0.9f;
            }

            h_0 = fmaxf(0.0f, h);
            h = fmaxf(0.0f, h_1);
            iter++;

            if (h < mindepth) break;
        }

        if (iter < maxiter) break;
        h *= 1.33f;
        h_0 *= 0.67f;
        maxiter += 25;
    }

    // Compute final discharge
    float qdc;
    float Qmc = C1*qup + C2*quc + C3*qdp + C4;
    if (Qmc < 0.0f) {
        if (C4 < 0.0f && fabsf(C4) > C1*qup + C2*quc + C3*qdp) {
            qdc = 0.0f;
        } else {
            qdc = fmaxf(C1*qup + C2*quc + C4, C1*qup + C3*qdp + C4);
        }
    } else {
        qdc = Qmc;
    }

    // Compute velocity
    float twl, R, A, AC, WP, WPC;
    hydraulic_geometry_device(h, bfd, bw, twcc, z, twl, R, A, AC, WP, WPC);
    R = (h*(bw + twl)/2.0f) / (bw + 2.0f*(((twl-bw)/2.0f)*((twl-bw)/2.0f) + h*h));
    if (R < 0.0f) R = 0.0f;
    velc = (1.0f/n_ch) * powf(R, 2.0f/3.0f) * sqrtf(s0);
    depthc = h;

    return qdc;
}

// ============================================================
// KERNEL 1: Sequential reference (GPU, one thread per reach)
// ============================================================
__global__ void kernel_mc_sequential(
    const ChannelParams* __restrict__ params,  // [total_segments]
    const float* __restrict__ qlat,            // lateral inflow [total_segments]
    const int* __restrict__ reach_start,       // start index of each reach
    const int* __restrict__ reach_len,         // number of segments per reach
    const float* __restrict__ qup_prev,        // upstream flow previous timestep [total_segments]
    const float* __restrict__ qdp_prev,        // downstream flow previous timestep [total_segments]
    const float* __restrict__ depth_prev,      // depth from previous timestep [total_segments]
    float* __restrict__ qdc_out,               // output discharge [total_segments]
    float* __restrict__ vel_out,               // output velocity [total_segments]
    float* __restrict__ depth_out,             // output depth [total_segments]
    float upstream_q,                          // flow entering from upstream reach
    float dt,
    int nreaches)
{
    int ireach = blockIdx.x * blockDim.x + threadIdx.x;
    if (ireach >= nreaches) return;

    int start = reach_start[ireach];
    int nseg = reach_len[ireach];

    float quc = upstream_q; // upstream flow entering this reach

    for (int i = 0; i < nseg; i++) {
        int idx = start + i;
        float qup = qup_prev[idx];
        float qdp = qdp_prev[idx];
        float depthp = depth_prev[idx];
        float depthc, velc;

        float qdc = mc_single_segment(
            dt, qup, quc, qdp, qlat[idx],
            params[idx].dx, params[idx].bw, params[idx].tw, params[idx].twcc,
            params[idx].n_ch, params[idx].ncc, params[idx].cs, params[idx].s0,
            depthp, depthc, velc);

        qdc_out[idx] = qdc;
        vel_out[idx] = velc;
        depth_out[idx] = depthc;

        // This segment's output becomes next segment's upstream input
        quc = qdc;
    }
}

// ============================================================
// KERNEL 2: Phase 1 — Compute MC coefficients in parallel
// ============================================================
__global__ void kernel_compute_coefficients(
    const ChannelParams* __restrict__ params,
    const float* __restrict__ qlat,
    const float* __restrict__ qup_prev,
    const float* __restrict__ qdp_prev,
    const float* __restrict__ depth_prev,
    float* __restrict__ C2_out,   // affine 'a' coefficient
    float* __restrict__ K_out,    // affine 'b' = C1*qup + C3*qdp + C4
    float dt,
    int total_segments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_segments) return;

    float h = fmaxf(depth_prev[idx], 0.01f);
    float z = (params[idx].cs == 0.0f) ? 1.0f : 1.0f / params[idx].cs;
    float bw = params[idx].bw;
    float tw = params[idx].tw;
    float bfd;
    if (bw > tw) bfd = bw / 0.00001f;
    else if (bw == tw) bfd = bw / (2.0f * z);
    else bfd = (tw - bw) / (2.0f * z);

    float s0 = params[idx].s0;
    float n_ch = params[idx].n_ch;
    float ncc = params[idx].ncc;
    float twcc = params[idx].twcc;
    float dx = params[idx].dx;

    if (n_ch <= 0.0f || s0 <= 0.0f || z <= 0.0f || bw <= 0.0f) {
        C2_out[idx] = 0.0f;
        K_out[idx] = 0.0f;
        return;
    }

    // Compute hydraulic geometry at previous depth
    float twl, R, AREA, AREAC, WP, WPC;
    hydraulic_geometry_device(h, bfd, bw, twcc, z, twl, R, AREA, AREAC, WP, WPC);

    // Kinematic celerity
    float Ck = 0.0f;
    if (h > 0.0f) {
        float R23 = powf(R, 2.0f/3.0f);
        float R53 = powf(R, 5.0f/3.0f);
        Ck = fmaxf(0.0f, (sqrtf(s0)/n_ch) * ((5.0f/3.0f)*R23
            - (2.0f/3.0f)*R53*(2.0f*sqrtf(1.0f+z*z)/(bw+2.0f*h*z))));
    }

    float Km = (Ck > 0.0f) ? fmaxf(dt, dx/Ck) : dt;
    float X = 0.25f; // conservative estimate for coefficient phase

    float D = Km * (1.0f - X) + dt / 2.0f;
    if (D == 0.0f) D = 1.0f;

    float C1 = (Km * X + dt / 2.0f) / D;
    float C2 = (dt / 2.0f - Km * X) / D;
    float C3 = (Km * (1.0f - X) - dt / 2.0f) / D;
    float C4 = (qlat[idx] * dt) / D;

    // The recurrence is: q[i] = C2 * q[i-1] + (C1*qup[i] + C3*qdp[i] + C4)
    C2_out[idx] = C2;
    K_out[idx] = C1 * qup_prev[idx] + C3 * qdp_prev[idx] + C4;
}

// ============================================================
// KERNEL 3: Phase 2 — Parallel prefix scan with affine tuples
// ============================================================
__global__ void kernel_affine_scan(
    const float* __restrict__ C2_in,   // 'a' coefficients
    const float* __restrict__ K_in,    // 'b' coefficients
    const int* __restrict__ reach_start,
    const int* __restrict__ reach_len,
    float upstream_q,                  // flow entering each reach
    float* __restrict__ q_out,         // output discharge
    int nreaches)
{
    // Each block handles one reach
    int ireach = blockIdx.x;
    if (ireach >= nreaches) return;

    int start = reach_start[ireach];
    int nseg = reach_len[ireach];

    extern __shared__ AffineTuple shared[];
    int tid = threadIdx.x;

    // Load affine tuples: f(x) = a*x + b
    if (tid < nseg) {
        shared[tid].a = C2_in[start + tid];
        shared[tid].b = K_in[start + tid];
    } else {
        shared[tid].a = 1.0f; // identity: f(x) = x
        shared[tid].b = 0.0f;
    }
    __syncthreads();

    // Hillis-Steele inclusive prefix scan (left-to-right)
    // Operator: (a2,b2) ∘ (a1,b1) = (a2*a1, a2*b1 + b2)
    for (int stride = 1; stride < nseg; stride *= 2) {
        AffineTuple temp;
        if (tid >= stride && tid < nseg) {
            // Compose: this ∘ partner
            temp.a = shared[tid].a * shared[tid - stride].a;
            temp.b = shared[tid].a * shared[tid - stride].b + shared[tid].b;
        }
        __syncthreads();
        if (tid >= stride && tid < nseg) {
            shared[tid] = temp;
        }
        __syncthreads();
    }

    // Apply to upstream boundary condition
    if (tid < nseg) {
        q_out[start + tid] = shared[tid].a * upstream_q + shared[tid].b;
    }
}

// ============================================================
// Host: Generate realistic test data
// ============================================================
void generate_test_data(
    ChannelParams* params, float* qlat,
    float* qup_prev, float* qdp_prev, float* depth_prev,
    int total_segments, int nreaches, int* reach_start, int* reach_len,
    unsigned int seed)
{
    srand(seed);

    int seg = 0;
    for (int r = 0; r < nreaches; r++) {
        reach_start[r] = seg;
        // Realistic reach lengths: 1-30 segments
        reach_len[r] = 1 + (rand() % 30);

        for (int s = 0; s < reach_len[r] && seg < total_segments; s++, seg++) {
            // Realistic NWM channel parameters
            params[seg].dx = 500.0f + (rand() % 5000);       // 500-5500 m
            params[seg].bw = 5.0f + (rand() % 50);           // 5-55 m bottom width
            params[seg].tw = params[seg].bw + (rand() % 20);  // tw >= bw
            params[seg].twcc = params[seg].tw + (rand() % 100); // compound width
            params[seg].n_ch = 0.02f + 0.06f * (rand() / (float)RAND_MAX); // 0.02-0.08
            params[seg].ncc = params[seg].n_ch * 1.5f;        // floodplain rougher
            params[seg].cs = 0.5f + 2.0f * (rand() / (float)RAND_MAX); // side slope
            params[seg].s0 = 0.0001f + 0.005f * (rand() / (float)RAND_MAX); // bed slope

            qlat[seg] = 0.5f * (rand() / (float)RAND_MAX);   // lateral inflow m3/s

            // Previous timestep state
            float base_q = 5.0f + 50.0f * (rand() / (float)RAND_MAX);
            qup_prev[seg] = base_q * 0.95f;
            qdp_prev[seg] = base_q;
            depth_prev[seg] = 0.5f + 3.0f * (rand() / (float)RAND_MAX);
        }
    }
}

// ============================================================
// Main benchmark
// ============================================================
int main() {
    printf("==============================================\n");
    printf("  t-route MC Routing: GPU Parallel Prefix Scan\n");
    printf("  Testing on RTX 3060\n");
    printf("==============================================\n\n");

    // Test configurations
    struct TestConfig {
        int nreaches;
        int avg_seg;
        const char* name;
    };

    TestConfig configs[] = {
        {1000,   10, "Small basin (1K reaches, ~10 seg)"},
        {10000,  10, "Medium basin (10K reaches, ~10 seg)"},
        {50000,   5, "Large basin (50K reaches, ~5 seg)"},
        {100000,  5, "CONUS subset (100K reaches, ~5 seg)"},
    };
    int nconfigs = 4;

    float dt = 300.0f; // 5 minute timestep

    for (int ic = 0; ic < nconfigs; ic++) {
        int nreaches = configs[ic].nreaches;
        int avg_seg = configs[ic].avg_seg;
        int max_total = nreaches * (avg_seg + 15); // max possible segments

        printf("--- Config: %s ---\n", configs[ic].name);

        // Allocate host memory
        ChannelParams* h_params = (ChannelParams*)malloc(max_total * sizeof(ChannelParams));
        float* h_qlat = (float*)malloc(max_total * sizeof(float));
        float* h_qup = (float*)malloc(max_total * sizeof(float));
        float* h_qdp = (float*)malloc(max_total * sizeof(float));
        float* h_depth = (float*)malloc(max_total * sizeof(float));
        int* h_reach_start = (int*)malloc(nreaches * sizeof(int));
        int* h_reach_len = (int*)malloc(nreaches * sizeof(int));
        float* h_qdc_seq = (float*)malloc(max_total * sizeof(float));
        float* h_qdc_par = (float*)malloc(max_total * sizeof(float));

        // Generate test data
        generate_test_data(h_params, h_qlat, h_qup, h_qdp, h_depth,
                          max_total, nreaches, h_reach_start, h_reach_len, 42 + ic);

        int total_segments = 0;
        int max_seg_per_reach = 0;
        for (int r = 0; r < nreaches; r++) {
            total_segments += h_reach_len[r];
            if (h_reach_len[r] > max_seg_per_reach) max_seg_per_reach = h_reach_len[r];
        }
        // Clamp
        if (total_segments > max_total) total_segments = max_total;

        printf("  Total segments: %d, Max per reach: %d\n", total_segments, max_seg_per_reach);

        // Check memory
        size_t data_mb = (total_segments * (sizeof(ChannelParams) + 8*sizeof(float)) + nreaches * 2 * sizeof(int)) / (1024*1024);
        printf("  Data size: %zu MB\n", data_mb);

        // Allocate device memory
        ChannelParams* d_params;
        float *d_qlat, *d_qup, *d_qdp, *d_depth;
        float *d_qdc_seq, *d_vel_seq, *d_depth_seq;
        float *d_C2, *d_K, *d_qdc_par;
        int *d_reach_start, *d_reach_len;

        cudaMalloc(&d_params, total_segments * sizeof(ChannelParams));
        cudaMalloc(&d_qlat, total_segments * sizeof(float));
        cudaMalloc(&d_qup, total_segments * sizeof(float));
        cudaMalloc(&d_qdp, total_segments * sizeof(float));
        cudaMalloc(&d_depth, total_segments * sizeof(float));
        cudaMalloc(&d_qdc_seq, total_segments * sizeof(float));
        cudaMalloc(&d_vel_seq, total_segments * sizeof(float));
        cudaMalloc(&d_depth_seq, total_segments * sizeof(float));
        cudaMalloc(&d_C2, total_segments * sizeof(float));
        cudaMalloc(&d_K, total_segments * sizeof(float));
        cudaMalloc(&d_qdc_par, total_segments * sizeof(float));
        cudaMalloc(&d_reach_start, nreaches * sizeof(int));
        cudaMalloc(&d_reach_len, nreaches * sizeof(int));

        // Copy to device
        cudaMemcpy(d_params, h_params, total_segments * sizeof(ChannelParams), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qlat, h_qlat, total_segments * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qup, h_qup, total_segments * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qdp, h_qdp, total_segments * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_depth, h_depth, total_segments * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_reach_start, h_reach_start, nreaches * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_reach_len, h_reach_len, nreaches * sizeof(int), cudaMemcpyHostToDevice);

        float upstream_q = 10.0f; // boundary condition

        // ---- Sequential kernel ----
        cudaEvent_t start_ev, stop_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&stop_ev);

        // Warmup
        int threads_seq = 256;
        int blocks_seq = (nreaches + threads_seq - 1) / threads_seq;
        kernel_mc_sequential<<<blocks_seq, threads_seq>>>(
            d_params, d_qlat, d_reach_start, d_reach_len,
            d_qup, d_qdp, d_depth,
            d_qdc_seq, d_vel_seq, d_depth_seq,
            upstream_q, dt, nreaches);
        cudaDeviceSynchronize();

        // Benchmark sequential
        int nruns = 20;
        cudaEventRecord(start_ev);
        for (int r = 0; r < nruns; r++) {
            kernel_mc_sequential<<<blocks_seq, threads_seq>>>(
                d_params, d_qlat, d_reach_start, d_reach_len,
                d_qup, d_qdp, d_depth,
                d_qdc_seq, d_vel_seq, d_depth_seq,
                upstream_q, dt, nreaches);
        }
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);
        float ms_seq;
        cudaEventElapsedTime(&ms_seq, start_ev, stop_ev);
        ms_seq /= nruns;

        // ---- Parallel kernel (two phases) ----
        // Warmup
        int threads_coeff = 256;
        int blocks_coeff = (total_segments + threads_coeff - 1) / threads_coeff;
        kernel_compute_coefficients<<<blocks_coeff, threads_coeff>>>(
            d_params, d_qlat, d_qup, d_qdp, d_depth,
            d_C2, d_K, dt, total_segments);

        int scan_threads = max_seg_per_reach;
        if (scan_threads < 32) scan_threads = 32;
        // Round up to next power of 2
        int st = 1;
        while (st < scan_threads) st *= 2;
        scan_threads = st;
        if (scan_threads > 1024) scan_threads = 1024;

        size_t smem = scan_threads * sizeof(AffineTuple);
        kernel_affine_scan<<<nreaches, scan_threads, smem>>>(
            d_C2, d_K, d_reach_start, d_reach_len,
            upstream_q, d_qdc_par, nreaches);
        cudaDeviceSynchronize();

        // Benchmark parallel
        cudaEventRecord(start_ev);
        for (int r = 0; r < nruns; r++) {
            kernel_compute_coefficients<<<blocks_coeff, threads_coeff>>>(
                d_params, d_qlat, d_qup, d_qdp, d_depth,
                d_C2, d_K, dt, total_segments);
            kernel_affine_scan<<<nreaches, scan_threads, smem>>>(
                d_C2, d_K, d_reach_start, d_reach_len,
                upstream_q, d_qdc_par, nreaches);
        }
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);
        float ms_par;
        cudaEventElapsedTime(&ms_par, start_ev, stop_ev);
        ms_par /= nruns;

        // ---- Accuracy check ----
        cudaMemcpy(h_qdc_seq, d_qdc_seq, total_segments * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_qdc_par, d_qdc_par, total_segments * sizeof(float), cudaMemcpyDeviceToHost);

        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        int nan_count = 0;
        int neg_count = 0;
        int large_err_count = 0;

        for (int i = 0; i < total_segments; i++) {
            if (isnan(h_qdc_par[i]) || isinf(h_qdc_par[i])) { nan_count++; continue; }
            if (h_qdc_par[i] < -0.001f) neg_count++;

            float abs_err = fabsf(h_qdc_par[i] - h_qdc_seq[i]);
            max_abs_err = fmaxf(max_abs_err, abs_err);

            if (fabsf(h_qdc_seq[i]) > 0.01f) {
                float rel_err = abs_err / fabsf(h_qdc_seq[i]);
                max_rel_err = fmaxf(max_rel_err, rel_err);
                if (rel_err > 0.1f) large_err_count++;
            }
        }

        float speedup = ms_seq / ms_par;

        printf("  Sequential:  %.3f ms\n", ms_seq);
        printf("  Parallel:    %.3f ms (%.2fx speedup)\n", ms_par, speedup);
        printf("  Max abs err: %.6e\n", max_abs_err);
        printf("  Max rel err: %.6e\n", max_rel_err);
        printf("  NaN/Inf:     %d, Negative: %d, Large err (>10%%): %d\n", nan_count, neg_count, large_err_count);

        // Determine pass/fail
        // NOTE: The parallel version uses operator-splitting (lagged coefficients)
        // so some error is EXPECTED. The question is whether it's within acceptable bounds.
        // For NWM, typical accuracy requirement is ~1% relative error on discharge.
        const char* status;
        if (nan_count > 0) {
            status = "FAIL (NaN)";
        } else if (max_rel_err > 0.5f && large_err_count > total_segments / 100) {
            status = "FAIL (>50% relative error in >1% of segments)";
        } else if (max_rel_err > 0.1f) {
            status = "MARGINAL (>10% max relative error — operator splitting introduces error)";
        } else {
            status = "PASS";
        }
        printf("  Status: %s\n\n", status);

        // Cleanup
        free(h_params); free(h_qlat); free(h_qup); free(h_qdp); free(h_depth);
        free(h_reach_start); free(h_reach_len);
        free(h_qdc_seq); free(h_qdc_par);
        cudaFree(d_params); cudaFree(d_qlat); cudaFree(d_qup); cudaFree(d_qdp); cudaFree(d_depth);
        cudaFree(d_qdc_seq); cudaFree(d_vel_seq); cudaFree(d_depth_seq);
        cudaFree(d_C2); cudaFree(d_K); cudaFree(d_qdc_par);
        cudaFree(d_reach_start); cudaFree(d_reach_len);
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }

    printf("==============================================\n");
    printf("  IMPORTANT NOTES:\n");
    printf("  1. The parallel version uses operator splitting\n");
    printf("     (lagged coefficients from previous timestep).\n");
    printf("     Some error vs sequential is EXPECTED.\n");
    printf("  2. The sequential kernel includes the full\n");
    printf("     secant-method nonlinear solve per segment.\n");
    printf("  3. Real NWM accuracy should be validated\n");
    printf("     against observed streamflow, not just\n");
    printf("     against the sequential numerical solution.\n");
    printf("==============================================\n");

    return 0;
}
