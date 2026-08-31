# Plan: compact finite-difference schemes

Reference implementation cross-checked against the Xcompact3d compact operators
(`schemes.f90`, `ifirstder==4` / `isecondder==4`): same coefficients.

## Status: DONE (integrated + tested)

## Goal (met)
1st and 2nd derivatives as **6th-order periodic compact** finite-difference
operators, wired into `computeDerivatives!` as an alternative to the FFT and
plain finite-difference plans, selected with `Plan(..., t=CompactPlan())`.

## What was done

### Step 1 — Prototype + validation (outside the package)
- 1D kernels + cyclic tridiagonal solve.
- **Solver validated to machine precision**: the effective matrix inverted by
  the Thomas factorization + forward/backward sweeps + cyclic corner
  correction is exactly the periodic circulant compact matrix
  (`max|M_eff − A_circ| = 2.2e-16`).
- **Pitfalls avoided**: the Thomas pass does not modify the super-diagonal
  input; the backward sweep uses the original super-diagonal `f[i]` (not
  `f[i+1]` nor `-b[i]`); copy the input before solving.
- **Measured convergence** (f = sin x + 0.5 cos 2y, Δx = 2π/N):
  - 1st derivative: **order 6** (6.02 / 6.01 / 6.00 / 5.99 for N = 16→128)
  - 2nd derivative: **order 6** (6.01 / 6.00 / 5.90 for N = 16→128)
  - apparent drop at N ≥ 256 = machine-epsilon floor (~1e-13).

### Step 2 — Integration: `PlanCompact`
- `src/Discretization/Plans.jl`:
  - `struct CompactPlan{B} <: PlanType` (exported) with a `backend` keyword
    (`:auto` default, `:thomas`, `:spectral`) — `CompactPlan()` ≙ `:auto`.
  - `abstract type AbstractCompactPlan{N}` + `PlanCompact2D` / `PlanCompact3D`
    (CPU Thomas path) + `PlanCompactGPU2D` / `PlanCompactGPU3D` (CUDA Thomas
    kernel path) + `PlanCompactFFT2D` / `PlanCompactFFT3D` (spectral-compact
    path, wrapping a `PlanFFT2D`/`PlanFFT3D` plus precomputed compact symbols).
  - `struct CompactAxis{R}`: precomputed multipliers for one axis
    (n, order, alpha, a, b, s, w, f, t, denom).
  - `compact_setup(n, Δ, order)`: builds the cyclic tridiagonal factorization
    plus the **precomputed correction vector** (RHS-independent — verified
    exact, diff = 0 on 50 random RHS).
  - `compact_multiplier(ξ, Δ, order)`: the Fourier symbol of the compact stencil
    (used by the spectral-compact path).
  - `Plan(f; t=CompactPlan())` branches for 2D and 3D, **auto-selecting the
    backend** on `A` (the field array type): `A === Array` → Thomas
    (`PlanCompact2D/3D`); GPU/non-Array → CUDA kernel (`PlanCompactGPU2D/3D`)
    when `t.backend ∈ (:auto, :thomas)`, spectral-compact
    (`PlanCompactFFT2D/3D`) for `t.backend === :spectral` (complex fields only).

### Step 2bis — GPU backend: CUDA Thomas kernel (`src/Discretization/compact_gpu.jl`)
- 4 `@cuda` kernels (1st/2nd derivative × 2D/3D), one thread per line, the
  compact stencil + Thomas sweep + cyclic correction done sequentially in the
  thread (validated to machine precision vs the CPU path).
- `CompactAxisGPU`: the same precomputed multipliers as `CompactAxis`, with
  the banded vectors uploaded to the device; built by `compact_setup_gpu`
  (which reuses `compact_setup`, guaranteeing identical coefficients).
- `using CUDA` is wrapped in `try/catch` at the top of `Plans.jl`: when CUDA
  cannot be loaded the kernel backend simply does not exist and a clear
  error is raised (suggesting `backend=:spectral`).
- Performance (RTX 3080, 256² complex, `computeDerivatives!`):
  kernel ≈ 3.5 ms vs spectral ≈ 0.3 ms vs FFT ≈ 0.3 ms — the sequential per-
  line sweep is ~10× slower than cuFFT for the *periodic* case. The kernel's
  advantages: works on **real and complex** fields (no complex-only
  limitation), no FFT scratch, and it is the only viable GPU path once the
  scheme becomes **non-periodic** (where no Fourier multiplier exists).
  `:auto` therefore picks the kernel on GPU for correctness generality;
  use `CompactPlan(backend=:spectral)` for the fast periodic-only path.

### Step 3 — `computeDerivatives!` dispatch
- `src/NumModels/derivatives.jl`: "Compact Finite Difference" section.
  - line kernels `compact1line!`/`compact2line!` (2D) and
    `compact1line3!`/`compact2line3!` (3D).
  - `_compact_solve!`: two Thomas sweeps + cyclic correction (in place).
  - four `computeDerivatives!(gf, plan::AbstractCompactPlan, ϕt)` methods:
    `GradientField2D/3D` + `GradientRotField2D/3D` (rx = y·dx, ry = −x·dy).
  - four `computeDerivatives!(gf, plan::PlanCompactGPU2D/3D, ϕt)` methods for
    the GPU kernel path: same layout as the CPU dispatch (x on the leading
    dimension; y/z through `PencilArray` transposes) but calling the CUDA line
    drivers on `parent(...)` raw arrays, with plan-preallocated scratch.
  - four `computeDerivatives!(gf, plan::PlanCompactFFT2D/3D, ϕt)` methods for
    the spectral-compact path (`mul_!`/`ldiv_!` with the precomputed compact
    symbols `p.T1x`/`p.T2x`/…). The compact symbol already contains the factor `i`
    of the derivative, so it replaces the `im*ξ` used by the spectral path.
  - Layout: x on the leading (local) dimension (full line, **no MPI
    communication**); y/z through `transpose!` (z → y → x, one permutation at
    a time).

### Step 4 — Package tests
- `test/runtests.jl`:
  - 4 compact CPU testsets (same Fourier-mode protocol as FFT/FD, `TOL_FD`):
    2D rotation=true (6 tests), 2D rotation=false (4), 3D rotation=true (8),
    3D rotation=false (6).
  - 1 `CompactPlan on GPU arrays` testset (skipped when no CUDA device, 18
    tests): 2D complex GPU kernel vs CPU Thomas on a non-constant field (all
    4 derivatives < 1e-10), 2D **RealField** kernel vs CPU (works, < 1e-10),
    3D complex non-constant field through the full x/y/z chain (all 6
    derivatives < 1e-10 vs CPU), and the `:spectral` backend (still available,
    still complex-only).
- **Full suite: 0 failures.** Measured errors (mode k=(2,3[,4])):
  - 2D: dx/ddy ~ 4e-9/3e-9, rx/ry exact.
  - 3D: dx/ddx ~ 2.7e-8/1.7e-8, dz/ddz ~ 1.8e-6/1.1e-6 (6th order).

## Decisions
- **Exact operator**: the solver is proven equivalent to the periodic circulant
  compact matrix; no generic Sherman–Morrison or closed-form alternative.
- **Sign convention**: the operators are pure derivatives (`∂x`, `∂xx`, ...);
  the physical model coefficient is applied by the model in `lapRot`.
- **Optimization**: the cyclic correction is precomputed once per axis
  (one `CompactAxis` per derivative order per axis), never recomputed per call.
- **Periodic only**: the non-periodic one-sided boundaries are deferred.
- **GPU backend (default): CUDA Thomas kernel.** A dedicated kernel solves the
  compact relation per line on the device (one thread per line, sequential
  sweep). It is dtype-agnostic (real *and* complex fields), needs no FFT
  scratch, and is the only viable GPU path for the (deferred) non-periodic
  case. It is ~10× slower than cuFFT for the periodic case (sequential sweep),
  so `CompactPlan(backend=:spectral)` remains available as the fast
  periodic-only option. `CompactPlan` auto-selects: Thomas on CPU, CUDA kernel
  on GPU. End-to-end check: CN GP model on GPU with the kernel plan matches
  the FFT plan to the compact truncation error (~1e-3 on a 64² Gaussian after
  5 steps).

## Remaining (optional, deferred)
- Public docs: document the `t=CompactPlan()` option, order 6, limits.
- Non-periodic boundary case.

## GPU (CUDA) support — via the CUDA Thomas kernel (default) or spectral (opt-in)
Verified on an RTX 3080 (`Grid(...; array_type=CuArray)`):
- **FFT**: works (existing GPU examples).
- **Finite difference (order 6)**: works; results are bit-identical to the CPU
  (row-slice operations are host-legal on GPU arrays).
- **Compact**: **works, on-device.** `CompactPlan()` on a GPU grid builds the
  **CUDA kernel** backend (`PlanCompactGPU2D/3D`):
  - 4 kernels in `src/Discretization/compact_gpu.jl` (1st/2nd derivative ×
    2D/3D); one thread per line performs the stencil, the Thomas sweep and
    the cyclic correction; the banded multipliers (`CompactAxisGPU`) are
    precomputed on the host (`compact_setup`) and uploaded once per axis;
  - `computeDerivatives!` runs entirely on device (kernels + `PencilArray`
    transposes), with plan-preallocated scratch (no per-call GPU malloc);
  - **identical to the CPU Thomas path to machine precision** (2D and 3D,
    complex and real fields, checked against the CPU Thomas path);
  - unlike the FFT-based plans, **RealField works** (the kernel is
    dtype-agnostic).
- **Spectral backend (opt-in)**: `CompactPlan(backend=:spectral)` builds
  `PlanCompactFFT2D/3D` — the periodic compact operator is a *circulant*
  pentadiagonal matrix, hence a Fourier multiplier: `IDFT(T(ξ)·FFT(ϕ))` with
  the compact symbol `T` (`compact_multiplier`), reusing the standard FFT
  machinery (fully GPU-native, cuFFT). Also identical to the Thomas result to
  machine precision. This is the **fastest** GPU path for the periodic case
  (~10× faster than the kernel at N=256) but requires a **complex** field
  (clear construction-time error on `RealField`), like the standard `FFTPlan`.
- `CompactPlan(backend=...)` summary: `:auto` (default) → Thomas on CPU, CUDA
  kernel on GPU; `:thomas` → force the kernel (CPU or GPU); `:spectral` →
  Fourier multiplier (GPU, complex only).

## Model compatibility (verified)
Compact derivatives are wired **only** through `computeDerivatives!` (dispatch
on the plan type).

| Model | Derivative path | Compact? | Note |
|---|---|---|---|
| GP `NumModelCrankNicolson` (+ QuasiNewton, T, QNT) | `lapRot` → `computeDerivatives!` | ✅ tested | `plantype=` keyword added to the 4 constructors; CN compact vs FFT: 1.2e-7 difference after 8 steps |
| GP `NumModelBackwardEuler` (+ NoPrecond) | `lapRot` → `computeDerivatives!` | ✅ tested | smoke test, 15 steps: mass 9.0, 3.5e-4 difference vs FFT |
| GP `NumModelGPRK` | `lapRot` (RHS) + FFT dealiasing (`mul_all!`) | ⚠️ partial | the RHS goes through the plan (compact OK), but the 2/3-rule dealiasing is **FFT-only** (`mul_all!`) and has no FD fallback either — would need work to support GPRK+compact |
| GP `NumModelADI1/2` | `mul_x!`/`ldiv_y!` **direct** (FFT) | ❌ | FFT-plan-only methods; no FD fallback in the package → out of scope (same as the plain FD today) |
| GP `NumModelExternalVelocity` | `mul_all!` **direct** (FFT) | ❌ | same |
| `NumModelHVBK` | fully spectral (curl/div/laplacian `i·k×û` in Fourier space) | ❌ | does not use `computeDerivatives!` |
| `NumModelNSGP` | spectral (NS laplacian `exp(-νΔt·k²)`, GP gradient spectral) | ❌ | same |
| NS `NumModelRK4Imp` | spectral (curl/divergence in Fourier space) | ❌ | same |

Note: `computeDerivatives!(GradientCurlField3D, plan)` exists only for
`AbstractFFTPlan` (a compact curl is not part of this port), which is why the
vector models stay FFT.

The `plantype::PlanType=FFTPlan()` keyword is available on:
`NumModelBackwardEuler`, `NumModelBackwardEulerNoPrecond`,
`NumModelCrankNicolson`, `NumModelCrankNicolsonQuasiNewton`,
`NumModelCrankNicolsonT`, `NumModelCrankNicolsonQuasiNewtonT`.
Usage: `NumModelCrankNicolson(f, p, Δt, n, freq; plantype=CompactPlan())`.
The ADI/ExternalVelocity/GPRK constructors do not have it (their direct
spectral operators make them non-compact anyway).

## Coefficients (periodic) — used by `compact_setup`
- 1st: `alpha=1/3`, `a=(7/9)/Δx`, `b=(1/36)/Δx`
- 2nd: `alpha=2/11`, `a=(12/11)/Δx²`, `b=(3/44)/Δx²`

## Cyclic solve
- Tridiagonal: `diag=1` (interior), `diag(1)=2`, `diag(n)=1+α²`,
  sub=super=α (`super(n)=0`).
- Thomas factorization: `w=c`; `s(i)=b(i-1)/w(i-1)`;
  `w(i)=w(i)-f(i-1)s(i)`; `w=1/w` — the input `f` is not modified.
- Forward: `r(i) -= r(i-1)s(i)`; backward: `r(i) = (r(i) - f(i)r(i+1))w(i)`
  (f = super-diagonal).
- Corner correction: `sx = (r(1) - α r(n)) / (1 + t(1) - α t(n))`;
  `r(i) -= sx t(i)`.
- `t` (correction vector) is precomputed: `t(1)=-1, t(n)=α`, then the same
  sweeps (RHS-independent).

## Files
- `src/Discretization/Plans.jl` — `CompactPlan`, `CompactAxis`, `compact_setup`,
  `compact_multiplier`, `PlanCompact2D/3D` (CPU), `PlanCompactGPU2D/3D` (CUDA
  kernel), `PlanCompactFFT2D/3D` (spectral), and the auto-selecting
  `Plan(f; t=CompactPlan())` branch (with the `try using CUDA` guard).
- `src/Discretization/compact_gpu.jl` — the CUDA kernels, the CUDA line
  drivers, `CompactAxisGPU` and `compact_setup_gpu`.
- `src/NumModels/derivatives.jl` — line kernels + `computeDerivatives!` dispatch
  (the `AbstractCompactPlan`/Thomas, the `PlanCompactGPU*/3D`/CUDA and the
  `PlanCompactFFT*/3D`/spectral methods).
- `src/NumModels/GrossPitaevskii/crank-nicolson.jl` — `plantype=` keyword on the
  4 CN constructors.
- `test/runtests.jl` — 4 compact CPU testsets + 1 `CompactPlan on GPU arrays`
  testset.
- `benchmarks/bench_derivatives.jl` — FFT vs FD vs compact benchmark.
