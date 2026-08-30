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
  - `struct CompactPlan <: PlanType` (exported).
  - `abstract type AbstractCompactPlan{N}` + `PlanCompact2D` / `PlanCompact3D`.
  - `struct CompactAxis{R}`: precomputed multipliers for one axis
    (n, order, alpha, a, b, s, w, f, t, denom).
  - `compact_setup(n, Δ, order)`: builds the cyclic tridiagonal factorization
    plus the **precomputed correction vector** (RHS-independent — verified
    exact, diff = 0 on 50 random RHS).
  - `Plan(f; t=CompactPlan())` branches for 2D and 3D (creates `pen_y`/`pen_z`
    + scratch buffers).

### Step 3 — `computeDerivatives!` dispatch
- `src/NumModels/derivatives.jl`: "Compact Finite Difference" section.
  - line kernels `compact1line!`/`compact2line!` (2D) and
    `compact1line3!`/`compact2line3!` (3D).
  - `_compact_solve!`: two Thomas sweeps + cyclic correction (in place).
  - four `computeDerivatives!(gf, plan::AbstractCompactPlan, ϕt)` methods:
    `GradientField2D/3D` + `GradientRotField2D/3D` (rx = y·dx, ry = −x·dy).
  - Layout: x on the leading (local) dimension (full line, **no MPI
    communication**); y/z through `transpose!` (z → y → x, one permutation at
    a time).

### Step 4 — Package tests
- `test/runtests.jl`: 4 compact testsets (same Fourier-mode protocol as
  FFT/FD, `TOL_FD`):
  - 2D rotation=true (6 tests), 2D rotation=false (4), 3D rotation=true (8),
    3D rotation=false (6).
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

## Remaining (optional, deferred)
- Public docs: document the `t=CompactPlan()` option, order 6, limits.
- Non-periodic boundary case.

## GPU (CUDA) support
Verified on an RTX 3080 (`Grid(...; array_type=CuArray)`):
- **FFT**: works (existing GPU examples).
- **Finite difference (order 6)**: works; results are bit-identical to the CPU
  (row-slice operations are host-legal on GPU arrays).
- **Compact**: **not supported** — the Thomas sweep needs scalar indexing on the
  local line (`u[i]`), which GPU arrays reject from the host (`assertscalar`).
  This is intrinsic to a sequential compact solve. `Plan(...; t=CompactPlan())`
  on a GPU grid now raises a clear error at construction time (2D and 3D).

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
  compact plans.
- `src/NumModels/derivatives.jl` — line kernels + `computeDerivatives!` dispatch.
- `src/NumModels/GrossPitaevskii/crank-nicolson.jl` — `plantype=` keyword on the
  4 CN constructors.
- `test/runtests.jl` — 4 compact testsets.
- `benchmarks/bench_derivatives.jl` — FFT vs FD vs compact benchmark.
