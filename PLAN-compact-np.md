# Plan: non-periodic (boundary) compact scheme — Dirichlet & Neumann

Port of the GPS `cdl==2` one-sided compact operators (6th-order interior,
Lele-style boundary relaxation), cross-checked against Xcompact3d
(`derive.f90` `derx_11/12/21/22`).

## Status (v1 DONE — homogeneous Dirichlet, CPU)
- `CompactAxisNP` + `compact_setup_np(n, Δ, order)` in `Plans.jl` (row-varying
  tridiagonal LHS + `s/w` Thomas factors, precomputed closure weights; 1st and
  2nd derivative).
- CPU line kernels: overloads of `compact1line!`/`compact2line!`/`compact1line3!`/
  `compact2line3!` on `c::CompactAxisNP` (plain Thomas, no cyclic correction),
  so the existing plan dispatch is unchanged (dispatch by axis type ⇒ mixed
  periodic/Dirichlet axes work in one plan).
- API: `CompactPlan(bcs=(0|1, ...))` per axis — `0`/`:periodic`,
  `1`/`:dirichlet`. `:auto`/`:thomas` build a CPU Thomas plan; `:spectral` + a
  Dirichlet axis → error (a Fourier multiplier only exists for the periodic
  circulant operator).
- Validated: 1D/2D/3D interior 6th-order convergence (vs analytic sin/cos field
  vanishing on the walls); mixed axes; coefficients byte-identical to the GPS
  `cdl==2` blocks and consistent with Xcompact3d `ncl==2`; new testsets pass
  with the full suite (CPU + GPU).
- Remaining: (a) Neumann (`ncl==1`, Xcompact3d `derx_2x`, mirror symmetry),
  (b) CUDA NP kernel (CPU-only for now, clear error on GPU for `bcs=1`).

## Why
The compact scheme's original motivation in GPS was precisely to handle
non-periodic boundaries (periodic compact is just a slower FFT). This makes
`CompactPlan` a true alternative to the spectral plans for bounded domains.

## References on this machine
- GPS: `gps-master/GPS_SRC_HYB/GPS_compute_compact.f90`
  - `initial_compact`: all coefficients (boundary one-sided + interior)
  - `derx_compact`/`dery_compact` (1st, `cdl==2`), `derxx_compact`/`deryy_compact` (2nd, `cdl==2`)
  - `prepare_compact`: plain tridiagonal Thomas factorization (no cyclic corner)
- Xcompact3d: `Incompact3d/src/derive.f90`
  - `derx_11` (DD), `derx_12` (DN), `derx_21` (ND), `derx_22` (NN) for 1st
  - `derxx_11`, `derxx_12`, `derxx_21`, `derxx_22` for 2nd
  - `ncl=0` periodic, `ncl=1` Neumann (du/dn=0, mirror symmetry `npaire==0`),
    `ncl=2` Dirichlet (u=0 at the boundary node)
- GPS `cdl==2` is Dirichlet (GPS params: "0 = periodic, 2 = Dirichlet, only
  with compact scheme"), which matches Xcompact3d `ncl==2`.
- GPS GPU variant: `gps-master/GPS_SRC_GPU/` (check which solver it uses)

## Operator structure (per axis, per BC pair)
- Interior i=3..n-2: same 6th-order compact stencil as the periodic path.
- Boundary rows (i=1, i=n) and relaxed rows (i=2, i=n-1):
  - 1st derivative: i=1 → one-sided (−5u1+4u2−u3)/(2Δ); i=2 → (3/4Δ)(u3−u1)
    with LHS super(2)=1/4 (Lele relaxation); mirrored at the right end.
  - 2nd derivative: i=1 → (13u1−27u2+15u3−u4)/Δ² (order 4); i=2 → (6/5Δ²)(u3−2u2+u1)
    with LHS sub(1)=1/10; mirrored at the right end.
- LHS matrix: tridiagonal, row-varying (b=super? sub/diag/super per row),
  solved by plain Thomas (no cyclic correction) — `prepare_compact` in GPS.
- Dirichlet vs Neumann changes both the stencil rows and the LHS rows:
  Neumann uses the mirror symmetry du/dn=0 (Xcompact3d `npaire==0` branch).
- NOTE (GPS vs Xcompact3d): both codes agree on the interior stencil (a,b) and
  the boundary one-sided stencils, but they choose *different* Lele relaxation
  rows at i=2 / i=n−1 (GPS row 2 = (3/4Δ)(u3−u1); Xcompact3d row 2 =
  af(u3−u1)+bf(u4−u2)). Both are 6th-order consistent in the interior (the
  boundary error is a thin, decaying boundary layer). This port follows GPS
  exactly; the Xcompact3d match is used as a coefficient cross-check.

## Design (mirrors the existing periodic infrastructure) — as implemented
- `AbstractCompactAxis{R}`; `CompactAxis` (periodic) and `CompactAxisNP`
  (Dirichlet) subtypes. The line kernels dispatch on the axis type, which is
  what makes mixed axes work with unchanged plan dispatch.
- `CompactAxisNP` fields: `n, order, alpha, a, b` (interior), closure weights
  (`af1,bf1,cf1`, `afn,bfn,cfn` / `as1..ds1`, `asn..dsn`, `as2`), LHS
  `sub/sup` (n-vectors) and Thomas factors `s/w`. No `t`/`denom` (no cyclic
  corner).
- `compact_setup_np(n, Δ, order)`: builds the per-axis data (errors if
  `n < 8`). `compact_axis(n, Δ, order, bc)` selects periodic vs NP by BC.
- Kernels: CPU `compact*line!` overloads (done); CUDA NP kernels (v2) —
  `_compact_cu_1knp!` etc., one thread per line, plain Thomas.
- Plans: extended in place — `PlanCompact2D/3D` now carry
  `AbstractCompactAxis` per axis; `CompactPlan(bcs=...)` selects the axis
  constructor. (No separate `PlanCompactNP*` types.)
- `computeDerivatives!` dispatch is unchanged: it already solves along the
  leading dim per axis (x direct, y/z via transposes), so NP only swaps the
  per-line kernel + axis data.

## Grid/field changes
- The domain stays the same (`Grid((nx,ny),((xmin,xmax),(ymin,ymax))`): the
  boundary nodes are the first/last grid points (GPS convention).
- No field change: the field values at i=1 and i=n are the boundary values
  (0 for homogeneous Dirichlet; the scheme only needs the values, the BC is
  baked into the stencil coefficients for homogeneous BCs).
- Homogeneous BCs only in v1 (GPS `cdl==2` is homogeneous; inhomogeneous
  would add the BC value to the RHS — deferred).

## Scope (v1 = homogeneous Dirichlet, CPU; v2 = Neumann + CUDA)
- v1 DONE: 2D + 3D, 1st + 2nd derivatives, homogeneous Dirichlet (GPS
  `cdl==2` / Xcompact3d `ncl==2`), CPU (Thomas). Mixed periodic/Dirichlet axes.
- v2 (todo):
  - Neumann (Xcompact3d `ncl==1`, `npaire==0` mirror branch): new LHS rows +
    mirror-symmetric RHS rows, new `bcs` entry `2`/`:neumann`.
  - CUDA NP kernel (one thread per line, plain Thomas — simpler than the
    periodic kernel, no corner) to lift the CPU-only restriction.
- Spectral backend: NOT available for NP (no transform diagonalizes it) —
  `backend=:spectral` + NP axes → clear error.
- Model wiring: nothing new needed — `computeDerivatives!` is the only entry
  point, CN/BE models already take `plantype=`.

## Validation
1. 1D/2D/3D against an independent reference:
   - high-order one-sided FD (5th/6th-order) at the boundary,
   - or Chebyshev spectral differentiation on a mapped grid,
   - check interior convergence order (should approach 6 in the interior,
     lower near the boundary by design).
2. Cross-check coefficients against Xcompact3d `derx_11/22` etc.
3. GPU kernel vs CPU NP path: machine precision.
4. End-to-end: a bounded-domain test problem (e.g. Gaussian with Dirichlet
   walls, energy decay) vs FD6.

## Risks / open questions
- Lele relaxation (row 2/n-1, order-dropped RHS) is subtle: port it exactly
  from GPS, do not "improve" it; validate convergence in the interior.
- Neumann LHS rows: GPS only does `cdl==2` (its default BC); the Neumann
  variant comes from Xcompact3d `derx_2x` — port from there.
- The relaxed LHS makes the matrix non-symmetric and non-circulant: plain
  Thomas handles it, but the periodic precomputed-cyclic-correction trick
  does NOT apply (no rank-1 corner). Slightly more multipliers to precompute
  (b,c,f per BC pair instead of the uniform α).
