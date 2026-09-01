# Plan

Plans describes the way derivatives can be computed. They also store temporary arrays distributed along each direction.

## Constructors

```@docs
Plan
```

## Structures

### FFT Plans

```@docs
SuperFluids.PlanFFT2D
```

```@docs
SuperFluids.PlanFFT3D
```

### Finite Difference Plans

```@docs
SuperFluids.PlanFD2D
```

```@docs
SuperFluids.PlanFD3D
```

### Compact Finite Difference Plans

The compact schemes resolve derivatives to 6th order at a cost comparable to
a second-order stencil, at the price of a tridiagonal line solve per
direction. The boundary condition on each axis is set through the `bcs`
keyword of [`SuperFluids.CompactPlan`](@ref): `0`/`:periodic` (default),
`1`/`:dirichlet` (homogeneous Dirichlet) or `2`/`:neumann` (homogeneous
Neumann, even-mirror closure). Axes may be mixed, e.g. `bcs = (2, 0)` bounds
`x` with Neumann and keeps `y` periodic.

```@docs
SuperFluids.CompactPlan
```

```@docs
SuperFluids.PlanCompact2D
```

```@docs
SuperFluids.PlanCompact3D
```