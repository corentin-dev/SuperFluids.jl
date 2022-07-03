# Numerical schemes

## Gross-Pitaevskii

### Imaginary time

```@docs
NumModelBackwardEuler(f::AbstractField, param::SuperFluids.AbstractParameters, Δt::Real, niter::Integer, freqbckp::Integer; nkrylov, tolkrylov, plantype)
NumModelBackwardEulerNoPrecond(f::AbstractField, param::SuperFluids.AbstractParameters, Δt::Real, niter::Integer, freqbckp::Integer; nkrylov, tolkrylov, plantype)
NumModelCrankNicolson
NumModelCrankNicolsonQuasiNewton
```

### Real time

```@docs
NumModelADI1
NumModelADI2
NumModelCrankNicolsonT
NumModelCrankNicolsonQuasiNewtonT
```

### External velocity

### Linear operator

```@docs
SuperFluids.prodA
```