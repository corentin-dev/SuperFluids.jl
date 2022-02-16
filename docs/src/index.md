# SuperFluids.jl

This is a package allowing simulation of superfluids. The first intention of this package is to solve the Gross-Pitaevskii equation to simulation Bose-Einstein Condensates. It evolved into a more advance package in order to solve Quantum-Turbulence. Soon, it should be able to solve the HVBK and Navier-Stokes equations (incompressible). Derivatives are estimated through Fourier transformations or finite differences.

In order to be parallel (distributed), this package exploits intensively `PencilArrays`. Most of the package is written using broadcast, and is compatible with both CPU arrays (`Array`) and CUDA arrays (`CuArray`). It was not tested for other array type, yet. Every array creation is inferred from the `Grid` array type.

## Quick start

In order to have any simulation, you need to create a `Grid` and a `Field`. Then, you have to create parameters related to the simulation you want to perform, a numerical model which will be able to solve either time dependant or steady problems. You can be helped by initializers, read a field from file or use your custom function.

### Grid

```@docs
Grid
```

### Field

```@docs
Field
```
