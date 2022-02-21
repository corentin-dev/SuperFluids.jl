# SuperFluids.jl

This is a package allowing simulation of superfluids. The first intention of this package is to solve the Gross-Pitaevskii equation to simulation Bose-Einstein Condensates. It evolved into a more advance package in order to solve Quantum-Turbulence. Soon, it should be able to solve the HVBK and Navier-Stokes equations (incompressible). Derivatives are estimated through Fourier transformations or finite differences.

In order to be parallel (distributed), this package exploits intensively `PencilArrays`. Most of the package is written using broadcast, and is compatible with both CPU arrays (`Array`) and CUDA arrays (`CuArray`). It was not tested for other array type, yet. Every array creation is inferred from the `Grid` array type.

This package is authored by Corentin Lothodé, and largely inspired by GPS a Fortran program by Philippe Parnaudeau.

## Get package

```
git clone git@plmlab.math.cnrs.fr:lmrs/num/SuperFluids.jl.git
```

Start Julia :
```
julia --project=.
```

Import package :
```
using SuperFluids
```
