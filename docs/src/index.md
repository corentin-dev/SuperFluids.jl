# SuperFluids.jl

This is a package allowing simulation of superfluids. The first intention of this package is to solve the Gross-Pitaevskii equation to simulation Bose-Einstein Condensates. It evolved into a more advance package in order to solve Quantum-Turbulence. It now also solves the incompressible Navier-Stokes equations, the coupled Gross-Pitaevskii / Navier-Stokes two-fluid model of Parnaudeau et al. (NSGP), and the linear Hall-Vinen-Bekarevich-Khalatnikov (HVBK) two-fluid model. Derivatives are estimated through Fourier transformations or finite differences.

In order to be parallel (distributed), this package exploits intensively `PencilArrays`. Most of the package is written using broadcast, and is compatible with both CPU arrays (`Array`) and CUDA arrays (`CuArray`). It was not tested for other array type, yet. Every array creation is inferred from the `Grid` array type.

This package is authored by Corentin Lothodé, and largely inspired by GPS, a Fortran program by Philippe Parnaudeau (Parnaudeau, Suzuki & Sac-Epée, ISC-2015, Research Posters Session).

## Installation

For the moment, `SuperFluids.jl` is not in the julia registry. You need to get it by yourself:

```
pkg> add https://plmlab.math.cnrs.fr/lmrs/num/SuperFluids.jl
```

## Get sources

To get sources, you can clone the project:

```
git clone git@plmlab.math.cnrs.fr:lmrs/num/SuperFluids.jl.git
```

Start Julia :
```bash
julia --project=.
```

Import package :
```julia-repl
julia> using SuperFluids
```

## Build documentation

Simply run:
```bash
julia --project --color=yes docs/make.jl
```