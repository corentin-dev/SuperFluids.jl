# SuperFluids.jl

This is a package allowing simulation of superfluids. The first intention of this package is to solve the Gross-Pitaevskii equation to simulation Bose-Einstein Condensates. It evolved into a more advance package in order to solve Quantum-Turbulence. It now also solves the incompressible Navier-Stokes equations, the coupled Gross-Pitaevskii / Navier-Stokes two-fluid model of Parnaudeau et al. (NSGP), and the linear Hall-Vinen-Bekarevich-Khalatnikov (HVBK) two-fluid model. Derivatives are estimated through Fourier transformations or finite differences.

In order to be parallel (distributed), this package exploits intensively `PencilArrays`. Most of the package is written using broadcast, and is compatible with both CPU arrays (`Array`) and CUDA arrays (`CuArray`). It was not tested for other array type, yet. Every array creation is inferred from the `Grid` array type.

This package is authored by Corentin Lothodé, and largely inspired by GPS a Fortran program by Philippe Parnaudeau.

## Models

- **Gross-Pitaevskii**: imaginary time (backward Euler, Crank-Nicolson, ADI) and real time (time-dependent Crank-Nicolson), plus an external-velocity solver. See `NumModelBackwardEuler`, `NumModelCrankNicolson`, `NumModelADI1`, `NumModelCrankNicolsonT`, `NumModelExternalVelocity`.
- **Navier-Stokes** (incompressible, 2D and 3D): semi-implicit RK4 solver `NumModelRK4Imp` (vorticity-advection form, exact implicit viscous multiplier, spectral Helmholtz projection).
- **NSGP** (coupled GP/Navier-Stokes two-fluid model of Parnaudeau et al., `NumModelNSGP`): a non-stationary Gross-Pitaevskii equation for the superfluid wavefunction coupled, through the Coste coupling, to a forced Navier-Stokes equation for the normal fluid (one-way or two-way).
- **HVBK** (linear two-fluid model, `NumModelHBVK`): two incompressible velocity fields coupled by the linear mutual friction `F = -1/2 rb |∇×u_s| (u_n - u_s)`, total momentum conserved.

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

## `MPI` and `HDF5`

This project uses  `MPIPreferences.jl` to setup `MPI.jl`. In order to use it, you can create a file named `LocalPreferences.toml` containing:

```
[MPIPreferences]
_format = "1.0"
abi = "OpenMPI"
binary = "system"
libmpi = "libmpi"
mpiexec = "mpiexec"
```

To the same `LocalPreferences.toml`, you can add the following informations:

```
[HDF5_jll]
libhdf5_path = "/usr/lib/libhdf5.so"
libhdf5_hl_path = "/usr/lib/libhdf5_hl.so"
```
