# # 3D Quantum Turbulence
#
# ## Introduction
#
# To test quantum turbulence with this package, we chose to use an example from
# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project example/QT_TG_3D/QT_TG_3D.jl`
# if $4$ is the desired number of processes.

# Here, the topoology is a pencil topology, meaning that $y$ and $z$ directions can be
# decomposed (if $n_\text{proc}$ is not prime). The pencil is transposed for some operations
# (FFT, finite differences, etc).

mpi_topo = SuperFluids.MPITopo2D()

# We use `Makie` for plots. For documentation purpose, we use `WGLMakie`
# that allows interactive javascript plots using WebGL, but you can also
# use `GLMakie` for interactive plots on your computer, or `CairoMakie`
# for static image plots.

if mpi_topo.size == 1 # hide
using WGLMakie
WGLMakie.activate!()
end # hide

using JSServe # hide
Page(exportable=true, offline=true) # hide

# We setup the wanted discretization. We want $n_x \times n_y \times n_z = 128\times 128\times 128$.
# The domain bounds are set to $[0,2\pi]\times[0,2\pi]\times[0,2\pi]$.

nx = 85
ny = 85
nz = 85

xrange = (0, 2*π)
yrange = (0, 2*π)
zrange = (0, 2*π)

# With those informations, we can create `Grid`, using the constructor:

grid = Grid((nx,ny,nz), (xrange,yrange,zrange), array_type=Array)

# A field, containing the simulation informations, is setup. The field
# is decomposed, hence the information of `mpi_topo` (which is not mandatory).
# In this specific simulation, for the Gross-Pitaevskii equation, we need
# a complex field ``ϕ``, and we specify it through the singleton `ComplexField()`.

field = Field(grid, ComplexField(), mpi_topo = mpi_topo)

# ## First step : convergence to a stationary field
#
# In order to solve the Gross-Pitaevskii equation, we setup the parameters.
# ```math
# i \dfrac{dϕ}{dt} = -0.05 Δϕ + 40 |ϕ|²ϕ  + V(x)ϕ
# ```
#
# Here, `V` is a `PotentialTaylorGreen`. It contains the following informations:
#
# - $u_{\text{adv}_x} =  \sin(x)\cos(y)\cos(z)$
# - $u_{\text{adv}_y} = -\cos(x)\sin(y)\cos(z)$
# - $u_{\text{adv}_z} = 0$
#
# and $V = \dfrac{\left( u_{\text{adv}_x}^2 + u_{\text{adv}_y}^2 + u_{\text{adv}_z}^2 \right)}{α}

param = GrossPitaevskiiParameters(
    coeffΔ = -0.075,
    β = 27,
    pot = PotentialTaylorGreen(field, α=4*27*0.075)
    )

# initialisation

init = InitExternalVelocity(field, param.coeffΔ, param.β)
initField!(init)

if mpi_topo.size == 1 # hide
volume(grid.x, grid.y, grid.z, abs2.(field.ϕ), algorithm=:iso, isovalue = 0.4, isorange = 0.2)
end # hide

# We setup the solver for the stationary, using:

Δt = 0.005
niter = 500
freqbckp = 100

# The solver uses the `NumModelExternalVelocity` `struct`.
# It uses the following scheme:
# ```math
# \dfrac{ϕ^{(n+1)}-ϕ^{(n)}}{Δt} = \dfrac{α}{2} Δϕ^{(n+1)} +
# \left(
#    \dfrac{α}{2} Δ - i u⃗_\text{adv} ⋅ ∇ - \dfrac{|u⃗_\text{adv}|²}{4α}
#    + β - β |ϕ^{(n)}|²
# \right) ϕ^{(n)}
# ```

nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)

# Solve the problem:

res = solve!(nummodel, plot=false);
res[end]

# Plot the solution

if mpi_topo.size == 1 # hide
volume(grid.x, grid.y, grid.z, abs2.(field.ϕ), algorithm=:iso, isovalue = 0.4, isorange = 0.2)
end # hide

# Cut at ``z = π``:

if mpi_topo.size == 1 # hide
surface(grid.x,grid.y,abs2.(field.ϕ[:,:,nz ÷ 2]))
end # hide

# ## Second step: unstationary restart

# We start by duplicating the field:

field_insta = Field(grid, ComplexField())
field_insta.ϕ .= field.ϕ;

# The parameters are similar to the previous stationary simulation, but
# the potential is different (``V=0``)

param_insta = GrossPitaevskiiParameters(
    coeffΔ = param.coeffΔ,
    β = param.β,
    pot = PotentialZero(field_insta)
    )

# We instantiate a `NumModelADI2` which corresponds to
# a second order Strangle scheme.

Δt_insta = Δt
niter_insta = 1000
freqbckp_insta = 10
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)

# And we start the solver.

res_insta = solve!(nummodel_insta, istart=niter, plot=false);
res_insta[end]

# Plot the solution.

if mpi_topo.size == 1 # hide
volume(grid.x, grid.y, grid.z, abs2.(field_insta.ϕ), algorithm=:iso, isovalue = 0.4, isorange = 0.2)
end # hide

# Cut at ``z = π``:

if mpi_topo.size == 1 # hide
surface(grid.x,grid.y,abs2.(field_insta.ϕ[:,:,nz ÷ 2]))
end # hide