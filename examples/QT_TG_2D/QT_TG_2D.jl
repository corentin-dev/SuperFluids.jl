use_saves = true # hide
#if you want to re-run the page, change to false # hide

# # 2D Quantum Turbulence
#
# ## Introduction
#
# To test quantum turbulence with this package, we chose to use an example from
# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project example/QT_TG_3D/QT_TG_3D.jl`
# if ``4`` is the desired number of processes.

# !!!note
#     This example, due to the way plots are done, is only valid for `mpirun -np 1`.

# Here, the topoology is a slab topology, meaning that ``y`` direction can be
# decomposed. The slab is transposed for some operations (FFT, finite differences, etc).

mpi_topo = SuperFluids.MPITopo1D()

# We use `Makie` for plots:
if mpi_topo.size == 1 # hide
using WGLMakie
WGLMakie.activate!()
end # hide

using JSServe # hide
Page(exportable=true, offline=true) # hide

# We setup the wanted discretization. We want ``n_x \times n_y = 128\times 128``.
# The domain bounds are set to ``[0,2\pi]\times[0,2\pi]``.

nx = 128
ny = 128

xrange = (0, 2*π)
yrange = (0, 2*π)

# With those informations, we can create `Grid`, using the constructor:

grid = Grid((nx,ny), (xrange,yrange), array_type=Array)

# A field, containing the simulation informations, is setup. The field
# is decomposed, hence the information of `mpi_topo` (which is not mandatory).
# In this specific simulation, for the Gross-Pitaevskii equation, we need
# a complex field `\phi`, and we specify it through the singleton `ComplexField()`.

field = Field(grid, ComplexField())

# ## First step: convergence to a stationary field
#
# In order to solve the Gross-Pitaevskii equation, we setup the parameters.
# ```math
# i \dfrac{dϕ}{dt} = -0.05 Δϕ + 40 |ϕ|²ϕ  + V(x)ϕ
# ```
#
# Here, `V` is a [`PotentialExternalVelocity`](@ref). It contains the following informations:
#
# - ``u_{\text{adv}_x} =  \sin(x)\cos(y)``
# - ``u_{\text{adv}_y} = -\cos(x)\sin(y)``
#
# and ``V = \dfrac{\left( u_{\text{adv}_x}^2 + u_{\text{adv}_y}^2 \right)}{α}``

param = GrossPitaevskiiParameters(
    coeffΔ = -0.05,
    β = 40,
    pot = PotentialExternalVelocity(field, α=4*40*0.05)
    )

# initialisation
init = InitExternalVelocity(field, param.coeffΔ, param.β)
initField!(init)

if mpi_topo.size == 1 # hide
surface(grid.x,grid.y,abs2.(field.ϕ))
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

# Convergence of energy

if mpi_topo.rank == 0 # hide
lines( [ l[2] for l in res ], [ l[6] for l in res ] )
end # hide

# Plot the solution

if mpi_topo.size == 1 # hide
surface(grid.x,grid.y,abs2.(field.ϕ))
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
niter_insta = 2500
freqbckp_insta = 10
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)

# And we start the solver.

res_insta = solve!(nummodel_insta, istart=niter, plot=false);
res_insta[end]

# Total energy should be (almost) constant in this case:

if mpi_topo.rank == 0 # hide
lines( [ l[2] for l in res_insta ], [ l[6] for l in res_insta ] )
end # hide

# Plot the solution.

if mpi_topo.size == 1 # hide
surface(grid.x,grid.y,abs2.(field_insta.ϕ))
end # hide