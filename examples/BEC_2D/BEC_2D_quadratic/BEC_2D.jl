use_saves = true # hide
#if you want to re-run the page, change to false # hide

# # 2D Bose-Einstein Condensate

# We want to simulate a [Bose-Einstein Condensate](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_condensate)
# in rotation (BEC).

# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# ## Discretization

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project example/QT_TG_3D/QT_TG_3D.jl`
# if ``4`` is the desired number of processes.

# !!!note
#     This example, due to the way plots are done, is only valid for `mpirun -np 1`.

# Here, the topoology is a slab topology, meaning that ``y`` direction can be
# decomposed. The slab is transposed for some operations (FFT, finite differences, etc).

mpi_topo = SuperFluids.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) for plots. For documentation purpose, we use `WGLMakie`
# that allows interactive javascript plots using WebGL, but you can also
# use `GLMakie` for interactive plots on your computer, or `CairoMakie`
# for static image plots.

if mpi_topo.size == 1 # hide
using WGLMakie
WGLMakie.activate!()
end # hide

using JSServe # hide
Page(exportable=true, offline=true) # hide

# We setup the wanted discretization. We want $n_x \times n_y = 128\times 128$.
# The domain bounds are set to $[-12,12]\times[-12,12]$.
# With those informations, we can create [`Grid`](@ref), using the constructor:

nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

grid = Grid((nx,ny), (xrange,yrange))

# A [`Field`](@ref), containing the simulation informations, is setup. The [`Field`](@ref)
# is decomposed, hence the information of `mpi_topo` (which is not mandatory).
# In this specific simulation, for the Gross-Pitaevskii equation, we need
# a complex field ``ϕ``, and we specify it through the singleton `ComplexField()`.

field = Field(grid, ComplexField(), mpi_topo = mpi_topo)

# ## Quadratic potential

# We want to use the following potential [`PotentialQuadratic`](@ref):
# ```math
# V(x,y) = \dfrac{1}{2} (1-α)(γ_x x² +γ_y y²)
# ```

α = 0
γx = 1
γy = 1
pot = PotentialQuadratic(field, γx = γx, γy = γy)

# In order to solve the Gross-Pitaevskii equation, we setup the parameters
#  [`GrossPitaevskiiParameters`](@ref).
# ```math
# i \dfrac{dϕ}{dt} = -\dfrac{1}{2} Δϕ + 1000 |ϕ|²ϕ  + V(x)ϕ - i 0.9 L_z ϕ
# ```

param = GrossPitaevskiiParameters(
    β = 1000,
    Ω = 0.9,
    pot = pot
    )

# The initialization is done through [`InitThomasFermi`](@ref) which
# is at the moment specialized toward `PotentialQuadratic`.

init = InitThomasFermi(field, param.β, γx = γx, γy = γy)

# [`InitGauss`](@ref) is another possibility but is not used in this example:
# `init = [InitGauss](@ref)(field, Ω = param.Ω)`

# We call `initField!` to initialize effectively the field:

initField!(init)

# We plot the initial solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000. )

# The solver use implicit [`NumModelBackwardEuler`](@ref) scheme:

Δt = 0.01
niter = 1000
freqbckp = 10
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp, nkrylov = 500, tolkrylov = 1e-6)

# Many other solvers exists in `SuperFluids`, and you can check the example `BEC_2D_all.jl` to see
# how to use them.

# Launch the solver with [`solve!`](@ref):

if isfile("../../save/BEC_2D_quad-990.h5") && use_saves # hide
read!(nummodel.writers.writerList[2], prefix="../../save/BEC_2D_quad", istep=990) # hide
nummodel.niter = 10 # hide
res = solve!(nummodel, istart=990, plot=false) # hide
nummodel.niter = 1000 # hide
else # hide
res = solve!(nummodel, plot=false)
end # hide
res[end]

# We plot the convergence:

if mpi_topo.rank == 0 # hide
lines(  1:length(res), [ l[6] for l in res ], label="Total energy")
lines!( 1:length(res), [ l[3] for l in res ], label="Rotational energy")
lines!( 1:length(res), [ l[4] for l in res ], label="Potential+kinetic energy")
lines!( 1:length(res), [ l[5] for l in res ], label="Interaction energy")
axislegend()
current_figure()
end # hide

# Number of Krylov iteration per imaginary time step:

if mpi_topo.rank == 0 # hide
lines( 1:length(res), [ l[1] for l in res ], label="Number of Krylov iterations")
axislegend()
current_figure()
end # hide

# We plot the solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000. ) # hide

#using CSV # hide
#CSV.write("energy-noprecond-$(Δt).csv",DataFrame(res),delim=" ",header=false) # hide

# ## Quartic potential

# We change the rotation speed to ``Ω = 3.2``
# and the potential to the [`PotentialQuarticQuadratic`](@ref):
# ```math
# V(x,y) = \dfrac{1}{2} (1-α)(γ_x x² +γ_y y²) + κ_4 r^4
# ```
# where ``r=x²+y²``.

param.pot = PotentialQuarticQuadratic(field, γx = 0.0, γy = 0.0, κ4 = 0.01)

# Again, we solve the problem using ['solve!`](@ref):

if isfile("../../save/BEC_2D_quart-990.h5") && use_saves # hide
read!(nummodel.writers.writerList[2], prefix="../../save/BEC_2D_quart", istep=990) # hide
nummodel.niter = 10 # hide
res_quad = solve!(nummodel, istart=990, plot=false) # hide
nummodel.niter = 1000 # hide
else # hide
res_quad = solve!(nummodel, plot=false)
end # hide
res_quad[end]

# We plot the solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000. )