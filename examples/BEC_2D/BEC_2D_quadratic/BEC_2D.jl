# # 2D Bose-Einstein Condensate

# We want to simulate a [Bose-Einstein Condensate](https://en.wikipedia.org/wiki/Bose%E2%80%93Einstein_condensate)
# in rotation (BEC).

# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# ## Quadratic potential

# ### Discretization

# We setup the wanted discretization. `SuperFluids` uses only regular rectangular grids in order to use either FFT or finite differences.

# For this case, we want $n_x \times n_y = 128\times 128$.
# The domain bounds are set to $[-12,12]\times[-12,12]$.
# With those informations, we can create a [`Grid`](@ref), using the constructor:

nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

grid = Grid((nx, ny), (xrange, yrange))

# A [`Field`](@ref), containing the simulation informations, is setup. The [`Field`](@ref)
# is decomposed, hence the information of `mpi_topo` (which is not mandatory).
# In this specific simulation, for the Gross-Pitaevskii equation, we need
# a complex field ``ϕ``, and we specify it through the singleton `ComplexField()`.

field = Field(grid, ComplexField())

# ### Setup of the first simulation

# We want to use the following potential [`PotentialQuadratic`](@ref):
# ```math
# V(x,y) = \dfrac{1}{2} (1-α)(γ_x x² +γ_y y²)
# ```

α = 0
γx = 1
γy = 1
pot = PotentialQuadratic(field; γx=γx, γy=γy)

# In order to solve the Gross-Pitaevskii equation, we setup the parameters
#  [`GrossPitaevskiiParameters`](@ref).
# ```math
# i \dfrac{dϕ}{dt} = -\dfrac{1}{2} Δϕ + 1000 |ϕ|²ϕ  + V(x)ϕ - i 0.9 L_z ϕ
# ```

param = GrossPitaevskiiParameters(; β=1000,
                                  Ω=0.5,
                                  pot=pot)

# The initialization is done through [`InitThomasFermi`](@ref) which
# is at the moment specialized toward `PotentialQuadratic`.

init = InitThomasFermi(field, param.β; γx=γx, γy=γy)

# [`InitGauss`](@ref) is another possibility but is not used in this example:
# `init = [InitGauss](@ref)(field, Ω = param.Ω)`

# We call `initField!` to initialize effectively the field:

initField!(init)

# ### Plot initial solution

# We use [`Makie`](https://makie.juliaplots.org/stable/) for plots.

# For documentation purpose, we use `WGLMakie` # hide
# that allows interactive javascript plots using WebGL, but you can also # hide
# use `GLMakie` for interactive plots on your computer, or `CairoMakie` # hide
# for static image plots. # hide

# You can change those values if you want PNG or GL plots # hide

makie_style = "WGLMakie" # hide
#makie_style = "GLMakie" # hide
#makie_style = "CairoMakie" # hide

if makie_style == "WGLMakie" # hide
    using WGLMakie
    WGLMakie.activate!() # hide
    using JSServe # hide
    Page(; exportable=true, offline=true) # hide
elseif makie_style == "GLMakie" # hide
    using GLMakie # hide
    GLMakie.activate!() # hide
elseif makie_style == "CairoMakie" # hide
    using CairoMakie # hide
    CairoMakie.activate!() # hide
end # hide

# We plot the initial solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000.0)

# The solver use implicit [`NumModelBackwardEuler`](@ref) scheme:

Δt = 0.01
niter = 1000
freqbckp = 10
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp; nkrylov=500,
                                 tolkrylov=1e-6)

# Many other solvers exists in `SuperFluids`, and you can check the example `BEC_2D_all.jl` to see
# how to use them.

# ### Run the first simulation

# Launch the solver with [`solve!`](@ref):
res = solve!(nummodel; plot=false)
res[end]

# We plot the convergence:

lines(1:length(res), [l[6] for l in res]; label="Total energy")
lines!(1:length(res), [l[3] for l in res]; label="Rotational energy")
lines!(1:length(res), [l[4] for l in res]; label="Potential+kinetic energy")
lines!(1:length(res), [l[5] for l in res]; label="Interaction energy")
axislegend()
current_figure()

# Number of Krylov iteration per imaginary time step:

lines(1:length(res), [l[1] for l in res]; label="Number of Krylov iterations")
axislegend()
current_figure()

# ### Solution for the quadratic potential

# We plot the solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000.0)

# ## Quartic potential

# We change the potential to the [`PotentialQuarticQuadratic`](@ref):
# ```math
# V(x,y) = \dfrac{1}{2} (1-α)(γ_x x² +γ_y y²) + κ_4 r^4
# ```
# where ``r=x²+y²``.

param.pot = PotentialQuarticQuadratic(field; γx=0.0, γy=0.0, κ4=0.01)

# Again, we solve the problem using ['solve!`](@ref):

res_quad = solve!(nummodel; istart=niter, plot=false)
res_quad[end]

# We plot the solution:

#surface(grid.x, grid.y, abs2.(field.ϕ),  axis=(type=Axis3, viewmode = :fit)) # hide
surface(grid.x, grid.y, abs2.(field.ϕ) * 1000.0)
