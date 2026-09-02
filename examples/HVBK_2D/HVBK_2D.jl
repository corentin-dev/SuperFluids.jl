# # 2D HVBK two-fluid (linear)

# We simulate the linear [HVBK two-fluid
# model](https://en.wikipedia.org/wiki/Two-fluid_model) with
# [`NumModelHBVK`](@ref). It describes a classical normal fluid `u_n` and an
# inviscid superfluid `u_s`, each advecting its own vorticity and coupled by the
# **linear** mutual-friction force ``F = -\tfrac12 r_b |w| (u_n - u_s)``
# where ``w = u_n - u_s`` is the relative velocity. The normal-fluid viscosity
# is treated implicitly.

# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project examples/HVBK_2D/HVBK_2D.jl`

# !!!note
#     This example, due to the way plots are done, is only valid for `mpirun -np 1`.

mpi_topo = SuperFluids.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) for plots. For documentation purpose, we use `WGLMakie`
# that allows interactive javascript plots using WebGL, but you can also
# use `GLMakie` for interactive plots on your computer, or `CairoMakie`
# for static image plots.

# You can change those values if you want PNG or GL plots

makie_style = "WGLMakie"
#makie_style = "GLMakie"
#makie_style = "CairoMakie"

if mpi_topo.size == 1 # hide
    use_plots = true # hide
else # hide
    use_plots = false # hide
end # hide

if use_plots # hide
    if makie_style == "WGLMakie"
        using WGLMakie
        WGLMakie.activate!()
        WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide
    elseif makie_style == "GLMakie"
        using GLMakie
        GLMakie.activate!()
    elseif makie_style == "CairoMakie"
        using CairoMakie
        CairoMakie.activate!()
    end
end # hide

# ## Discretization

# The two fluids share a periodic ``48\times36`` grid on $[-4,4]\times[-5,5]$.
# Each is a complex vector field (``u_n`` and ``u_s``):

grid = Grid((48, 36), ((-4, 4), (-5, 5)))
fn = Field(grid, ComplexField(); ndims=2)   # normal fluid
fs = Field(grid, ComplexField(); ndims=2)   # superfluid

# ## Initialisation

# Both fluids start from divergence-free Taylor-Green-like background modes with
# slightly different wavelengths, so the relative motion (and hence the mutual
# friction) is non-trivial:

kx = 2π / grid.Lx; ky = 2π / grid.Ly
@. fn.ux = sin(ky * fn.y) * cos(kx * fn.x)
@. fn.uy = -cos(ky * fn.y) * sin(kx * fn.x)
@. fs.ux = 0.7 * cos(ky * fs.y) * sin(kx * fs.x)
@. fs.uy = 0.7 * sin(ky * fs.y) * cos(kx * fs.x)

# ## Parameters

# [`HBVKParameters`](@ref): `ν` the normal viscosity, `νs` the (zero) superfluid
# viscosity, `rb` the mutual-friction coefficient, `ρn`/`ρs` the two masses.

param = HBVKParameters(; ν=0.01, νs=0.001, rb=1.5, ρn=1.0, ρs=1.0)

# ## Solver

# We integrate with the second-order scheme (`stepper="RK2"`):

Δt = 0.01
niter = 400
freqbckp = 100

model = NumModelHBVK(fn, fs, param, Δt, niter, freqbckp; stepper="RK2")

# The model stores the **scalar** (2D) vorticity of each fluid in
# `model.un_vort[2]` and `model.us_vort[2]`. We plot them.
#
# Initial normal-fluid vorticity:

if use_plots
    heatmap(grid.x, grid.y, real(model.un_vort[2]))
end

# Initial superfluid vorticity:

if use_plots
    heatmap(grid.x, grid.y, real(model.us_vort[2]))
end

# ## Time integration

# Solve with [`solve!`](@ref). The mutual friction relaxes the relative velocity
# toward zero while the normal viscosity dissipates the normal vorticity:

res = solve!(model; plot=false)

# After $t = 4$ the two vorticities have evolved; we plot the final state:

# Final normal-fluid vorticity:

if use_plots
    heatmap(grid.x, grid.y, real(model.un_vort[2]))
end

# Final superfluid vorticity:

if use_plots
    heatmap(grid.x, grid.y, real(model.us_vort[2]))
end
