# # 2D Navier-Stokes: Taylor-Green vortex decay

# We simulate the decay of a 2D [Taylor-Green
# vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex) with the
# incompressible Navier-Stokes equations, using the semi-implicit Runge-Kutta
# scheme [`NumModelRK4Imp`](@ref).
# The nonlinear advection is treated explicitly (RK4), the viscosity
# implicitly through the exact spectral multiplier ``exp(-νΔt |k|²)``, and a
# 2/3-rule dealiasing is applied to the nonlinear term.

# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 4 julia --project examples/NS_2D/NS_2D.jl`

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

# We discretize the periodic box ``[0,2π]\times[0,2π]`` with
# $n_x \times n_y = 64\times 64$ points. For the Navier-Stokes equations the
# unknown is a **vector** field `u = (ux, uy)`, stored in a complex field
# (the FFT is a complex one):

nx = 64
ny = 64

grid = Grid((nx, ny), ((0, 2 * π), (0, 2 * π)))
field = Field(grid, ComplexField(); ndims=2)

# The only physical parameter is the kinematic viscosity `ν`
# (`ρ=1` is the default):

param = NavierStokesParameters(; ν=0.01)

# ## Solver

# We instantiate [`NumModelRK4Imp`](@ref):
# `Δt = 2×10⁻³`, `niter = 500` time steps (i.e. `t = 1`), a backup every 100 steps.

Δt = 0.002
niter = 500
freqbckp = 100

n = NumModelRK4Imp(field, param, Δt, niter, freqbckp)

# ## Initialisation

# [`taylor_green!`](@ref) fills `field` with the 2D Taylor-Green field, which is
# periodic and **exactly divergence-free**:
#
# ```math
# u = \left( \sin(k_x x)\cos(k_y y),\ -\dfrac{k_x}{k_y}\cos(k_x x)\sin(k_y y) \right),
# \qquad k = (2π/L_x, 2π/L_y)
# ```

taylor_green!(field, grid.x, grid.y)

# We can plot the initial vorticity
# ``ω = \partial_x u_y - \partial_y u_x``. It is computed spectrally:
# ``\hat{ω} = i (k_x \hat{u}_y - k_y \hat{u}_x)`` then an inverse FFT:

function vorticity2D(n) # hide
    g = SuperFluids.spectral_grid(n.plan)
    SuperFluids.mul_all!(n.u_hat, n.plan, n.f.u)
    ω_hat = @. 1im * (g.x * n.u_hat[2] - g.y * n.u_hat[1])
    ω = similar(n.f.ux)
    SuperFluids.ldiv_all!(ω, n.plan, ω_hat)
    return real(ω)
end # hide

if use_plots
    heatmap(grid.x, grid.y, vorticity2D(n))
end

# ## Time integration

# Solve the problem with [`solve!`](@ref):

res = solve!(n; plot=false)

# The kinetic energy $E = \tfrac12 \int |u|^2\,\mathrm{d}x\,\mathrm{d}y$
# should decay monotonically. We plot it against the total number of time steps:

if use_plots
    lines([l[2] for l in res], [l[6] for l in res]; label="kinetic energy E")
    axislegend()
    current_figure()
end

# ## Final state

# After $t=1$ the vortex has decayed towards large scales.
# We plot the final vorticity field:

if use_plots
    heatmap(grid.x, grid.y, vorticity2D(n))
end
