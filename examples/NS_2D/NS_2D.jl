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

mpi_topo = SuperFluids.MPITopo1D()

# We use [`Makie`](https://makie.juliaplots.org/stable/) (with `WGLMakie`,
# which produces interactive WebGL plots in the documentation) for the plots:

using WGLMakie # hide
WGLMakie.activate!() # hide
WGLMakie.Bonito.Page(; exportable=true, offline=true) # hide

# ## Discretization

# We discretize the periodic box $[0,2π]\times[0,2π]$ with
# $n_x \times n_y = 64\times 64$ points. For the Navier-Stokes equations the
# unknown is a **vector** field `u = (ux, uy)`, stored in a complex field
# (the FFT is a complex one):

nx = 64
ny = 64

grid = Grid((nx, ny), ((0, 2 * π), (0, 2 * π)))
field = Field(grid, ComplexField(); ndims=2)

# The only physical parameter is the kinematic viscosity `ν`
# (`ρ=1` is the default):

param = NavierStokesParameters(; ν=0.1)

# ## Solver

# We instantiate [`NumModelRK4Imp`](@ref):
# `Δt = 2×10⁻³`, `niter = 1000` time steps (i.e. `t = 2`), a backup every 100 steps.

Δt = 0.002
niter = 1000
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

ω0 = vorticity2D(n)
ωmax = maximum(abs.(ω0))  # hide  (fixed color range so the decay is visible)
heatmap(grid.x, grid.y, ω0; colorrange=(-ωmax, ωmax))

# ## Time integration

# Solve the problem with [`solve!`](@ref):

res = solve!(n; plot=false)

# The kinetic energy $E = \tfrac12 \int |u|^2\,\mathrm{d}x\,\mathrm{d}y$
# should decay monotonically. We plot it against the total number of time steps:

e_plot = lines([l[2] for l in res], [l[6] for l in res]; label="kinetic energy E")
axislegend()
e_plot

# ## Final state

# After $t=2$ the kinetic energy has decayed by a factor $e^{-4νt} ≈ 0.45$
# and the vortex towards large scales. We plot the final vorticity field
# (same color range as the initial one, so the decay is visible):

heatmap(grid.x, grid.y, vorticity2D(n); colorrange=(-ωmax, ωmax))
