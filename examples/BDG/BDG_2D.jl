# # Bogoliubov-de Gennes excitation spectrum

# We compute the elementary excitation spectrum of a 2D condensate trapped in an
# **anisotropic harmonic potential** with the matrix-free
# [`NumModelBdG`](@ref) solver. The linearised problem reads, for the complex
# amplitude $(u, v)$ of a small perturbation $ψ = ψ_0 + u + v^*$,

# ```math
# \begin{pmatrix} H & K \\ -K^* & -H \end{pmatrix}
# \binom{u}{v^*} = ω \binom{u}{v^*},
# \qquad
# H = -\tfrac{1}{2}\nabla^2 + V + 2β|ψ_0|^2 - μ - iΩ L_z,\quad
# K = β\,ψ_0^2
# ```

# where $ψ_0$ is the stationary condensate and $μ$ its chemical potential.
# The solver is **matrix-free**: `H u - K v` and `K^* u - H v` are applied on the
# fly with the package's own derivative machinery and the eigensystem is
# diagonalised by ARPACK (`which=:SM`, smallest $|ω|$). The two Goldstone zero
# modes (phase and particle-number conservation) are rejected by their overlap
# with $(ψ_0, ψ_0^*)$ and the positive-norm branch $ω>0$ is kept.

# We first load the package, in order to have all the constructors and functions available.

using SuperFluids

# We setup the topology used, if we want to use `MPI` for parallelization.
# In this case, run with `mpirun -np 1 julia --project examples/BDG/BDG_2D.jl`

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

# ## Setup

# The anisotropic oscillator $V = \tfrac12(γ_x x^2 + γ_y y^2)$ with
# `γx=1, γy=2` has single-particle frequencies ``ω_x = 1`` and
# ``ω_y = \sqrt{2}``. For the **non-interacting** case ($β=0$) the BdG
# frequencies are exactly ``ω = E - E_{00}``, i.e. the excitation energies
# $\{ω_x, ω_y, 2ω_x, ω_x+ω_y, \dots\}$ — a clean, non-degenerate reference
# spectrum.

grid = Grid((20, 20), ((-6.0, 6.0), (-6.0, 6.0)))
field = Field(grid, ComplexField())
X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
ωx, ωy = 1.0, sqrt(2.0)

# The exact discrete ground state of the anisotropic oscillator is
# ``ψ_0 ∝ exp(-ω_x x²/2 - ω_y y²/2)``; we normalise it:

field.ϕ .= exp.(-ωx * X.^2 / 2 .- ωy * Y.^2 / 2)
field.ϕ ./= sqrt(sum(abs2.(parent(field.ϕ))) * grid.Δx * grid.Δy)

# We build the potential and the BdG parameters:

pot = PotentialQuadratic(field; γx=ωx, γy=ωy^2)
param = BdGParameters(coeffΔ=-0.5, β=0.0, pot=pot, Ω=0.0)

# ## Solver

# We request `nev=3` positive-frequency modes:

model = NumModelBdG(field, param, 1, 1; nev=3)

# The eigensystem is computed in one (trivial) `timeStep!` call. `solve!` works
# through the standard driver too:

res = solve!(model; plot=false)
res[end]

# `model.ωs` holds the computed positive frequencies and `model.us`/`model.vs`
# the corresponding normalised amplitudes (``‖u‖²-‖v‖² = 1``):

model.ωs

# For $β=0$ the amplitudes satisfy $v ≡ 0$ and $u$ is a plain oscillator
# eigenfunction. We plot the three first excited states $|u_1|^2, |u_2|^2,
# |u_3|^2$:

# First excitation $|u_1|^2$ (frequency $ω_1 ≈ ω_x = 1$):

if use_plots
    heatmap(grid.x, grid.y, abs2.(model.us[1]); colormap=:viridis)
    current_axis().title[] = "|u₁|² — ω = " * string(round(model.ωs[1]; digits=4))
    current_figure()
end

# Second excitation $|u_2|^2$ (frequency $ω_2 ≈ ω_y = √2$):

if use_plots
    heatmap(grid.x, grid.y, abs2.(model.us[2]); colormap=:viridis)
    current_axis().title[] = "|u₂|² — ω = " * string(round(model.ωs[2]; digits=4))
    current_figure()
end

# Third excitation $|u_3|^2$ (frequency $ω_3 ≈ 2ω_x = 2$):

if use_plots
    heatmap(grid.x, grid.y, abs2.(model.us[3]); colormap=:viridis)
    current_axis().title[] = "|u₃|² — ω = " * string(round(model.ωs[3]; digits=4))
    current_figure()
end
