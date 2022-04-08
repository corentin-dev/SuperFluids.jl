# Vortex pair

using SuperFluids

mpi_topo = SuperFluids.MPITopo1D()

#makie_style = "WGLMakie"
makie_style = "GLMakie"
#makie_style = "CairoMakie"

if mpi_topo.size == 1 # hide
    use_plots = true # hide
else # hide
    use_plots = false # hide
end # hide
use_plots = false # hide

if use_plots # hide
if makie_style == "WGLMakie"
    using WGLMakie
    WGLMakie.activate!()
    using JSServe # hide
    Page(exportable=true, offline=true) # hide
elseif makie_style == "GLMakie"
    using GLMakie
    GLMakie.activate!()
elseif makie_style == "CairoMakie"
    using CairoMakie
    CairoMakie.activate!()
end
end # hide

nx = 128
ny = 128
xrange = (-π, π)
yrange = (-π, π)
xrange = (0., 2π)
yrange = (0., 2π)

grid = Grid((nx,ny), (xrange,yrange))
field = Field(grid, ComplexField(), mpi_topo = mpi_topo)

α = 0.16572815184059705
#ξ = 1.5 * grid.Δx
ξ = 3 * grid.Δx
d = π / ξ
β = α / ξ^2
coeffΔ = - α / β
v₊ = ( grid.Lx/4 + grid.xmin, grid.Ly/2 + d*ξ/2 + grid.ymin, +1)
v₋ = ( grid.Lx/4 + grid.xmin, grid.Ly/2 - d*ξ/2 + grid.ymin, -1)
uadv_function_VR_2D = ((x,y)->-α * ( (1+cos(d*ξ)) /sin(d*ξ) + d*ξ /π ) ,(x,y)->0.);

pot = PotentialExternalVelocity(field, uadv_function=uadv_function_VR_2D, α = 4α)
param = GrossPitaevskiiParameters(
    coeffΔ = coeffΔ,
    β = β,
    pot = pot
    )

function ρ_vortex(r)
    a₁, a₂, b₁, b₂ = 11. / 32., 11. / 384., 1. / 3., 11. / 384.
    return √( ( a₁ * r^2 + a₂ * r^4 ) / ( 1 + b₁ * r^2 + b₂ * r^4  ) )
end

function init_VR_tanh(x,y,ξ,v₊,v₋,grid)
    θ, ρ = 0., 1.

    ρ = ρ_vortex( √( (x-v₋[1])^2 + (y-v₋[2])^2 )/ξ ) * ρ_vortex( √( (x-v₊[1])^2 + (y-v₊[2])^2 )/ξ )

    x₋, y₋ = 2π/grid.Lx * (x - v₋[1]), 2π/grid.Ly * (y - v₋[2])
    x₊, y₊ = 2π/grid.Lx * (x - v₊[1]), 2π/grid.Ly * (y - v₊[2])

    θ = (v₋[1]-v₊[1])*y / 2π
    for k ∈ -10:10
        θ += (
            atan( tanh( 0.5 * (y₋+2π*k) ) * tan(0.5(x₋-π)) )
          - atan( tanh( 0.5 * (y₊+2π*k) ) * tan(0.5(x₊-π)) )
          + π * ( ( (x₊ > 0) ? 1. : 0.) - ( (x₋ > 0) ? 1. : 0.) )
        )
    end
    return ρ * exp(im*θ)
end

my_init = (x,y) -> init_VR_tanh(x,y,ξ,v₊,v₋,grid)
@. field.ϕ = my_init(field.x,field.y)

if use_plots # hide
contourf(grid.x, grid.y, atan.(imag.(field.ϕ),real.(field.ϕ)) )
contourf(grid.x, grid.y, abs2.(field.ϕ) )
end # hide

run(`bash clean.sh`)

Δt = 0.001
niter = 2000
freqbckp = 100
nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)
res = solve!(nummodel, plot=false);

if use_plots # hide
contourf(grid.x, grid.y, atan.(imag.(field.ϕ),real.(field.ϕ)) )
contourf(grid.x, grid.y, abs2.(field.ϕ) )
end # hide

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

if use_plots # hide
contourf(grid.x, grid.y, atan.(imag.(field_insta.ϕ),real.(field_insta.ϕ)) )
contourf(grid.x, grid.y, abs2.(field_insta.ϕ) )
end # hide