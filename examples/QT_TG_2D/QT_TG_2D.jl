using SuperFluids

# simulation parameters
nx = 256
ny = 256

xrange = (0, 2*π)
yrange = (0, 2*π)

# creating a grid
grid = Grid((nx,ny), (xrange,yrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField())
println(field)

# equation
param = GrossPitaevskiiParameters(
    coeffΔ = -0.05,
    β = 40,
    pot = PotentialTaylorGreen(field)
    )

# solver
Δt = 0.0125
niter = 1000
freqbckp = 10

# initialisation
init = InitExternalVelocity(field, param.coeffΔ, param.β)
# init = InitGauss(field, Ω = Ω)
field.ϕ .= init.(grid.x,grid.y)

nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

field_insta = Field(grid, ComplexField())
field_insta.ϕ .= field.ϕ

param_insta = GrossPitaevskiiParameters(
    coeffΔ = -0.05,
    β = 40,
    pot = PotentialZero(field_insta)
    )

Δt_insta = Δt / 100.
niter_insta = 5000
freqbckp_insta = 10
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)
solve!(nummodel_insta, istart=niter, plot=false)

nothing