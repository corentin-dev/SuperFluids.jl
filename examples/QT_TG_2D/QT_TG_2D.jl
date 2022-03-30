using SuperFluids

# simulation parameters
nx = 128
ny = 128

xrange = (0, 2*π)
yrange = (0, 2*π)

# creating a grid
grid = Grid((nx,ny), (xrange,yrange), array_type=Array)
println_parallel(grid)

# allocating a field
field = Field(grid, ComplexField())
println_parallel(field)

# equation
param = GrossPitaevskiiParameters(
    coeffΔ = -0.05,
    β = 40,
    pot = PotentialTaylorGreen(field)
    )

# solver
Δt = 0.01
niter = 500
freqbckp = 10

# initialisation
init = InitExternalVelocity(field, param.coeffΔ, param.β)
initField!(init)

nummodel = NumModelExternalVelocity(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
solve!(nummodel, plot=false)

field_insta = Field(grid, ComplexField())
field_insta.ϕ .= field.ϕ

param_insta = GrossPitaevskiiParameters(
    coeffΔ = param.coeffΔ,
    β = param.coeffΔ,
    pot = PotentialZero(field_insta)
    )

Δt_insta = Δt / 2.
niter_insta = 5000
freqbckp_insta = 10
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)
solve!(nummodel_insta, istart=niter, plot=false)

nothing