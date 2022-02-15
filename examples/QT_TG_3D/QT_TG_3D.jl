using SuperFluids

# simulation parameters
nx = 64
ny = 64
nz = 64

xrange = (0, 2*π)
yrange = (0, 2*π)
zrange = (0, 2*π)

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange), array_type=Array)
println_parallel(grid)
# allocating a field
field = Field(grid, ComplexField())
println_parallel(field)

# equation
param = GrossPitaevskiiParameters(
    coeffΔ = -0.1,
    β = 20,
    pot = PotentialTaylorGreen(field)
    )

# solver
Δt = 0.02
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
    β = param.β,
    pot = PotentialZero(field_insta)
    )

Δt_insta = Δt / 2.
niter_insta = 5000
freqbckp_insta = 100
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)
solve!(nummodel_insta, istart=niter, plot=false)

nothing