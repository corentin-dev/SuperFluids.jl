using SuperFluids

mpi_topo = SuperFluids.MPITopo2D();

# simulation parameters
nx = 128
ny = 128
nz = 128

xrange = (0, 2*π)
yrange = (0, 2*π)
zrange = (0, 2*π)

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange), array_type=Array)
println_parallel(grid)
# allocating a field
field = Field(grid, ComplexField(), mpi_topo = mpi_topo)
println_parallel(field)

# equation
param = GrossPitaevskiiParameters(
    coeffΔ = -0.05,
    β = 40,
    pot = PotentialTaylorGreen(field)
    )

# solver
Δt = 0.001
niter = 800
freqbckp = 100

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

Δt_insta = Δt
niter_insta = 1000
freqbckp_insta = 100
nummodel_insta = NumModelADI2(field_insta, param_insta, Δt_insta, niter_insta, freqbckp_insta)
solve!(nummodel_insta, istart=niter, plot=false)

nothing