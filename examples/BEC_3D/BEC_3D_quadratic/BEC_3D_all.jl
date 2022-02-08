using SuperFluids

# simulation parameters
nx = 64
ny = 64
nz = 64

xrange = (-8, 8)
yrange = (-8, 8)
zrange = (-8, 8)

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField())
println(field)

# potential
α = 0
γx = 1
γy = 1
γz = 1

# equation
param = GrossPitaevskiiParameters(
    β = 1000,
    Ω = 0.8,
    pot = PotentialQuadratic(field, γx = γx, γy = γy, γz = γz)
    )

# solver
Δt = 0.01
niter = 25
freqbckp = 10

# initialisation
init = InitThomasFermi(field, param.β, γx = γx, γy = γy, γz = γz)
# init = InitGauss(field, Ω = Ω)
initField!(init)

# BackwardEuler (with precond)
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# BackwardEulerNoPrecond
nummodel = NumModelBackwardEulerNoPrecond(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# CrankNicolson
nummodel = NumModelCrankNicolson(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# CrankNicolsonQuasiNewton
nummodel = NumModelCrankNicolsonQuasiNewton(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# ADI1
nummodel = NumModelADI1(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# ADI2
nummodel = NumModelADI2(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# CrankNicolsonT
nummodel = NumModelCrankNicolsonT(field, param, Δt*0.1, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

# CrankNicolsonQuasiNewtonT
nummodel = NumModelCrankNicolsonQuasiNewtonT(field, param, Δt*0.1, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

nothing
