using SuperFluids

# simulation parameters
nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

# potential
α = 0
γx = 1
γy = 1

# equation
param = GrossPitaevskiiParameters(β = 1000,
                                  Ω = 0.9,
                                  V = PotentialQuadratic2D(γx = γx, γy = γy)
                                 )

# solver
Δt = 0.01
niter = 25
freqbckp = 10

# creating a grid
grid = Grid((nx,ny), (xrange,yrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField())
println(field)
# initialisation
init = InitThomasFermi(field, param.β, γx = γx, γy = γy)
# init = InitGauss(field, Ω = Ω)
field.ϕ .= init.(grid.x,grid.y)
normalize!(field)

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