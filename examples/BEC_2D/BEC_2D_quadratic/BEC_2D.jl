using SuperFluids
using DataFrames
using CSV

# simulation parameters
nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

# creating a grid
grid = Grid((nx,ny), (xrange,yrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField())
println(field)

# potential
α = 0
γx = 1
γy = 1

# equation
param = GrossPitaevskiiParameters(
    β = 1000,
    Ω = 0.9,
    pot = PotentialQuadratic(field, γx = γx, γy = γy)
    )

# solver
Δt = 0.01
niter = 1000
freqbckp = 10

# initialisation
init = InitThomasFermi(field, param.β, γx = γx, γy = γy)
# init = InitGauss(field, Ω = param.Ω)
field.ϕ .= init.(grid.x,grid.y)
normalize!(field)
# solver
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp, nkrylov = 500, tolkrylov = 1e-6)
println(nummodel)

# solving
res = solve!(nummodel, plot=false)
CSV.write("energy-noprecond-$(Δt).csv",DataFrame(res),delim=" ",header=false)
nothing
