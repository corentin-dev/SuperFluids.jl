using SuperFluids
using LinearAlgebra

# simulation parameters
nx = 128
ny = 128
nz = 128

xrange = (-pi, pi)
yrange = (-pi, pi)
zrange = (-pi, pi)

# equation
param = NavierStokesParameters(ν = 0.001)

# solver
Δt = 0.001
niter = 1000
freqbckp = 10

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField(), ndims=3)
println(field)
# solver
nummodel = NumModelForwardEuler(field, param, Δt, niter, freqbckp)
println(nummodel)
# initialisation
taylor_green!(field.ϕ, grid.x, grid.y, grid.z)
mul!(nummodel.ϕ_hat, nummodel.plan.plan, field.ϕ)

# solving
solve!(nummodel, plot=false)
