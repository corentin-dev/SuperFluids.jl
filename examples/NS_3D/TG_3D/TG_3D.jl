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
niter = 200
freqbckp = 10

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange))
println_parallel(grid)
# allocating a field
field = Field(grid, ComplexField(), ndims=3)
println_parallel(field)
# solver
nummodel = NumModelForwardEuler(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)
# initialisation
taylor_green!(field, grid.x, grid.y, grid.z)
# solving
solve!(nummodel, plot=false)