using SuperFluids

# simulation parameters
nx = 32
ny = 32
nz = 32

xrange = (-pi, pi)
yrange = (-pi, pi)
zrange = (-pi, pi)

# potential
α = 0
γx = 1
γy = 1

# equation
param = NavierStokesParameters(ν = 0.01)

# solver
Δt = 0.01
niter = 1000
freqbckp = 10

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField(), ndims=3)
println(field)
# initialisation
taylor_green!(field.ϕ, grid.x, grid.y, grid.z)
# solver
nummodel = NumModelForwardEuler(field, param, Δt, niter, freqbckp)
println(nummodel)

# solving
solve!(nummodel, plot=false)
