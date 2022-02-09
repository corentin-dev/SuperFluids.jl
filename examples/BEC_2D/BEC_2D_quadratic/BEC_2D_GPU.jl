using SuperFluids
using CUDA

mpi_topo = SuperFluids.MPITopo1D();

# simulation parameters
nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

# creating a grid
grid = Grid((nx,ny), (xrange,yrange), array_type=CuArray)
println(grid)
# allocating a field
field = Field(grid, ComplexField(), mpi_topo=mpi_topo)
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
# init = InitGauss(field, Ω = Ω)
initField!(init)

# solver
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)
println(nummodel)

# solving
solve!(nummodel, plot=false)
nothing
