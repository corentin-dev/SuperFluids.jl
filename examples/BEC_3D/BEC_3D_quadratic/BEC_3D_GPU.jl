using SuperFluids
using CUDA

# simulation parameters
nx = 128
ny = 128
nz = 128

xrange = (-12, 12)
yrange = (-12, 12)
zrange = (-12, 12)

mpi_topo = SuperFluids.MPITopo2D();

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange), array_type=CuArray)
println_parallel(grid)

field = Field(grid, ComplexField(), ndims=1, mpi_topo = mpi_topo)
println_parallel(field)

# potential
α = 0;
γx = 1;
γy = 1;
γz = 1;

# equation
param = GrossPitaevskiiParameters(
    β = 1000,
    Ω = 0.9,
    pot = PotentialQuadratic(field, γx = γx, γy = γy, γz = γz)
    )
println_parallel(param)

# solver
Δt = 0.01;
niter = 100;
freqbckp = 10;

# initialisation
init = InitThomasFermi(field, param.β, γx = γx, γy = γy, γz = γz)
# init = InitGauss(field, Ω = param.Ω)
println_parallel(init)
initField!(init)

# BackwardEuler (with precond)
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)
println_parallel(nummodel)

# Solve
solve!(nummodel, plot=false)
nothing
