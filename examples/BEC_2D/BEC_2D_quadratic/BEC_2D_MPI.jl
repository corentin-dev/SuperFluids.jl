using SuperFluids

# simulation parameters
nx = 64
ny = 64
nz = 64

xrange = (-8, 8)
yrange = (-8, 8)
zrange = (-8, 8)

device = GPU()
#device = CPU()
#if device.rank==0
#    println(device)
#end

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange), device=device)
println(grid)


field = Field(grid, ComplexField(), ndims=1, mpi_topo=SuperFluids.MPITopo2D())
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
initField!(init)

# BackwardEuler (with precond)
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)

# n = nummodel
# @. n.M = 1. / ( 1 / n.Δt + n.param.pot.V + n.param.β * abs2(n.f.ϕ) );
# # b = M × ϕ × Δt⁻¹
# @. n.b = n.M * n.f.ϕ / n.Δt;
# #n.nkrylov = 2;
# #
# # solving
# n.nkrylov = 10
# nkrylov = SuperFluids.krylov!(n, n.f.ϕ)

# println(nummodel)
solve!(nummodel, plot=false)