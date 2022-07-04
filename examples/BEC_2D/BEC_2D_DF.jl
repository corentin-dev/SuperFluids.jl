using SuperFluids

# using DataFrames
# using CSV

mpi_topo = SuperFluids.MPITopo1D();

# simulation parameters
nx = 128
ny = 128

xrange = (-12, 12)
yrange = (-12, 12)

# creating a grid
grid = Grid((nx, ny), (xrange, yrange))
println_parallel(grid)
# allocating a field
field = Field(grid, ComplexField(); mpi_topo=mpi_topo)
println_parallel(field)

# potential
α = 0
γx = 1
γy = 1

# equation
param = GrossPitaevskiiParameters(; β=1000,
                                  Ω=0.9,
                                  pot=PotentialQuadratic(field; γx=γx, γy=γy))

# solver
Δt = 0.01
niter = 100
freqbckp = 10

# initialisation
init = InitThomasFermi(field, param.β; γx=γx, γy=γy)
# init = InitGauss(field, Ω = param.Ω)
initField!(init)

# solver
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp; nkrylov=500,
                                 tolkrylov=1e-6, plantype=FiniteDifferencePlan())
println_parallel(nummodel)

# solving
res = solve!(nummodel; plot=false)
initField!(init)

#CSV.write("energy-noprecond-$(Δt).csv",DataFrame(res),delim=" ",header=false)
nothing
