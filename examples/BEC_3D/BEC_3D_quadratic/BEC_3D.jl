using SuperFluids

# simulation parameters
nx = 64
ny = 64
nz = 64

xrange = (-8, 8)
yrange = (-8, 8)
zrange = (-8, 8)

# potential
α = 0
γx = 1
γy = 1
γz = 1

# equation
param = GrossPitaevskiiParameters(β = 1000,
                                  Ω = 0.8,
                                  V = PotentialQuadratic3D(γx = γx, γy = γy, γz = γz)
                                 )

# solver
Δt = 0.01
niter = 1000
freqbckp = 10

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange))
println(grid)
# allocating a field
field = Field(grid, ComplexField())
println(field)
# initialisation
init = InitThomasFermi(field, param.β, γx = γx, γy = γy, γz = γz)
# init = InitGauss(field, Ω = Ω)
field.ϕ .= init.(grid.x,grid.y,grid.z)
normalize!(field)

# BackwardEuler (with precond)
nummodel = NumModelBackwardEuler(field, param, Δt, niter, freqbckp)
println(nummodel)
solve!(nummodel, plot=false)

nothing
