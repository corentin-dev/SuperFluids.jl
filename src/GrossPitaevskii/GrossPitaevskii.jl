include("Inits/Inits.jl")
include("NumModels/NumModels.jl")
include("Potentials/Potentials.jl")

mutable struct GrossPitaevskiiSolver{C,G,F,P,I,N} <: AbstractSolver{C,G,F,P,I,N}
   conf :: C
   grid :: G
   field :: F
   potential :: P
   init :: I
   nummodel :: N
end

function GrossPitaevskiiSolver(;fileName="GPS_input.init"::String)
   device = CPU()
   # device = GPU()
   # CUDA.allowscalar(false)
   conf = Config(fileName)
   # grid = Grid(conf,D=device)
   grid = Grid(conf,FT=Float64,D=device)
   field = Field(grid,ComplexField())
   ninit = retrieve(conf, "model", "initcondition", Int64)
   init = Init(field,ninit,conf)
   potential = Potential(field, conf)
   nummodel = NumModel(field, potential, conf)
   return GrossPitaevskiiSolver{
                  typeof(conf),
                  typeof(grid),
                  typeof(field),
                  typeof(potential),
                  typeof(init),
                  typeof(nummodel)
                }(
                  conf,
                  grid,
                  field,
                  potential,
                  init,
                  nummodel
                 )
end

function initField!(s::GrossPitaevskiiSolver)
   initField!(s.init)
end

function solve!(s::GrossPitaevskiiSolver; istart=1, plot=false)
   solve!(s.nummodel, istart=istart, plot=plot)
end

Base.show(io::IO, s::GrossPitaevskiiSolver) = print(s.grid,'\n',s.field,'\n',s.init,'\n',s.potential,'\n',s.nummodel)
