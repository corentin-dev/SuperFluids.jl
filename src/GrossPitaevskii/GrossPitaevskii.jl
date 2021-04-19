include("Inits.jl")
include("NumModels.jl")
include("Potentials.jl")

mutable struct GrossPitaevskiiSolver{G,F,P,I,N} <: AbstractSolver{G,F,P,I,N}
   conf :: ConfParse
   grid :: G
   field :: F
   potential :: P
   init :: I
   nummodel :: N
end

function GrossPitaevskiiSolver(FT=Float64::Type,fileName="GPS_input.init"::String)
   conf = ConfParse(fileName)
   parse_conf!(conf)
   grid = Grid(conf)
   field = Field(grid,ComplexField())
   ninit = retrieve(conf, "model", "initcondition", Int64)
   init = Init(field,ninit,conf)
   potential = Potential(field, conf)
   nummodel = NumModel(field, potential, conf)
   return GrossPitaevskiiSolver{
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

function solve!(s::GrossPitaevskiiSolver)
   solve!(s.nummodel)
end

Base.show(io::IO, s::GrossPitaevskiiSolver) = print(s.grid,'\n',s.field,'\n',s.init,'\n',s.potential,'\n',s.nummodel)
