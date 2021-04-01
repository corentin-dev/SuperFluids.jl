include("Inits.jl")
include("NumModels.jl")
include("Potentials.jl")

mutable struct SolverGrossPitaevskii{G,F,P,I,N} <: AbstractSolver{G,F,P,I,N}
   conf :: ConfParse
   grid :: G
   field :: F
   potential :: P
   init :: I
   nummodel :: N
end

function SolverGrossPitaevskii(FT=Float64,conf="GPS_input.init")
   conf = ConfParse("GPS_input.init")
   parse_conf!(conf)
   grid = Grid(conf)
   field = Field(grid,ComplexField())
   ninit = parse(Int64,retrieve(conf, "model", "initcondition"))
   init = Init(field,ninit,conf)
   potential = Potential(field, conf)
   nummodel = NumModel(field, potential, conf)
   return SolverGrossPitaevskii{
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

function initField!(s::SolverGrossPitaevskii)
   initField!(s.init)
end

function solve!(s::SolverGrossPitaevskii)
   solve!(s.nummodel)
end

Base.show(io::IO, s::SolverGrossPitaevskii) = print(s.grid,'\n',s.field,'\n',s.init,'\n',s.potential,'\n',s.nummodel)
