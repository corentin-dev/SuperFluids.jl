using ConfParser

include("Grids.jl")
include("Fields.jl")
include("Inits.jl")
include("NumModels.jl")
include("Potentials.jl")

abstract type AbstractSolver{G,F,I,N,P} end

struct Solver{G,F,P,I,N} <: AbstractSolver{G,F,P,I,N}
   conf :: ConfParse
   grid :: G
   field :: F
   potential :: P
   init :: I
   nummodel :: N
end

function Solver(FT=Float64,conf="GPS_input.init")
   conf = ConfParse("GPS_input.init")
   parse_conf!(conf)
   grid = Grid(conf)
   field = Field(grid)
   ninit = parse(Int64,retrieve(conf, "model", "initcondition"))
   init = Init(field,ninit,conf)
   potential = Potential(field, conf)
   nummodel = NumModel(field, potential, conf)
   return Solver{
                 typeof(grid),typeof(field),typeof(potential),typeof(init),typeof(nummodel)
                }(
                 conf,
                  grid,field,potential,init,nummodel
                  )
end

function solve!(s::Solver)
   solve!(s.nummodel)
end

Base.show(io::IO, s::Solver) = print(s.grid,'\n',s.field,'\n',s.init,'\n',s.potential,'\n',s.nummodel)
