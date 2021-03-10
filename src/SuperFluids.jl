module SuperFluids

abstract type AbstractSolver{G,F,I,N,P} end

include("GrossPitaevskii/GrossPitaevskii.jl")
export SolverGrossPitaevskii, solve!

end # module
