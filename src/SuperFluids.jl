module SuperFluids

"Abstract supertype for device."
abstract type Device end

"CPU device."
struct CPU <: Device end

"GPU device."
struct GPU <: Device end

"Abstract supertype for numerical models."
abstract type AbstractNumModel{F,P} end

"Abstract supertype for FFT plans."
abstract type AbstractPlan{F} end

"Abstract supertype for solvers."
abstract type AbstractSolver{G,F,I,N,P} end

include("GrossPitaevskii/GrossPitaevskii.jl")
export SolverGrossPitaevskii, solve!, initField!, energy

end # module
