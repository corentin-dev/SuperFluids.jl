module SuperFluids

using ConfParser
using AbstractFFTs
using FFTW
using LinearAlgebra: mul!, ldiv!
using CUDA

"Abstract supertype for device."
abstract type Device end
"CPU device."
struct CPU <: Device end
"MPI device."
struct MPI <: Device end
"GPU device."
struct GPU <: Device end

include("Config/Config.jl")
include("Discretization/Discretization.jl")
include("IO/IO.jl")

"Abstract supertype for numerical models."
abstract type AbstractNumModel{F,P} end

"Abstract supertype for solvers."
abstract type AbstractSolver{C,G,F,I,N,P} end

function initField!(s::AbstractSolver)
   initField!(s.init)
end

function solve!(s::AbstractSolver, plot=false)
   solve!(s.nummodel, plot)
end

include("GrossPitaevskii/GrossPitaevskii.jl")
export GrossPitaevskiiSolver, solve!, initField!, energy, finishWriter!
export CPU, MPI, GPU, ComplexField, RealField, Grid, Field

end # module
