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

include("Discretization/Discretization.jl")
include("Inits/Inits.jl")
include("IO/IO.jl")
include("NumModels/NumModels.jl")
include("Potentials/Potentials.jl")

export solve!, initField!, energy, finishWriter!
export CPU, MPI, GPU, Grid, Field

end # module
