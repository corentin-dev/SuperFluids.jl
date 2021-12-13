module SuperFluids

using ConfParser
using AbstractFFTs
using FFTW
using LinearAlgebra: mul!, ldiv!
using CUDA
using MPI
using PencilFFTs

"Abstract supertype for device."
abstract type Device end
"CPU device."
struct CPU <: Device end
"GPU device."
struct GPU <: Device end

include("MPI/MPI.jl")
include("Discretization/Discretization.jl")
include("Potentials/Potentials.jl")
include("Parameters/Parameters.jl")
include("Inits/Inits.jl")
include("IO/IO.jl")
include("NumModels/NumModels.jl")

export initField!, energy, finishWriter!
export CPU, MPI, GPU, Grid, Field

end # module
