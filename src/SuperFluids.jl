module SuperFluids

using AbstractFFTs
using FFTW
using LinearAlgebra: mul!, ldiv!, transpose!
using MPI
using HDF5
using PencilArrays

include("MPI/MPI.jl")
include("Discretization/Discretization.jl")
include("Potentials/Potentials.jl")
include("Parameters/Parameters.jl")
include("Inits/Inits.jl")
include("IO/IO.jl")
include("NumModels/NumModels.jl")

export initField!, energy, finishWriter!
export Grid, Field
export print_parallel, println_parallel

end # module
