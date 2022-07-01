"Abstract supertype for field initialization classes."
abstract type AbstractInit{F} end

export initField!
export InitNone, InitThomasFermi, InitGauss, InitExternalVelocity

struct InitNone{F} <: AbstractInit{F} end

include("thomas-fermi.jl")
include("gauss.jl")
include("external-velocity.jl")

Base.show(io::IO, init::InitNone) = print(io, "No Init")
