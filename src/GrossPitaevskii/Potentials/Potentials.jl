"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential end
abstract type AbstractPotential2D <: AbstractPotential end
abstract type AbstractPotential3D <: AbstractPotential end

export PotentialZero, PotentialQuadratic, PotentialQuarticQuadratic

include("quadratic.jl")
include("quartic.jl")
include("external-velocity.jl")

struct PotentialZero <: AbstractPotential
   V
end

Base.show(io::IO, p::PotentialZero) = print(io, "Zero Potential")
