export PotentialZero, PotentialQuarticQuadratic, PotentialQuadratic, PotentialExternalVelocity

"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential{F} end

struct PotentialZero{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
end

function PotentialZero(f::F) where {F<:AbstractField{1,FT,FFT,A}} where {FT,FFT,A}
   V = PencilArray(f.ϕ.pencil, A{FT}(undef, size_local(f.ϕ)))
   V .= 0.
   return PotentialZero{F}(f, V)
end

function compute!(p::PotentialZero{F}) where {F<:AbstractField}
   @. p.V = 0.
end

Base.show(io::IO, p::PotentialZero) = print(io, "Zero Potential")

include("quadratic.jl")
include("quartic.jl")
include("external-velocity.jl")