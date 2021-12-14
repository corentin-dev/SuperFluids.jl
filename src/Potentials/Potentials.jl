export PotentialZero, PotentialQuarticQuadratic, PotentialQuadratic, PotentialTaylorGreen

"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential{F} end

function similar_real(a::AbstractArray{A}) where A
   T = get_real_type_array(a)
   myArray = get_array_type(a)
   if A <: PencilArray
      return PencilArray(a.pencil, myArray{T}(undef, size(a.data)))
   else
      return myArray{T}(undef,size(a))
   end
end

struct PotentialZero{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
end

function PotentialZero(f::F) where {F<:AbstractField}
   V = similar_real(f.ϕ)
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