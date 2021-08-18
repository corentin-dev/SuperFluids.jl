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

function Potential(f::AbstractField, conf::AbstractConfig)
   npot = retrieve(conf, "model", "potential", Int64)
   CUDA.allowscalar(true)
   FT = eltype(real(f.ϕ[1]))
   CUDA.allowscalar(false)
   V = real(similar(f.ϕ))
   # V =zeros(FT, size(f.ϕ))
   if npot == 0
      V .= 0.
      return PotentialZero(V)
   elseif npot == 1
      return PotentialQuadratic(V, f, conf)
   elseif npot == 2
      return PotentialQuarticQuadratic(V, f, conf)
   elseif npot == 8
      return PotentialTaylorGreen(V, f, conf)
   end
   return nothing
end
