"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential end
abstract type AbstractPotential2D <: AbstractPotential end
abstract type AbstractPotential3D <: AbstractPotential end

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

# function Potential(f::AbstractField{A,G}, conf::AbstractConfig) where {A<:CuArray,G}
#    npot = retrieve(conf, "model", "potential", Int64)
#    V = CUDA.zeros(FT, size(f.ϕ))
#    if npot == 0
#       return PotentialZero(V)
#    elseif npot == 1
#       return PotentialQuadratic(V, f, conf)
#    elseif npot == 2
#       return PotentialQuarticQuadratic(V, f, conf)
#    elseif npot == 8
#       return PotentialTaylorGreen(V, f, conf)
#    end
#    return nothing
# end

struct PotentialQuadratic2D <: AbstractPotential2D
   V
   α :: Real
   γx :: Real
   γy :: Real
end

function PotentialQuadratic(V, f::AbstractField2D, conf::AbstractConfig)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)
   return PotentialQuadratic2D(V, α, γx, γy)
end

Base.show(io::IO, p::PotentialQuadratic2D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

struct PotentialQuadratic3D <: AbstractPotential3D
   V
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

function PotentialQuadratic(V, f::AbstractField3D, conf::AbstractConfig)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   γz = retrieve(conf, "potential", "gamma_z", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)
   return PotentialQuadratic3D(V, α, γx, γy, γz)
end

Base.show(io::IO, p::PotentialQuadratic3D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")

struct PotentialQuarticQuadratic2D <: AbstractPotential2D
   V
   α :: Real
   γx :: Real
   γy :: Real
   κ4 :: Real
end

function PotentialQuarticQuadratic(V, f::AbstractField2D, conf::AbstractConfig)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   κ4 = retrieve(conf, "potential", "kappa4", Float64)
   # @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic2D(V, α, γx, γy, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic2D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")

struct PotentialQuarticQuadratic3D <: AbstractPotential3D
   V
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
   κ4 :: Real
end

function PotentialQuarticQuadratic(V, f::AbstractField3D, conf::AbstractConfig)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   γz = retrieve(conf, "potential", "gamma_z", Float64)
   κ4 = retrieve(conf, "potential", "kappa4", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic3D(V, α, γx, γy, γz, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic3D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz) κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")

struct PotentialTaylorGreen2D <: AbstractPotential2D
   V
   uadvx
   uadvy
end

function PotentialTaylorGreen(V, f::AbstractField2D, conf::AbstractConfig)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y)
   uadvy = -cos.(f.g.x).*sin.(f.g.y)
   # potential
   coeffΔ = retrieve(conf, "model", "delta", Float64)
   β = retrieve(conf, "model", "beta", Float64)
   @. V = ( uadvx.^2 .+ uadvy.^2 ) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen2D(V, uadvx, uadvy)
end

Base.show(io::IO, p::PotentialTaylorGreen2D) = print(io, "TaylorGreen Potential")

struct PotentialTaylorGreen3D <: AbstractPotential3D
   V
   uadvx
   uadvy
   uadvz
end

function PotentialTaylorGreen(V, f::AbstractField3D, conf::AbstractConfig)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y).*cos.(f.g.z)
   uadvy = -cos.(f.g.x).*sin.(f.g.y).*cos.(f.g.z)
   uadvz =  zeros(Real, size(uadvx))
   # potential
   coeffΔ = retrieve(conf, "model", "delta", Float64)
   β = retrieve(conf, "model", "beta", Float64)
   @. V = ( uadvx.^2 .+ uadvy.^2 .+ uadvz.^2) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen3D(V, uadvx, uadvy, uadvz)
end

Base.show(io::IO, p::PotentialTaylorGreen3D) = print(io, "TaylorGreen Potential")
