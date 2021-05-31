"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential{FT} end
abstract type AbstractPotential2D{FT} <: AbstractPotential{FT} end
abstract type AbstractPotential3D{FT} <: AbstractPotential{FT} end

struct PotentialZero{FT} <: AbstractPotential{FT}
   V :: Array{FT}
end

Base.show(io::IO, p::PotentialZero) = print(io, "Zero Potential")

@inline Base.getindex(p::PotentialZero{FT}, i::Any) where FT = FT(0)
@inline Base.setindex!(p::PotentialZero{FT}, v, i::Any) where FT = nothing

@inline Base.getindex(p::P, i::Any) where P<:AbstractPotential = p.V[i]
@inline Base.setindex!(p::P, v, i::Any) where P<:AbstractPotential = (p.V[i] = v)

function Potential(f::AbstractField, conf::AbstractConfig)
   npot = retrieve(conf, "model", "potential", Int64)
   FT = eltype(real(f.ϕ[1]))
   if npot == 0
      V = zeros(FT, size(f.ϕ))
      return PotentialZero{FT}(V)
   elseif npot == 1
      return PotentialQuadratic(FT, f, conf)
   elseif npot == 2
      return PotentialQuarticQuadratic(FT, f, conf)
   elseif npot == 8
      return PotentialTaylorGreen(FT, f, conf)
   end
   return nothing
end

struct PotentialQuadratic2D{FT} <: AbstractPotential2D{FT}
   V :: Array{FT,2}
   α :: FT
   γx :: FT
   γy :: FT
end

function PotentialQuadratic(FT::DataType, f::AbstractField2D, conf::AbstractConfig)
   V = zeros(FT, f.g.nx, f.g.ny)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)
   return PotentialQuadratic2D{FT}(V, α, γx, γy)
end

Base.show(io::IO, p::PotentialQuadratic2D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

struct PotentialQuadratic3D{FT} <: AbstractPotential3D{FT}
   V :: Array{FT,3}
   α :: FT
   γx :: FT
   γy :: FT
   γz :: FT
end

function PotentialQuadratic(FT::DataType, f::AbstractField3D, conf::AbstractConfig)
   V = zeros(FT, f.g.nx, f.g.ny, f.g.nz)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   γz = retrieve(conf, "potential", "gamma_z", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)
   return PotentialQuadratic3D{FT}(V, α, γx, γy, γz)
end

Base.show(io::IO, p::PotentialQuadratic3D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")

struct PotentialQuarticQuadratic2D{FT} <: AbstractPotential2D{FT}
   V :: Array{FT,2}
   α :: FT
   γx :: FT
   γy :: FT
   κ4 :: FT
end

function PotentialQuarticQuadratic(FT::DataType, f::AbstractField2D, conf::AbstractConfig)
   V = zeros(FT, f.g.nx, f.g.ny)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   κ4 = retrieve(conf, "potential", "kappa4", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic2D{FT}(V, α, γx, γy, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic2D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")

struct PotentialQuarticQuadratic3D{FT} <: AbstractPotential3D{FT}
   V :: Array{FT,3}
   α :: FT
   γx :: FT
   γy :: FT
   γz :: FT
   κ4 :: FT
end

function PotentialQuarticQuadratic(FT::DataType, f::AbstractField3D, conf::AbstractConfig)
   V = zeros(FT, f.g.nx, f.g.ny, f.g.nz)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   γz = retrieve(conf, "potential", "gamma_z", Float64)
   κ4 = retrieve(conf, "potential", "kappa4", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic3D{FT}(V, α, γx, γy, γz, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic3D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz) κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")

struct PotentialTaylorGreen2D{FT} <: AbstractPotential2D{FT}
   V :: Array{FT,2}
   uadvx :: Array{FT}
   uadvy :: Array{FT}
end

function PotentialTaylorGreen(FT::DataType, f::AbstractField2D, conf::AbstractConfig)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y)
   uadvy = -cos.(f.g.x).*sin.(f.g.y)
   # potential
   coeffΔ = retrieve(conf, "model", "delta", Float64)
   β = retrieve(conf, "model", "beta", Float64)
   V = zeros(FT, size(uadvx))
   V = ( uadvx.^2 .+ uadvy.^2 ) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen2D{FT}(V, uadvx, uadvy)
end

Base.show(io::IO, p::PotentialTaylorGreen2D) = print(io, "TaylorGreen Potential")

struct PotentialTaylorGreen3D{FT} <: AbstractPotential3D{FT}
   V :: Array{FT,3}
   uadvx :: Array{FT}
   uadvy :: Array{FT}
   uadvz :: Array{FT}
end

function PotentialTaylorGreen(FT::DataType, f::AbstractField3D, conf::AbstractConfig)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y).*cos.(f.g.z)
   uadvy = -cos.(f.g.x).*sin.(f.g.y).*cos.(f.g.z)
   uadvz =  zeros(FT, size(uadvx))
   # potential
   coeffΔ = retrieve(conf, "model", "delta", Float64)
   β = retrieve(conf, "model", "beta", Float64)
   V = zeros(FT, size(uadvx))
   V = ( uadvx.^2 .+ uadvy.^2 .+ uadvz.^2) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen3D{FT}(V, uadvx, uadvy, uadvz)
end

Base.show(io::IO, p::PotentialTaylorGreen3D) = print(io, "TaylorGreen Potential")
