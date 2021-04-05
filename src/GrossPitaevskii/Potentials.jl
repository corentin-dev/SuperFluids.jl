"""
    AbstractPotential

Abstract supertype for field initialization classes.
"""
abstract type AbstractPotential{FT} end
abstract type AbstractPotential2D{FT} <: AbstractPotential{FT} end
abstract type AbstractPotential3D{FT} <: AbstractPotential{FT} end

struct PotentialZero{FT} <: AbstractPotential{FT} end

@inline Base.getindex(p::PotentialZero{FT}, i::Any) where FT = FT(0)
@inline Base.setindex!(p::PotentialZero{FT}, v, i::Any) where FT = nothing

@inline Base.getindex(p::P, i::Any) where P<:AbstractPotential = p.V[i]
@inline Base.setindex!(p::P, v, i::Any) where P<:AbstractPotential = (p.V[i] = v)

function Potential(f::AbstractField, conf::ConfParse)
   npot = parse(Int64,retrieve(conf, "model", "potential"))
   FT = eltype(real(f.ϕ[1]))
   if npot == 0
      return PotentialZero{FT}()
   end
   if npot == 1
      return PotentialQuadratic(FT, f, conf)
   end
   if npot == 2
      return PotentialQuarticQuadratic(FT, f, conf)
   end
   return nothing
end

struct PotentialQuadratic2D{FT} <: AbstractPotential2D{FT}
   V :: Array{FT,2}
   α :: FT
   γx :: FT
   γy :: FT
end

struct PotentialQuadratic3D{FT} <: AbstractPotential3D{FT}
   V :: Array{FT,3}
   α :: FT
   γx :: FT
   γy :: FT
   γz :: FT
end

function PotentialQuadratic(FT::DataType, f::AbstractField2D, conf::ConfParse)
   V = zeros(FT, f.g.nx, f.g.ny)
   α  = parse(Float64,retrieve(conf, "potential", "alpha"))
   γx = parse(Float64,retrieve(conf, "potential", "gamma_x"))
   γy = parse(Float64,retrieve(conf, "potential", "gamma_y"))
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)
   return PotentialQuadratic2D{FT}(V, α, γx, γy)
end

Base.show(io::IO, p::PotentialQuadratic2D) =
     print(io, "Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

function PotentialQuadratic(FT::DataType, f::AbstractField3D, conf::ConfParse)
   V = zeros(FT, f.g.nx, f.g.ny, f.g.nz)
   α  = parse(Float64,retrieve(conf, "potential", "alpha"))
   γx = parse(Float64,retrieve(conf, "potential", "gamma_x"))
   γy = parse(Float64,retrieve(conf, "potential", "gamma_y"))
   γz = parse(Float64,retrieve(conf, "potential", "gamma_z"))
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

struct PotentialQuarticQuadratic3D{FT} <: AbstractPotential3D{FT}
   V :: Array{FT,3}
   α :: FT
   γx :: FT
   γy :: FT
   γz :: FT
   κ4 :: FT
end

function PotentialQuarticQuadratic(FT::DataType, f::AbstractField2D, conf::ConfParse)
   V = zeros(FT, f.g.nx, f.g.ny)
   α  = parse(Float64,retrieve(conf, "potential", "alpha"))
   γx = parse(Float64,retrieve(conf, "potential", "gamma_x"))
   γy = parse(Float64,retrieve(conf, "potential", "gamma_y"))
   κ4 = parse(Float64,retrieve(conf, "potential", "kappa4"))
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic2D{FT}(V, α, γx, γy, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic2D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")

function PotentialQuarticQuadratic(FT::DataType, f::AbstractField3D, conf::ConfParse)
   V = zeros(FT, f.g.nx, f.g.ny, f.g.nz)
   α  = parse(Float64,retrieve(conf, "potential", "alpha"))
   γx = parse(Float64,retrieve(conf, "potential", "gamma_x"))
   γy = parse(Float64,retrieve(conf, "potential", "gamma_y"))
   γz = parse(Float64,retrieve(conf, "potential", "gamma_z"))
   κ4 = parse(Float64,retrieve(conf, "potential", "kappa4"))
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic3D{FT}(V, α, γx, γy, γz, κ4)
end

Base.show(io::IO, p::PotentialQuarticQuadratic3D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz) κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")
