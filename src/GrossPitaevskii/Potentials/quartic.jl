struct PotentialQuarticQuadratic2D <: AbstractPotential2D
   V
   α :: Real
   γx :: Real
   γy :: Real
   κ4 :: Real
end

struct PotentialQuarticQuadratic3D <: AbstractPotential3D
   V
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
   κ4 :: Real
end

function PotentialQuarticQuadratic(f::AbstractField2D, α::Real, γx::Real, γy::Real, κ4::Real)
   V = 0.5*(1-α)*(γx*f.g.x.^2 .+ γy*f.g.y.^2) .+ 0.5*κ4*(f.g.x.^2 .+ f.g.y.^2).^2
   return PotentialQuadratic2D(V, α, γx, γy, κ4)
end

function PotentialQuarticQuadratic(f::AbstractField3D, α::Real, γx::Real, γy::Real, γz::Real, κ4::Real)
   V = 0.5*(1-α)*(γx*f.g.x.^2 .+ γy*f.g.y.^2 .+ γz*f.g.z.^2) .+ 0.5*κ4*(f.g.x.^2 + f.g.y.^2).^2
   return PotentialQuadratic3D(V, α, γx, γy, γz, κ4)
end

function PotentialQuarticQuadratic(V, f::AbstractField2D, conf::AbstractConfig)
   α  = retrieve(conf, "potential", "alpha", Float64)
   γx = retrieve(conf, "potential", "gamma_x", Float64)
   γy = retrieve(conf, "potential", "gamma_y", Float64)
   κ4 = retrieve(conf, "potential", "kappa4", Float64)
   @. V = 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2)+0.5*κ4*(f.g.x^2+f.g.y^2)^2
   return PotentialQuarticQuadratic2D(V, α, γx, γy, κ4)
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

Base.show(io::IO, p::PotentialQuarticQuadratic2D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²) + κ₄/2 r⁴")

Base.show(io::IO, p::PotentialQuarticQuadratic3D) =
     print(io, "Quartic-Quadratic Potential\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)  γz $(p.γz) κ₄ $(p.κ4)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²) + κ₄/2 r⁴")
