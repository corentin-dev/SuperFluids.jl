struct PotentialQuadratic{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
   α :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

function PotentialQuadratic(f::F; α::Real = 0, γx::Real = 1, γy::Real = 1, γz::Real = 1) where {F<:AbstractField{FT,FFT,A}} where {FT,FFT,A}
   V = PencilArray(f.ϕ.pencil, A{FT}(undef, size_local(f.ϕ)))

   p = PotentialQuadratic{F}(f, V, α, γx, γy, γz)
   compute!(p)
   return p
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField2D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.x^2+p.γy*p.f.y^2)
end

function compute!(p::PotentialQuadratic{F}) where {F<:AbstractField3D}
   @. p.V = 0.5*(1-p.α)*(p.γx*p.f.x^2+p.γy*p.f.y^2+p.γz*p.f.z^2)
end

Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField2D} =
     print(io, "Quadratic Potential for 2D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y²)")

Base.show(io::IO, p::PotentialQuadratic{F}) where {F<:AbstractField3D} =
     print(io, "Quadratic Potential for 3D fields\n",
         "  ├──────  parameters: α $(p.α) γx $(p.γx) γy $(p.γy) γz $(p.γz)\n",
         "  └──────────────  V = (1-α)/2 × (γx x² + γy y² + γz z²)")