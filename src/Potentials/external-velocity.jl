struct PotentialTaylorGreen{F} <: AbstractPotential{F}
   f :: F
   V :: AbstractArray
   uadvx :: AbstractArray
   uadvy :: AbstractArray
   uadvz :: AbstractArray
end

function PotentialTaylorGreen(f::F; α :: Real = 1) where {F<:AbstractField2D}
   V = real.(similar(f.ϕ))
   uadvx = similar(V)
   uadvy = similar(V)
   uadvz = []
   # velocity field
   @. uadvx =  sin(f.x)*cos.(f.y)
   @. uadvy = -cos(f.x)*sin.(f.y)
   # potential
   @. V = ( uadvx^2 + uadvy^2 ) / α # (-4*param.coeffΔ .- param.β)
   return PotentialTaylorGreen{F}(f, V, uadvx, uadvy, uadvz)
end

function PotentialTaylorGreen(f::F; α :: Real = 1) where {F<:AbstractField3D}
   V = real.(similar(f.ϕ))
   uadvx = similar(V)
   uadvy = similar(V)
   uadvz = similar(V)
   # velocity field
   @. uadvx =  sin(f.x)*cos.(f.y).*cos.(f.z)
   @. uadvy = -cos(f.x)*sin.(f.y).*cos.(f.z)
   @. uadvz .= 0.
   # potential
   @. V = ( uadvx^2 + uadvy^2 + uadvz^2 ) / α # (-4*param.coeffΔ .- param.β)
   return PotentialTaylorGreen{F}(f, V, uadvx, uadvy, uadvz)
end

Base.show(io::IO, p::PotentialTaylorGreen) = print(io, "TaylorGreen Potential")