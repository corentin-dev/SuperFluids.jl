struct PotentialTaylorGreen2D <: AbstractPotential2D
   V
   uadvx
   uadvy
end

struct PotentialTaylorGreen3D <: AbstractPotential3D
   V
   uadvx
   uadvy
   uadvz
end

function PotentialTaylorGreen(f::AbstractField2D, coeffΔ::Real, β::Real)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y)
   uadvy = -cos.(f.g.x).*sin.(f.g.y)
   V = ( uadvx.^2 .+ uadvy.^2 ) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen2D(V, uadvx, uadvy)
end

function PotentialTaylorGreen(f::AbstractField3D, coeffΔ::Real, β::Real)
   # velocity field
   uadvx =  sin.(f.g.x).*cos.(f.g.y).*cos.(f.g.z)
   uadvy = -cos.(f.g.x).*sin.(f.g.y).*cos.(f.g.z)
   uadvz =  zeros(Real, size(uadvx))
   V = ( uadvx.^2 .+ uadvy.^2 .+ uadvz.^2) ./ (-4*coeffΔ .- β)
   return PotentialTaylorGreen3D(V, uadvx, uadvy, uadvz)
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

Base.show(io::IO, p::PotentialTaylorGreen2D) = print(io, "TaylorGreen Potential")

Base.show(io::IO, p::PotentialTaylorGreen3D) = print(io, "TaylorGreen Potential")
