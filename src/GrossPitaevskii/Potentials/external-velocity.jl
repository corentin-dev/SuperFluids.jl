function PotentialTaylorGreen(f::AbstractField2D, β::Real; coeffΔ::Real=-0.5)
   return (x,y) <- ( (sin(x)*cos(y))^2 + (-cos(x)*sin(y))^2 ) / (-4*coeffΔ - β)
end

function PotentialTaylorGreen(f::AbstractField3D, β::Real; coeffΔ::Real=-0.5)
   return (x,y,z) <- ( (sin(x)*cos(y)*cos(z))^2 + (-cos(x)*sin(y)*cos(z))^2 ) / (-4*coeffΔ - β)
end

# function PotentialTaylorGreen(f::AbstractField3D, coeffΔ::Real, β::Real)
#    # velocity field
#    uadvx =  sin.(f.g.x).*cos.(f.g.y).*cos.(f.g.z)
#    uadvy = -cos.(f.g.x).*sin.(f.g.y).*cos.(f.g.z)
#    uadvz =  zeros(Real, size(uadvx))
#    V = ( uadvx.^2 .+ uadvy.^2 .+ uadvz.^2) ./ (-4*coeffΔ .- β)
#    return PotentialTaylorGreen3D(V, uadvx, uadvy, uadvz)
# end

# function PotentialTaylorGreen(f::AbstractField2D, coeffΔ::Real, β::Real)
#    # velocity field
#    uadvx =  sin.(f.g.x).*cos.(f.g.y)
#    uadvy = -cos.(f.g.x).*sin.(f.g.y)
#    V = ( uadvx.^2 .+ uadvy.^2 ) ./ (-4*coeffΔ .- β)
#    return PotentialTaylorGreen2D(V, uadvx, uadvy)
# end

# function PotentialTaylorGreen(f::AbstractField3D, coeffΔ::Real, β::Real)
#    # velocity field
#    uadvx =  sin.(f.g.x).*cos.(f.g.y).*cos.(f.g.z)
#    uadvy = -cos.(f.g.x).*sin.(f.g.y).*cos.(f.g.z)
#    uadvz =  zeros(Real, size(uadvx))
#    V = ( uadvx.^2 .+ uadvy.^2 .+ uadvz.^2) ./ (-4*coeffΔ .- β)
#    return PotentialTaylorGreen3D(V, uadvx, uadvy, uadvz)
# end
