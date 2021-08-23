export PotentialZero, PotentialQuadratic, PotentialQuarticQuadratic

function PotentialZero(f::AbstractField2D)
   return (x,y) -> 0
end

function PotentialZero(f::AbstractField3D)
   return (x,y,z) -> 0
end

include("quadratic.jl")
include("quartic.jl")
include("external-velocity.jl")
