export PotentialZero2D, PotentialZero3D
export PotentialQuarticQuadratic2D, PotentialQuarticQuadratic3D
export PotentialQuadratic2D, PotentialQuadratic3D

function PotentialZero2D()
   return (x,y) -> 0
end

function PotentialZero3D()
   return (x,y,z) -> 0
end

include("quadratic.jl")
include("quartic.jl")
include("external-velocity.jl")
