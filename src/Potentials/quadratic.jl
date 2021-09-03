function PotentialQuadratic2D(;α::Real = 0, γx::Real = 1, γy::Real = 1)
   return (x,y) -> 0.5*(1-α)*(γx*x^2+γy*y^2)
end

function PotentialQuadratic3D(;α::Real = 0, γx::Real = 1, γy::Real = 1, γz::Real = 1)
   return (x,y,z) -> 0.5*(1-α)*(γx*f.g.x^2+γy*f.g.y^2+γz*f.g.z^2)
end
