struct InitExternalVelocity{F} <: AbstractInit{F}
   f :: F
   ξ :: Real
   coeffΔ :: Real
   type :: String
end

function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField2D}
   # references
   ξ = init.ξ
   #ξ = √(-coeffΔ / β)
   coeffΔ = init.coeffΔ
   ϕ = init.f.ϕ
   x, y = init.f.g.x, init.f.g.y
   for I in CartesianIndices(init.f.ϕ)
      ix, iy = Tuple(I)
      λ=cos(x[ix])*√(2.)
      μ =cos(y[iy])*√(2.)
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      ϕ[ix,iy] = (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end
   return nothing
end

function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField3D}
   # references
   ξ = init.ξ
   #ξ = √(-coeffΔ / β)
   coeffΔ = init.coeffΔ
   ϕ = init.f.ϕ
   x, y, z = init.f.g.x, init.f.g.y, init.f.g.z
   for I in CartesianIndices(init.f.ϕ)
      ix, iy, iz = Tuple(I)
      λ=cos(x[ix])*√(2*abs(cos(z[iz])))
      μ =cos(y[iy])*√(2*abs(cos(z[iz])))*sign(cos(z[iz]))
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      ϕ[ix,iy, iz] =(vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end
   return nothing
end

Base.show(io::IO, init::InitExternalVelocity) =
     print(io, "InitExternalVelocity\n",
         "  └───────────── type: $(init.type)")




function InitExternalVelocity(f :: AbstractField2D, coeffΔ, β)
   ξ = √(-coeffΔ / β)
   function TG(x,y)
      λ=cos(x)*√(2.)
      μ=cos(y)*√(2.)
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end
   return TG
end

function InitExternalVelocity(f :: AbstractField3D, coeffΔ, β)
   ξ = √(-coeffΔ / β)
   function TG(x,y,z)
      λ=cos(x)*√(2*abs(cos(z)))
      μ=cos(y)*√(2*abs(cos(z)))*sign(cos(z))
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end
   return TG
end