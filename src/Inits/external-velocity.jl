struct InitExternalVelocity{F} <: AbstractInit{F}
   f :: F
   ξ :: Real
   coeffΔ :: Real
   type :: String
end

function InitExternalVelocity(f :: F, coeffΔ :: Real, β :: Real) where {F<:AbstractField}
   ξ = √(-coeffΔ / β)
   return InitExternalVelocity{F}(f, ξ, coeffΔ, "TG")
end

Base.show(io::IO, init::InitExternalVelocity) =
   print(io, "InitExternalVelocity\n",
      "  └───────────── type: $(init.type)")

function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField2D}
   ξ = init.ξ
   coeffΔ = init.coeffΔ
   function TG(x,y)
      λ=cos(x)*√(2.)
      μ=cos(y)*√(2.)
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end

   @. init.f.ϕ = TG(init.f.g.x, init.f.g.y)
   return nothing
end

function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField3D}
   ξ = init.ξ
   coeffΔ = init.coeffΔ
   function TG(x,y,z)
      λ=cos(x)*√(2*abs(cos(z)))
      μ=cos(y)*√(2*abs(cos(z)))*sign(cos(z))
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end

   @. init.f.ϕ = TG(init.f.g.x, init.f.g.y, init.f.g.z)
   return nothing
end