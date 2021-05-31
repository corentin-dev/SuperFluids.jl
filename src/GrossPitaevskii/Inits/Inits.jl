"""
    AbstractInit

Abstract supertype for field initialization classes.
"""
abstract type AbstractInit{F} end

function Init(f::F, ninit::Integer,conf::AbstractConfig) where {F <: AbstractField2D}
   init = nothing
   if ninit == 0
      init = InitNone{typeof(f)}
   elseif ninit == 1
      init = InitThomasFermi2D{typeof(f)}(f,
                       retrieve(conf, "model", "beta", Float64),
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64))
   elseif ninit == 2
      init = InitGauss2D{typeof(f)}(f,
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "model", "omega", Float64))
   elseif ninit == 6
      coeffΔ = retrieve(conf, "model", "delta", Float64)
      β = retrieve(conf, "model", "beta", Float64)
      ξ = √( -coeffΔ / β)
      init = InitExternalVelocity{typeof(f)}(f,ξ,coeffΔ,"taylorgreen")
   end
   return init
end

function Init(f::F, ninit::Integer,conf::AbstractConfig) where {F <: AbstractField3D}
   init = nothing
   if ninit == 0
      init = InitNone{typeof(f)}
   elseif ninit == 1
      init = InitThomasFermi3D{typeof(f)}(f,
                       retrieve(conf, "model", "beta", Float64),
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "potential", "gamma_z", Float64))
   elseif ninit == 2
      init = InitGauss3D{typeof(f)}(f,
                       retrieve(conf, "potential", "gamma_x", Float64),
                       retrieve(conf, "potential", "gamma_y", Float64),
                       retrieve(conf, "potential", "gamma_z", Float64),
                       retrieve(conf, "model", "omega", Float64))
   elseif ninit == 6
      coeffΔ = retrieve(conf, "model", "delta", Float64)
      β = retrieve(conf, "model", "beta", Float64)
      ξ = √( -coeffΔ / β)
      init = InitExternalVelocity{typeof(f)}(f,ξ,coeffΔ,"taylorgreen")
   end
   return init
end

struct InitNone{F} <: AbstractInit{F} end

Base.show(io::IO, init::InitNone) = print(io, "No Init")

struct InitGauss2D{F} <: AbstractInit{F}
   f :: F
   γx :: Real
   γy :: Real
   Ω :: Real
end

function initField!(init::InitGauss2D{F}) where {F<:AbstractField2D}
   ϕ = init.f.ϕ
   x = init.f.g.x
   y = init.f.g.y
   Ω = init.Ω

   @. ϕ = (1-Ω) * (1/sqrt(π)*exp(-0.5*(x^2+y^2))) + Ω * (1*(x+im*y)/sqrt(π)*exp(-0.5*(x^2+y^2)))
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitGauss2D{F}) where {F<:AbstractField2D} =
     print(io, "InitGauss\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) Ω $(init.Ω)\n",
         "  ├─────────────  s1 = 1 / π^1/2 × exp(-1/2 × (x²+y²))\n",
         "  ├─────────────  s2 = (x+iy) / π^1/2 × exp(-1/2 × (x²+y²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

struct InitGauss3D{F} <: AbstractInit{F}
   f :: F
   γx :: Real
   γy :: Real
   γz :: Real
   Ω :: Real
end

function initField!(init::InitGauss3D{F}) where {F<:AbstractField3D}
   ϕ = init.f.ϕ
   x = init.f.g.x
   y = init.f.g.y
   z = init.f.g.z
   γx = init.γx
   γy = init.γy
   γz = init.γz
   Ω = init.Ω

   s1 = similar(ϕ)
   s2 = similar(ϕ)
   @. s1 = exp(-0.5*(x^2+y^2+γz*z^2))
   s1 *= γz^0.25/π^0.75
   @. s2 = (x+im*y+0*z)*exp(-0.5*(x^2+y^2+γz*z^2)) 
   s2 *= γz^0.25/π^0.75
   @. ϕ = (1-Ω) * s1 + Ω * s2
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitGauss3D{F}) where {F<:AbstractField3D} =
     print(io, "InitGauss\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) γz $(init.γz) Ω $(init.Ω)\n",
         "  ├─────────────  s1 = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  ├─────────────  s2 = γz^1/4 (x+iy) / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

struct InitThomasFermi2D{F} <: AbstractInit{F}
   f :: F
   β :: Real
   γx :: Real
   γy :: Real
end

function initField!(init::InitThomasFermi2D{F}) where {F<:AbstractField2D}
   ϕ = init.f.ϕ
   x = init.f.g.x
   y = init.f.g.y
   β = init.β
   γx = init.γx
   γy = init.γy
   @. ϕ = √(
            (
             √(β*γx*γy/π) - # μ
             0.5*((γx*x)^2+(γy*y)^2) # V
            )/β
           )
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitThomasFermi2D{F}) where {F<:AbstractField2D} =
     print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) β $(init.β)\n",
         "  ├──────────────  μ = √(β γx γy)\n",
         "  └──────────────  ϕ = √( √μ - V )")

struct InitThomasFermi3D{F} <: AbstractInit{F}
   f :: F
   β :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

function initField!(init::InitThomasFermi3D{F}) where {F<:AbstractField3D}
   ϕ = init.f.ϕ
   x = init.f.g.x
   y = init.f.g.y
   z = init.f.g.z
   β = init.β
   γx = init.γx
   γy = init.γy
   γz = init.γz
   @. ϕ = √(
            (
             0.5 * (15. * β * γx * γy * γz / 4. / π)^(2. / 5.) - # μ
             0.5 * ( (γx*x)^2 + (γy*y)^2 + (γz*z)^2 )  # V
            )/β
           )
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitThomasFermi3D{F}) where {F<:AbstractField3D} =
     print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) γz $(init.γz) β $(init.β)\n",
         "  ├──────────────  μ = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

struct InitExternalVelocity{F} <: AbstractInit{F}
   f :: F
   ξ :: Real
   coeffΔ :: Real
   type :: String
end

function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField2D}
   # references
   ξ = init.ξ
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
