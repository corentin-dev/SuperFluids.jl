struct InitThomasFermi2D{F} <: AbstractInit{F}
   f :: F
   β :: Real
   γx :: Real
   γy :: Real
end

struct InitThomasFermi3D{F} <: AbstractInit{F}
   f :: F
   β :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

function InitThomasFermi(f :: F, β :: Real; γx :: Real = 1., γy :: Real = 1.) where {F<:AbstractField2D}
   return InitThomasFermi2D{F}(f, β, γx, γy)
end

function InitThomasFermi(f :: F, β :: Real; γx :: Real = 1., γy :: Real = 1., γz :: Real = 1.) where {F<:AbstractField3D}
   return InitThomasFermi3D{F}(f, β, γx, γy, γz)
end

function initField!(init::InitThomasFermi2D{F}) where {F<:AbstractField2D}
   x = init.f.x
   y = init.f.y
   ρ0 = √( 4* init.β * √(init.γx * init.γy) / π )
   function TF(x,y)
      if x^2 + y^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 )
      end
   end
   @. init.f.ϕ = TF(x,y)
   normalize!(init.f)
   return nothing
end

function initField!(init::InitThomasFermi3D{F}) where {F<:AbstractField3D}
   x = init.f.x
   y = init.f.y
   z = init.f.z
   ρ0 = ( 30 * init.β * √(init.γx * init.γy * init.γz) / ( 8 * π ) )^(2/5)
   function TF(x,y,z)
      if x^2 + y^2 + z^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 + z^2)
      end
   end
   @. init.f.ϕ = TF(x,y,z)
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitThomasFermi3D{F}) where {F<:AbstractField3D} =
     print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) γz $(init.γz) β $(init.β)\n",
         "  ├──────────────  μ = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

Base.show(io::IO, init::InitThomasFermi2D{F}) where {F<:AbstractField2D} =
     print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) β $(init.β)\n",
         "  ├──────────────  μ = √(β γx γy)\n",
         "  └──────────────  ϕ = √( √μ - V )")