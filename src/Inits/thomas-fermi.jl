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
   @. init.f.ϕ = √(
            (im*
             √(init.β*init.γx*init.γy/π) - # μ
             0.5*((init.γx*x)^2+(init.γy*y)^2) # V
            )/init.β
           )
   normalize!(init.f)
   return nothing
end

function initField!(init::InitThomasFermi3D{F}) where {F<:AbstractField3D}
   x = init.f.x
   y = init.f.y
   z = init.f.z
   @. init.f.ϕ = √(im*
            (
             0.5 * (15. * init.β * init.γx * init.γy * init.γz / 4. / π)^(2. / 5.) - # μ
             0.5 * ( init.γx*x^2 + init.γy*y^2 + init.γz*z^2 )  # V
            )/init.β
           )
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