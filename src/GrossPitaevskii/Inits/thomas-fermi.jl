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

function InitThomasFermi(f :: F, β :: Real, γx :: Real, γy :: Real) where F
   return InitThomasFermi2D{F}(f, β, γx, γy)
end

function InitThomasFermi(f :: F, β :: Real, γx :: Real, γy :: Real, γz :: Real) where F
   return InitThomasFermi3D{F}(f, β, γx, γy, γz)
end

function initField!(init::InitThomasFermi2D{F}) where {F<:AbstractField2D}
   ϕ = init.f.ϕ
   x = init.f.g.x
   y = init.f.g.y
   β = init.β
   γx = init.γx
   γy = init.γy
   @. ϕ = √(
            (im*
             √(β*γx*γy/π) - # μ
             0.5*((γx*x)^2+(γy*y)^2) # V
            )/β
           )
   normalize!(init.f)
   return nothing
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
   @. ϕ = √(im*
            (
             0.5 * (15. * β * γx * γy * γz / 4. / π)^(2. / 5.) - # μ
             0.5 * ( γx*x^2 + γy*y^2 + γz*z^2 )  # V
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

Base.show(io::IO, init::InitThomasFermi2D{F}) where {F<:AbstractField2D} =
     print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) β $(init.β)\n",
         "  ├──────────────  μ = √(β γx γy)\n",
         "  └──────────────  ϕ = √( √μ - V )")
