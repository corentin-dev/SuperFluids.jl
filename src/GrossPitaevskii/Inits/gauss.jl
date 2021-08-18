struct InitGauss2D{F} <: AbstractInit{F}
   f :: F
   γx :: Real
   γy :: Real
   Ω :: Real
end

struct InitGauss3D{F} <: AbstractInit{F}
   f :: F
   γx :: Real
   γy :: Real
   γz :: Real
   Ω :: Real
end

function InitGauss(f :: F, γx :: Real, γy :: Real, Ω :: Real) where F
   return InitGauss2D{F}(f, γx, γy, Ω)
end

function InitGauss(f :: F, γx :: Real, γy :: Real, γz :: Real, Ω :: Real) where F
   return InitGauss3D{F}(f, γx, γy, γz, Ω)
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

Base.show(io::IO, init::InitGauss2D{F}) where {F<:AbstractField2D} =
     print(io, "InitGauss\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) Ω $(init.Ω)\n",
         "  ├─────────────  s1 = 1 / π^1/2 × exp(-1/2 × (x²+y²))\n",
         "  ├─────────────  s2 = (x+iy) / π^1/2 × exp(-1/2 × (x²+y²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

Base.show(io::IO, init::InitGauss3D{F}) where {F<:AbstractField3D} =
     print(io, "InitGauss\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) γz $(init.γz) Ω $(init.Ω)\n",
         "  ├─────────────  s1 = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  ├─────────────  s2 = γz^1/4 (x+iy) / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")
