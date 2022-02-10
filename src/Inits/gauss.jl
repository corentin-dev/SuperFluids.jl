struct InitGauss2D{F} <: AbstractInit{F}
   f :: F
   Ω :: Real
end

struct InitGauss3D{F} <: AbstractInit{F}
   f :: F
   γz :: Real
   Ω :: Real
end

function InitGauss(f :: F; Ω :: Real = 0.) where {F<:AbstractField2D}
   return InitGauss2D{F}(f, Ω)
end

function InitGauss(f :: F; γz :: Real = 1., Ω :: Real = 0.) where {F<:AbstractField3D}
   return InitGauss3D{F}(f, γz, Ω)
end

function initField!(init::InitGauss2D{F}) where {F<:AbstractField2D}
   @. init.f.ϕ = (1-init.Ω) * (1/sqrt(π)*exp(-0.5*(init.f.x^2+init.f.y^2))) + init.Ω * (1*(init.f.x+im*init.f.y)/sqrt(π)*exp(-0.5*(init.f.x^2+init.f.y^2)))
   normalize!(init.f)
   return nothing
end

function initField!(init::InitGauss3D{F}) where {F<:AbstractField3D}
   x = init.f.x
   y = init.f.y
   z = init.f.z

   @. init.f.ϕ = (1-init.Ω) * exp(-0.5*(x^2+y^2+init.γz*z^2))*(init.γz^0.25/π^0.75) + init.Ω * (x+im*y+0*z)*exp(-0.5*(x^2+y^2+init.γz*z^2))*(init.γz^0.25/π^0.75)
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