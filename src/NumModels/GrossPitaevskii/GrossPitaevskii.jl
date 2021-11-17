export GrossPitaevskiiParameters

"""
    GrossPitaevskiiParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.
"""
struct GrossPitaevskiiParameters <: AbstractParameters
   "coefficient in front of Δ term"
   coeffΔ :: Real
   "interaction coefficient"
   β :: Real
   "potential"
   V :: Function
   "rotation coefficient"
   Ω :: Real
end

"""
    GrossPitaevskiiParameters(;
      coeffΔ = -0.5 :: Real,
      β = 1. :: Real,
      Ω = 0. :: Real)

Returns a GrossPitaevskiiParameters.

Example
=======
```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8)
GrossPitaevskiiParameters
  ├──────  equation solved:
  ├─────── idϕ/dt = -0.5 Δϕ + 1000 |ϕ|²ϕ  + V(x)ϕ - i Lz 0.8 ϕ
  └──────────────── coeffΔ: -0.5, β: 1000, Ω: 0.8
```
"""
function GrossPitaevskiiParameters(;
      coeffΔ :: Real = -0.5,
      β :: Real = 1.,
      V :: Function = (x...) -> 0.,
      Ω :: Real = 0.)
   return GrossPitaevskiiParameters(coeffΔ, β, V, Ω)
end

Base.show(io::IO, param::GrossPitaevskiiParameters) =
     print(io, "GrossPitaevskiiParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── idϕ/dt = $(param.coeffΔ) Δϕ + $(param.β) |ϕ|²ϕ  + V(x)ϕ - i Lz $(param.Ω) ϕ \n",
         "  └──────────────── coeffΔ: $(param.coeffΔ), β: $(param.β), Ω: $(param.Ω)")

@inline function non_linear(field::AbstractField2D, param::GrossPitaevskiiParameters)
   return param.V.(field.g.x, field.g.y)              .+ param.β .* real.( field.ϕ .* conj.(field.ϕ) )
end

@inline function non_linear(field::AbstractField3D, param::GrossPitaevskiiParameters)
   return param.V.(field.g.x, field.g.y, n.field.g.z) .+ param.β .* real.( field.ϕ .* conj.(field.ϕ) )
end

function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   return -coeffΔ .* ( n.gf.ddx .+ n.gf.ddy) + Ω .* im .* (n.gf.rx+n.gf.ry)
end

function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField3D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   return -coeffΔ .* ( n.gf.ddx .+ n.gf.ddy .+ n.gf.ddz) + Ω .* im .* (n.gf.rx+n.gf.ry)
end

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField2D,P<:GrossPitaevskiiParameters,Plan}
   # references
   ϕ = n.f.ϕ
   coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
   x, y = n.f.g.x, n.f.g.y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V = n.param.V

   computeDerivatives!(n.gf, n.plan, ϕ)

   abs∇ϕ_x = sum( real.( n.gf.dx .* conj(n.gf.dx) ) )
   abs∇ϕ_y = sum( real.( n.gf.dy .* conj(n.gf.dy) ) )

   # compute energies
   EΩ = sum( real.( im.*conj.(ϕ).*(
      Ω.*( n.gf.rx .+  n.gf.ry )
      ) ) ) * Δx * Δy
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(real.(V.(x,y)).*real.(ϕ.*conj.(ϕ)))) * Δx * Δy
   Eβ = sum( 0.5*β*(real.(ϕ.*conj.(ϕ)).^2)) * Δx * Δy
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println("Angular Momentum Energy : $(EΩ)")
      println("Kinetic + Potential Energy : $(EΔ)")
      println("Interaction Energy : $(Eβ)")
      println("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters,Plan<:AbstractFFTPlan}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat_z = n.f.ϕ, n.plan.ϕ_hat, n.plan.ϕ_hat, n.plan.ϕ_hat
   coeffΔ, Ω, β = n.coeffΔ, n.Ω, n.β
   x, y, z = n.f.g.x, n.f.g.y, n.f.g.z
   ξx, ξy, ξz = n.plan.ξx, n.plan.ξy, n.plan.ξz
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V = n.param.V
   # perform FFT
   mul!(ϕhat_x, plan_x, ϕ)
   # compute grad x
   ∇ϕ_hat = im .* ξx .* ϕhat_x
   ∇ϕ_x = plan_x \ ∇ϕ_hat # we get grad x
   abs∇ϕ_x = sum(real.(∇ϕ_x.*conj(∇ϕ_x)))
   # compute rotx
   ∇ϕ_hat .= im .*  Ω .* y .* ξx .* ϕhat_x
   ldiv!(∇ϕ_x, plan_x, ∇ϕ_hat) # we get rot x
   # compute grad y
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_hat = im .* ξy .* ϕhat_y
   ∇ϕ_y = plan_y \ ∇ϕ_hat # we get grad y
   abs∇ϕ_y = sum(real.(∇ϕ_y.*conj(∇ϕ_y)))
   # compute roty
   ∇ϕ_hat = im .* -Ω .* x .* ξy .* ϕhat_y
   ldiv!(∇ϕ_y, plan_y, ∇ϕ_hat) # we get rot y
   # compute grad z
   mul!(ϕhat_z, plan_z, ϕ)
   ∇ϕ_hat = im .* ξz .* ϕhat_z
   ∇ϕ_z = plan_z \ ∇ϕ_hat # we get grad x
   abs∇ϕ_z = sum(real.(∇ϕ_z.*conj(∇ϕ_z)))
   # computing energies (locally)
   EΩ = sum(real.(im.*conj.(ϕ).*(∇ϕ_x.+∇ϕ_y+∇ϕ_z))) * Δx * Δy *Δz
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y.+abs∇ϕ_z) + sum(V.(x,y,z).*real.(ϕ.*conj.(ϕ)))) * Δx * Δy * Δz
   Eβ = sum( 0.5*β*(real.(ϕ.*conj.(ϕ)).^2)) * Δx * Δy * Δz
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println("Angular Momentum Energy : $(EΩ)")
      println("Kinetic + Potential Energy : $(EΔ)")
      println("Interaction Energy : $(Eβ)")
      println("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end
