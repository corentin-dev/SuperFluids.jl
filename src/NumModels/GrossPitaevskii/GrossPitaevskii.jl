function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   tmp = similar(ϕt)
   @. tmp = -coeffΔ * ( n.gf.ddx + n.gf.ddy) + Ω * im * (n.gf.rx + n.gf.ry)
   return tmp
end

function lapRot(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField3D, P<:GrossPitaevskiiParameters, Plan}
   # references
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # compute derivatives
   computeDerivatives!(n.gf, n.plan, ϕt)
   # return computation
   tmp = similar(ϕt)
   @. tmp = -coeffΔ * ( n.gf.ddx + n.gf.ddy + n.gf.ddz) + Ω * im * (n.gf.rx + n.gf.ry)
   return tmp
end

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField2D,P<:GrossPitaevskiiParameters,Plan}
   # references
   ϕ = n.f.ϕ
   coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V = n.param.pot.V

   computeDerivatives!(n.gf, n.plan, ϕ)

   abs∇ϕ_x = sum( abs2.( n.gf.dx ) )
   abs∇ϕ_y = sum( abs2.( n.gf.dy ) )

   # compute energies
   EΩ = sum( real.( im.*conj.(ϕ).*(
      Ω.*( n.gf.rx .+  n.gf.ry )
      ) ) ) * Δx * Δy
   EΔ = (-coeffΔ .* (abs∇ϕ_x.+abs∇ϕ_y) + sum(V.*abs2.(ϕ))) * Δx * Δy
   Eβ = sum( 0.5*β*(abs2.(ϕ).^2)) * Δx * Δy
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println_parallel("Angular Momentum Energy : $(EΩ)")
      println_parallel("Kinetic + Potential Energy : $(EΔ)")
      println_parallel("Interaction Energy : $(Eβ)")
      println_parallel("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end

function energy(n::AbstractNumModel{F,P,Plan}, showEnergy=false) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters,Plan}
   # references
   ϕ = n.f.ϕ
   coeffΔ, Ω, β = n.param.coeffΔ, n.param.Ω, n.param.β
   Δx, Δy, Δz = n.f.g.Δx, n.f.g.Δy, n.f.g.Δz
   V = n.param.pot.V

   computeDerivatives!(n.gf, n.plan, ϕ)

   abs∇ϕ_x = sum( abs2.( n.gf.dx ) )
   abs∇ϕ_y = sum( abs2.( n.gf.dy ) )
   abs∇ϕ_z = sum( abs2.( n.gf.dz ) )

   # compute energies
   EΩ = sum( real.( im.*conj.(ϕ).*(
      Ω.*( n.gf.rx .+  n.gf.ry )
      ) ) ) * Δx * Δy * Δz
   EΔ = (-coeffΔ * (abs∇ϕ_x.+abs∇ϕ_y.+abs∇ϕ_z) + sum(V.*abs2.(ϕ))) * Δx * Δy * Δz
   Eβ = sum( 0.5*β*(abs2.(ϕ).^2)) * Δx * Δy * Δz
   # compute sum
   E = -EΩ + EΔ + Eβ

   if(showEnergy)
      println_parallel("Angular Momentum Energy : $(EΩ)")
      println_parallel("Kinetic + Potential Energy : $(EΔ)")
      println_parallel("Interaction Energy : $(Eβ)")
      println_parallel("Total Energy: $(E)")
   end

   return EΩ, EΔ, Eβ, E
end

include("backward-euler.jl")
include("crank-nicolson.jl")
include("adi.jl")
include("external-velocity.jl")
