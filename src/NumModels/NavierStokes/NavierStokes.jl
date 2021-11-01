export NavierStokesParameters
export taylor_green!

"""
    NavierStokesParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.
"""
struct NavierStokesParameters <: AbstractParameters
   "viscosity"
   ν :: Real
   "volumic mass"
   ρ :: Real
end

"""
    NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1)

Returns a NavierStokesParameters.

Example
=======
```jldoctest
julia> param = NavierStokesParameters()
NavierStokesParameters
  ├──────  equation solved:
  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu
  ├─────── ∇⋅u = 0
  └─────── ν: 0.001
```
"""
function NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1)
   return NavierStokesParameters(ν,ρ)
end

Base.show(io::IO, param::NavierStokesParameters) =
     print(io, "NavierStokesParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu\n",
         "  ├─────── ∇⋅u = 0\n",
         "  └─────── ν: $(param.ν) ρ: $(param.ρ)")

function non_linear(field::AbstractField3D, param::NavierStokesParameters)
   return 0
end

function taylor_green!(u,x,y,z;θ::Real=0)
   @. u[:,:,:,1] = 2/√3 * sin(θ+2π/3) * sin(x) * cos(y) * cos(z)
   @. u[:,:,:,2] = 2/√3 * sin(θ-2π/3) * cos(x) * sin(y) * cos(z)
   @. u[:,:,:,3] = 2/√3 * sin(θ)      * cos(x) * cos(y) * sin(z)
   return nothing
end

function rhs(n :: AbstractNumModel{F,P}) where {F<:AbstractField3D, P<:NavierStokesParameters}
   # references
   ξx = n.plan.ξx
   ξy = n.plan.ξy
   ξz = n.plan.ξz
   # computation
   ω_hat = curl_hat(n.ϕ_hat, ξx, ξy, ξz)
   ω = n.plan.plan \ ω_hat
   # dU = u ∧ ω
   dU = cross(n.f.ϕ, ω)
   # dU_hat = F(u ∧ ω)
   dU_hat = n.plan.plan * dU
   #dealias dU_hat
   dealias!(dU_hat, ξx, ξy, ξz)
   # P_hat = ∇ ⋅ dU / Δ
   P_hat = - im * (
            ξx .* dU_hat[:,:,:,1] .+
            ξy .* dU_hat[:,:,:,2] .+
            ξz .* dU_hat[:,:,:,3]) ./ ξsquared.(ξx, ξy, ξz)
   # dU = (u∧ω) - νΔu
   dU_hat .-= n.param.ν .* (ξx.^2 .+ ξy.^2 .+ ξz.^2) .* n.ϕ_hat
   # dU = (u∧ω) - νΔu - ∇P
   dU_hat[:,:,:,1] .-= im .* ξx .* P_hat
   dU_hat[:,:,:,2] .-= im .* ξy .* P_hat
   dU_hat[:,:,:,2] .-= im .* ξz .* P_hat
   return dU_hat
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField2D, P<:NavierStokesParameters}
   E = sum( 0.5 .* real.(
      n.f.ϕ[:,:,:,1].^2 .+
      n.f.ϕ[:,:,:,2].^2 .+
      n.f.ϕ[:,:,:,3].^2 ) .* ( n.f.g.Δx * n.f.g.Δy * n.f.g.Δz )
   )
   if(showEnergy)
      println("E = $(E)")
   end
   return 0., E, 0., E
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField3D, P<:NavierStokesParameters}
   E = sum( 0.5 .* real.(
                    n.f.ϕ[:,:,:,1].^2 .+
                    n.f.ϕ[:,:,:,2].^2 .+
                    n.f.ϕ[:,:,:,3].^2 ) .* ( n.f.g.Δx * n.f.g.Δy * n.f.g.Δz )
          )
   if(showEnergy)
      println("E = $(E)")
   end
   return 0., E, 0., E
end
