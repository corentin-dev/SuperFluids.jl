export NavierStokesParameters

"""
    NavierStokesParameters <: AbstractParameters

Parameters to compute Gross-Pitarvskii equations.
"""
struct NavierStokesParameters <: AbstractParameters
   "viscosity"
   ν :: Real
   "volumic mass"
   ρ :: Real
   "forcing term"
   f :: Function
end

"""
    NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1,
      f :: Function)

Returns a NavierStokesParameters.

Example
=======
```jldoctest
julia> param = NavierStokesParameters()
NavierStokesParameters
  ├──────  equation solved:
  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu + f
  ├─────── ∇⋅u = 0
  └─────── ν: 0.001
```
"""
function NavierStokesParameters(;
      ν :: Real = 0.001,
      ρ :: Real = 1,
      f :: Function)
   return NavierStokesParameters(coeffΔ, β, V, Ω)
end

Base.show(io::IO, param::NavierStokesParameters) =
     print(io, "NavierStokesParameters\n",
         "  ├──────  equation solved:\n",
         "  ├─────── du/dt + ∇⋅(uu) = -1/ρ ∇p + ν Δu + f\n",
         "  ├─────── ∇⋅u = 0\n",
         "  └─────── ν: $(param.ν) ρ: $(param.ρ)")

function non_linear(field<:AbstractField3D, param<:NavierStokesParameters)
   return 0
end

function curl_hat(v_hat, ξx, ξy, ξz)
   v_curl_hat = similar(v_hat)
   @. v_curl_hat[:,:,:,1] = im * (ξy * v_hat[:,:,:,3] - ξz * v_hat[:,:,:,2])
   @. v_curl_hat[:,:,:,2] = im * (ξz * v_hat[:,:,:,1] - ξx * v_hat[:,:,:,3])
   @. v_curl_hat[:,:,:,3] = im * (ξx * v_hat[:,:,:,2] - ξy * v_hat[:,:,:,1])
   return v_curl_hat
end

function cross(a, b)
   @assert sice(a) == sice(b)
   c = similar(a)
   @. c[:,:,:,1] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   @. c[:,:,:,2] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   @. c[:,:,:,3] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   return c
end

function dealias!(u_hat, ξx, ξy, ξz)
   ξmax = 4/9 * minimum((maximum(ξx.^2),maximum(ξy.^2),maximum(ξz.^2)))
   u_hat[:,:,:,1] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   u_hat[:,:,:,2] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   u_hat[:,:,:,3] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   return nothing
end

function ξsquared(ξx::Real, ξy::Real, ξz::Real)
   a = ξx^2 + ξy^2 + ξz^2
   if abs(a) < 1e-8
      return 1
   else
      return a
   end
end

function rhs(n <: AbstractNumModel{F,P}) where {F<:AbstractField3D, P<:NavierStokesParameters}
   # references
   ξx = n.plan.ξx
   ξy = n.plan.ξy
   ξz = n.plan.ξz
   # computation
   ω_hat = curl_hat(n.f.ϕ_hat, ξx, ξy, ξz) 
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
   dU_hat .-= param.ν .* (ξx.^2 .+ ξy.^2 .+ ξz.^2) .* n.f.ϕ_hat
   # dU = (u∧ω) - νΔu - ∇P
   dU_hat[:,:,:,1] .-= im .* ξx .* P_hat
   dU_hat[:,:,:,2] .-= im .* ξy .* P_hat
   dU_hat[:,:,:,2] .-= im .* ξz .* P_hat
   return dU_hat
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField2D,P<:NavierStokesParameters}

   if(showEnergy)
      println("for the moment no energy is computed")
   end

   return nothing
end

function energy(n::AbstractNumModel{F}, showEnergy=false) where {F<:AbstractField3D}

   if(showEnergy)
      println("for the moment no energy is computed")
   end

   return nothing
end
