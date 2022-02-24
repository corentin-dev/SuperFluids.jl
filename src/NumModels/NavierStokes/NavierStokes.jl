export taylor_green!

function taylor_green!(f,x,y,z;θ::Real=0)
   @. f.ux[:,:,:] = 2/√3 * sin(θ+2π/3) * sin(x) * cos(y) * cos(z)
   @. f.uy[:,:,:] = 2/√3 * sin(θ-2π/3) * cos(x) * sin(y) * cos(z)
   @. f.uz[:,:,:] = 2/√3 * sin(θ)      * cos(x) * cos(y) * sin(z)
   return nothing
end

function rhs(n :: AbstractNumModel{F,P}) where {F<:AbstractField3D, P<:NavierStokesParameters}
   # references
   ξx = n.plan.ξx
   ξy = n.plan.ξy
   ξz = n.plan.ξz
   gridξ = localgrid(n.plan.pen_z, (ξx, ξy, ξz))
   # compute ω
   computeDerivatives!(n.gf, n.plan, n.f.u)
   # dU = u ∧ ω
   dU = cross(n.f.u, n.gf.ω)
   # dU_hat = F(u ∧ ω)
   uhat = n.plan.uz_hat
   uxtmphat = n.plan.uxtmp_hat
   uytmphat = n.plan.uytmp_hat
   uztmphat = n.plan.uztmp_hat
   uytmp2hat = n.plan.uytmp2_hat
   uztmp2hat = n.plan.uztmp2_hat
   dU_hat = n.plan.uztmp_hat
   for i = 1:3
      mul!(parent(uxtmphat[i]), n.plan.plan_x, parent(dU[i]))
      transpose!(uytmp2hat[i], uxtmphat[i])
      mul!(parent(uytmphat[i]), n.plan.plan_y, parent(uytmp2hat[i]))
      transpose!(uztmp2hat[i], uytmphat[i])
      mul!(parent(dU_hat[i]), n.plan.plan_z, parent(uztmp2hat[i]))
   end
   #dealias dU_hat
   dealias!(dU_hat, ξx, ξy, ξz)
   # P_hat = ∇ ⋅ dU / Δ
   # P_hat = - im * (
   #          ξx .* dU_hat[:,:,:,1] .+
   #          ξy .* dU_hat[:,:,:,2] .+
   #          ξz .* dU_hat[:,:,:,3]) ./ ξsquared.(ξx, ξy, ξz)
   # dU = (u∧ω) - νΔu - ∇P
   @. uztmphat[1] = gridξ.x * dU_hat[1] + gridξ.y * dU_hat[2] + gridξ.z * dU_hat[3]
   for i = 1:3
      @. dU_hat[i] = dU_hat[i] - gridξ[i] * uztmphat[1] / ξsquared(gridξ.x, gridξ.y, gridξ.z)
   end
   # dU_hat[:,:,:,1] .-= im .* ξx .* P_hat
   # dU_hat[:,:,:,2] .-= im .* ξy .* P_hat
   # dU_hat[:,:,:,2] .-= im .* ξz .* P_hat
   # dU = (u∧ω) - νΔu
   for i = 1:3
      @. dU_hat[i] -= n.param.ν * (gridξ.x^2 + gridξ.y^2 + gridξ.z^2) * uhat[i]
   end
   # FFT inv
   for i = 1:3
      ldiv!(parent(uztmp2hat[i]), n.plan.plan_z, parent(dU_hat[i]))
      transpose!(uytmphat[i],uztmp2hat[i])
      ldiv!(parent(uytmp2hat[i]), n.plan.plan_y, parent(uytmphat[i]))
      transpose!(uxtmphat[i],uytmp2hat[i])
      ldiv!(parent(dU[i]), n.plan.plan_x, parent(uxtmphat[i]))
   end
   return dU
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField2D, P<:NavierStokesParameters}
   E = sum( 0.5 .* real.(
      n.f.ux.^2 .+
      n.f.uy.^2 .+
      n.f.uz.^2 ) .* ( n.f.g.Δx * n.f.g.Δy * n.f.g.Δz )
   )
   if(showEnergy)
      println("E = $(E)")
   end
   return 0., E, 0., E
end

function energy(n::AbstractNumModel{F,P}, showEnergy=false) where {F<:AbstractField3D, P<:NavierStokesParameters}
   E = sum( 0.5 .* real.(
                    n.f.ux[:,:,:].^2 .+
                    n.f.uy[:,:,:].^2 .+
                    n.f.uz[:,:,:].^2 ) .* ( n.f.g.Δx * n.f.g.Δy * n.f.g.Δz )
          )
   if(showEnergy)
      println("E = $(E)")
   end
   return 0., E, 0., E
end
