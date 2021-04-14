

mutable struct NumModelExternalVelocity{F,P,W} <: AbstractNumModel{F,P,W}
   f :: F
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   coeffΔ :: Real
   β :: Real
   Ω :: Real
   plan :: AbstractPlan{F}
   ϕ_hat :: Array
   potential :: P
   writer :: W
end

Base.show(io::IO, n::NumModelExternalVelocity) = print(io,
         "ARGLE scheme\n",
         "  ├───────────  model: coeff Δ : $(n.coeffΔ) β : $(n.β), Ω : $(n.Ω)", '\n', 
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

"""
    timeStep!(n::NumModelExternalVelocity)

Performs a single time step for stationnary field approximating a velocity field.
"""
# psi_1(:,:,:,:) = czero
# call model_EI(1.d0, 0.d0)
# call GPS_copy_data(phi_tilde(:,:,:,1),phi_0(:,:,:,1))
#SUBROUTINE model_EI(gammark, rhork)
#  al = abs(coeff_Delta)
#  be = beta
#  ga = beta
#  alphark = gammark + rhork
#  CALL comput_lap_rot(4,phi_0(:,:,:,1),rMb)
#  do
#           psin = phi_0(i,j,k,1)
#           modphi = real(psin*conjg(psin))
#           modu = (u_adv(i,j,k,1)**2+u_adv(i,j,k,2)**2+u_adv(i,j,k,3)**2)
#           fci(i,j,k) = psin + gammark*delta_t*( -be*modphi*psin + ga*psin -modu/(4.d0*al)*psin - &
#                uim*u_adv(i,j,k,1)*gradientx(i,j,k) -uim* u_adv(i,j,k,2)*gradienty(i,j,k))
#           if (nzb>1) then
#              fci(i,j,k) = fci(i,j,k) - uim* u_adv(i,j,k,3)*gradientz(i,j,k)*gammark*delta_t
#           end if
#  CALL comput_lap_rot(4,psi_1(:,:,:,1),rMb)
#  call fl_fft_all(fci,fci,1)
#  call fl_fft_all(phi_0(:,:,:,1),phi_tilde(:,:,:,1),1)
#  do
#           wvn = Lapxfl(i) + Lapyfl(j)
#           if (nzb>1) wvn = wvn+Lapzfl(k)
          
#           fci(i,j,k) = ( fci(i,j,k) - alphark*delta_t*al*wvn*phi_tilde(i,j,k,1)/2.d0 )/(1.d0+alphark*delta_t*al*wvn/2.d0)
#  call fl_fft_all(fci,fci,2)
#  phi_tilde(:,:,:,1) = fci
function timeStep!(n::NumModelExternalVelocity{F}) where {F<:AbstractField2D}
   # references
   ϕ, ϕhat_x, ϕhat_y, ϕhat = n.f.ϕ, n.ϕ_hat, n.ϕ_hat, n.ϕ_hat
   Δt, coeffΔ, Ω, β = n.Δt, n.coeffΔ, n.Ω, n.β
   x, y = n.f.g.x, n.f.g.y
   ξx, ξy = n.f.g.ξx, n.f.g.ξy
   plan, plan_x, plan_y = n.plan.plan, n.plan.plan_x, n.plan.plan_y
   Δx, Δy = n.f.g.Δx, n.f.g.Δy
   V, uadvx, uadvy = n.potential.V, n.potential.uadvx, n.potential.uadvy
   α = 1.
   γ = 1.
   # create working vectors
   ψ₁ = copy(ϕ)
   ψ₁ .= 0.
   ϕw = similar(ϕ)

   # compute gradients
   mul!(ϕhat_x, plan_x, ϕ)
   ∇ϕ_x = plan_x \ (im .* ξx .* ϕhat_x) # we get grad x
   mul!(ϕhat_y, plan_y, ϕ)
   ∇ϕ_y = plan_y \ (im .* ξy .* ϕhat_y) # we get grad y
   # compute ψ₁
   @. ψ₁ = ϕ + γ * Δt * (
                - β * real(ϕ * conj(ϕ)) * ϕ
                + β * ϕ
                -  (uadvx^2+uadvy^2)/(-4*coeffΔ) * ϕ
                - im * uadvx * ∇ϕ_x - im * uadvy * ∇ϕ_y
               )
   # all to frequency domain
   ψ₁hat = plan * ψ₁
   mul!(ϕhat, plan, ϕ)
   @. ψ₁hat = ( ψ₁hat + α * Δt * coeffΔ * (ξx^2+ξy^2) * ϕhat / 2 )/(
            1 - α * Δt * coeffΔ * (ξx^2+ξy^2) / 2 )
   ldiv!(ϕ, plan, ψ₁hat)
end
