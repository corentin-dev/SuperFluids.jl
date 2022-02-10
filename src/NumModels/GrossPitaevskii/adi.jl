mutable struct NumModelADI1{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

function NumModelADI1(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelADI1{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
         writers
      )
end

Base.show(io::IO, n::NumModelADI1) =
   print(io,
         "Splitting Order 1\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI1)
   solveLapRot!(n,n.Δt)
   solveNL!(n,n.Δt)
   return 1
end

mutable struct NumModelADI2{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: Any
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

function NumModelADI2(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   gf = GradientField(f,rotation=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelADI2{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
         writers
      )
end

Base.show(io::IO, n::NumModelADI2) =
   print(io,
         "Splitting Order 2\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelADI2)
   solveLapRot!(n,n.Δt*0.5)
   solveNL!(n,n.Δt)
   solveLapRot!(n,n.Δt*0.5)
   return 2
end

function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField2D}
   # field
   ϕ = n.f.ϕ
   # parameters
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # temporary fields
   ϕxthat = n.plan.ϕx_hat
   ϕythat = n.plan.ϕy_hat
   ϕxtmphat = n.plan.ϕxtmp_hat
   ϕytmphat = n.plan.ϕytmp_hat
   # plans
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   grid = localgrid(n.plan.pen_x, (n.f.g.x, n.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(n.plan.pen_x, (n.plan.ξx, n.plan.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   mul!(parent(ϕxthat), plan_x, parent(ϕ))
   @. ϕxthat = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕxthat
   ldiv!(parent(ϕ), plan_x, parent(ϕxthat))
   # FFT y
   grid = localgrid(n.plan.pen_y, (n.f.g.x, n.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(n.plan.pen_y, (n.plan.ξx, n.plan.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   transpose!(ϕytmphat, ϕ)
   mul!(parent(ϕythat), plan_y, parent(ϕytmphat))
   @. ϕythat = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕythat
   ldiv!(parent(ϕytmphat), plan_y, parent(ϕythat))
   transpose!(ϕ, ϕytmphat)
   return nothing
 end

 function solveLapRot!(n::AbstractNumModel{F,P}, Δtl) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters}
   # field
   ϕ = n.f.ϕ
   # parameters
   coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
   # temporary fields
   ϕxthat = n.plan.ϕx_hat
   ϕythat = n.plan.ϕy_hat
   ϕzthat = n.plan.ϕz_hat
   ϕxtmphat = n.plan.ϕxtmp_hat
   ϕytmphat = n.plan.ϕytmp_hat
   ϕztmphat = n.plan.ϕztmp_hat
   # plans
   plan_x, plan_y, plan_z = n.plan.plan_x, n.plan.plan_y, n.plan.plan_z
   # FFT x
   grid = localgrid(n.plan.pen_x, (n.f.g.x, n.f.g.y, n.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(n.plan.pen_x, (n.plan.ξx, n.plan.ξy, n.plan.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   mul!(parent(ϕxthat), plan_x, parent(ϕ))
   @. ϕxthat = exp(im*(coeffΔ*ξx^2-Ω*y*ξx)*Δtl) * ϕxthat
   ldiv!(parent(ϕ), plan_x, parent(ϕxthat))
   # FFT y
   grid = localgrid(n.plan.pen_y, (n.f.g.x, n.f.g.y, n.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(n.plan.pen_y, (n.plan.ξx, n.plan.ξy, n.plan.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   transpose!(ϕytmphat, ϕ)
   mul!(parent(ϕythat), plan_y, parent(ϕytmphat))
   @. ϕythat = exp(im*(coeffΔ*ξy^2+Ω*x*ξy)*Δtl) * ϕythat
   ldiv!(parent(ϕytmphat), plan_y, parent(ϕythat))
   transpose!(ϕ, ϕytmphat)
   # FFT z
   grid = localgrid(n.plan.pen_z, (n.f.g.x, n.f.g.y, n.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(n.plan.pen_z, (n.plan.ξx, n.plan.ξy, n.plan.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   transpose!(ϕytmphat, ϕ)
   transpose!(ϕztmphat, ϕytmphat)
   mul!(parent(ϕzthat), plan_z, parent(ϕztmphat))
   @. ϕzthat = exp(im*(coeffΔ*ξz^2)*Δtl) * ϕzthat
   ldiv!(parent(ϕztmphat), plan_z, parent(ϕzthat))
   transpose!(ϕztmphat, ϕytmphat)
   transpose!(ϕ, ϕytmphat)
   return nothing
 end

 function solveNL!(n::AbstractNumModel{F,P}, Δtl) where {F<:AbstractField, P<:GrossPitaevskiiParameters}
    # ϕ ↦ exp ( -i ( V + ∥ϕ∥² ) Δt ) ϕ
    @.n.f.ϕ = exp(-1im * ( n.param.pot.V + n.param.β * abs2(n.f.ϕ) ) * Δtl) * n.f.ϕ
 end