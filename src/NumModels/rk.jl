mutable struct NumModelForwardEuler{F,P,Plan} <: AbstractNumModel{F,P,Plan}
   f :: F
   gf :: AbstractGradientField
   param :: P
   Δt :: Real
   niter :: Integer
   freqbckp :: Integer
   plan :: Plan
   writers :: AbstractWriterCollection{F}
end

"""
    NumModelForwardEuler(f::AbstractField, param::AbstractParameters,
          Δt::Real, niter::Integer, freqbckp::Integer)

Returns a Runge-Kutta 2 numerical model.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> param = GrossPitaevskiiParameters();
julia> nummodel = NumModelForwardEuler(field, param, 0.01, 1000, 100)
Backward Euler
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelForwardEuler(f::AbstractField, param::AbstractParameters,
      Δt::Real, niter::Integer, freqbckp::Integer)
   gf = GradientField(f,rotation=false,laplacian=false,vorticity=true)
   plan = Plan(f)
   writer = WriterVTK(f)
   saver = WriterSave(f)
   writers = WriterCollection([writer,saver])
   return NumModelForwardEuler{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
         Δt, niter, freqbckp,
         plan,
         writers
      )
end

Base.show(io::IO, n::NumModelForwardEuler) = print(io,
         "Backward Euler\n",
         "  ├───────  time step: $(n.Δt)\n",
         "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")

function timeStep!(n::NumModelForwardEuler{F}) where {F<:AbstractField3D}
   # for FFT
   ξx = n.plan.ξx
   ξy = n.plan.ξy
   ξz = n.plan.ξz
   gridξ = localgrid(n.plan.pen_z, (ξx, ξy, ξz))
   # tmp fields
   uxtmphat = n.plan.uxtmp_hat
   uytmphat = n.plan.uytmp_hat
   uytmp2hat = n.plan.uytmp2_hat
   uztmp2hat = n.plan.uztmp2_hat

   # create working fields
   u_hat = similar_data(n.plan.uz_hat)
   uk_hat = similar_data(n.plan.uz_hat)
   uk = similar_data(n.f.u)

   # coefficients
   # k1 = rhs(ut)
   # k2 = rhs(ut+Δt/2 k1)
   # k3 = rhs(ut+Δt/2 k2)
   # k4 = rhs(t+Δt, ut+Δt k3)
   # ϕt+1 = ϕt + Δt/6 (k1 + 2k2 + 2k3 + k4)

   # initialize
   # compute rhs and output in spectral
   du_hat = rhs(n, n.f.u) # k1
   for i = 1:3
      @. u_hat[i] = n.plan.uz_hat[i]
      @. u_hat[i] += n.Δt / 6. * du_hat[i] # u = u + Δt/6 (k1)
      @. uk_hat[i] = u_hat[i] + n.Δt / 2. * du_hat[i] # u + Δt/2 k1
      # back to physical
      ldiv!(parent(uztmp2hat[i]), n.plan.plan_z, parent(uk_hat[i]))
      transpose!(uytmphat[i],uztmp2hat[i])
      ldiv!(parent(uytmp2hat[i]), n.plan.plan_y, parent(uytmphat[i]))
      transpose!(uxtmphat[i],uytmp2hat[i])
      ldiv!(parent(uk[i]), n.plan.plan_x, parent(uxtmphat[i]))
   end
   du_hat = rhs(n, uk) # k2
   for i = 1:3
      @. u_hat[i] += n.Δt / 3. * du_hat[i] # u = u + Δt/6 (k1 + 2 k2)
      @. uk_hat[i] = u_hat[i] + n.Δt / 2. * du_hat[i] # u + 1/2 k2
      # back to physical
      ldiv!(parent(uztmp2hat[i]), n.plan.plan_z, parent(uk_hat[i]))
      transpose!(uytmphat[i],uztmp2hat[i])
      ldiv!(parent(uytmp2hat[i]), n.plan.plan_y, parent(uytmphat[i]))
      transpose!(uxtmphat[i],uytmp2hat[i])
      ldiv!(parent(uk[i]), n.plan.plan_x, parent(uxtmphat[i]))
   end
   du_hat = rhs(n, uk) # k3
   for i = 1:3
      @. u_hat[i] += n.Δt/3 * du_hat[i] # u = u + Δt/6 (k1 + 2 k2 + 2 k3)
      @. uk_hat[i] = u_hat[i] + n.Δt * du_hat[i] # u + k3
      # back to physical
      ldiv!(parent(uztmp2hat[i]), n.plan.plan_z, parent(uk_hat[i]))
      transpose!(uytmphat[i],uztmp2hat[i])
      ldiv!(parent(uytmp2hat[i]), n.plan.plan_y, parent(uytmphat[i]))
      transpose!(uxtmphat[i],uytmp2hat[i])
      ldiv!(parent(uk[i]), n.plan.plan_x, parent(uxtmphat[i]))
   end
   du_hat = rhs(n, uk) # k4
   for i = 1:3
      @. u_hat[i] += n.Δt/6 * du_hat[i] # u = u + Δt/6 (k1 + 2 k2 + 2 k3 + k4)
      @. u_hat[i] *= exp( - n.param.ν * n.Δt * (gridξ.x^2 + gridξ.y^2 + gridξ.z^2) )
      # back to physical
      ldiv!(parent(uztmp2hat[i]), n.plan.plan_z, parent(u_hat[i]))
      transpose!(uytmphat[i],uztmp2hat[i])
      ldiv!(parent(uytmp2hat[i]), n.plan.plan_y, parent(uytmphat[i]))
      transpose!(uxtmphat[i],uytmp2hat[i])
      ldiv!(parent(n.f.u[i]), n.plan.plan_x, parent(uxtmphat[i]))
   end
   return 1
end