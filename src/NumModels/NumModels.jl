"Abstract supertype for numerical models."
abstract type AbstractNumModel{AbstractField,AbstractParameters,AbstractPlan} end

include("GrossPitaevskii/GrossPitaevskii.jl")
include("NavierStokes/NavierStokes.jl")

export NumModelADI1, NumModelADI2
export NumModelForwardEuler
export NumModelBackwardEuler, NumModelBackwardEulerNoPrecond, NumModelBackwardEulerNL
export NumModelCrankNicolson, NumModelCrankNicolsonQuasiNewton, NumModelCrankNicolsonT, NumModelCrankNicolsonQuasiNewtonT
export NumModelExternalVelocity
export solve!

include("rk.jl")
# include("external-velocity.jl")

include("krylov.jl")

include("derivatives.jl")

function solve!(n::AbstractNumModel;istart=1,plot=false)
   if plot
      @eval using SuperFluids.Plots
      p = Plot(n.f)
      createPlot!(p)
   end
   res = Tuple{Int64, Int64, Float64, Float64, Float64, Float64}[]
   nkrylovtotal = 0
   write!(n.writers,prefix="res",icpu=0,istep=istart,Δt=n.Δt)
   for it = istart:istart+n.niter
      println("iteration $(it)")
      E = energy(n,true)
      nkrylov = timeStep!(n)
      nkrylovtotal += nkrylov
      push!(res, (nkrylov,nkrylovtotal,E...) )
      if it % n.freqbckp == 0
         write!(n.writers,prefix="res",icpu=0,istep=it,Δt=n.Δt)
      end
      if plot
         updatePlot!(p)
      end
   end
   return res
end