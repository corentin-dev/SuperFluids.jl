"Abstract supertype for parameters."
abstract type AbstractParameters end

"Abstract supertype for numerical models."
abstract type AbstractNumModel{AbstractField,AbstractParameters,AbstractPlan} end

include("GrossPitaevskii/GrossPitaevskii.jl")

export NumModelADI1, NumModelADI2
export NumModelBackwardEuler, NumModelBackwardEulerNoPrecond, NumModelBackwardEulerNL
export NumModelCrankNicolson, NumModelCrankNicolsonQuasiNewton, NumModelCrankNicolsonT, NumModelCrankNicolsonQuasiNewtonT
export NumModelExternalVelocity
export solve!

include("adi.jl")
include("backward-euler.jl")
include("crank-nicolson.jl")
# include("external-velocity.jl")

include("krylov.jl")

function solve!(n::AbstractNumModel;istart=1,plot=false)
   if plot
      @eval using SuperFluids.Plots
      p = Plot(n.f)
      createPlot!(p)
   end
   write!(n.writers,prefix="res",icpu=0,istep=istart,Δt=n.Δt)
   for it = istart:istart+n.niter
      println("iteration $(it)")
      energy(n,true)
      timeStep!(n)
      if it % n.freqbckp == 0
         write!(n.writers,prefix="res",icpu=0,istep=it,Δt=n.Δt)
      end
      if plot
         updatePlot!(p)
      end
   end
   energy(n,true)
   if plot
      return p
   else
      return nothing
   end
end
