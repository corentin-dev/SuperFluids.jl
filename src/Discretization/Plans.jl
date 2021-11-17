"Abstract supertype for plan type."
abstract type PlanType end
"FFT plan"
struct FFTPlan <: PlanType end
"Finite difference plan"
struct FiniteDifferentePlan <: PlanType end

"Abstract supertype for plans."
abstract type AbstractPlan{F} end
"Abstract supertype for FFT plans."
abstract type AbstractFFTPlan{F} end
"Abstract supertype for Finite Difference plans."
abstract type AbstractFDPlan{F} end

struct PlanFFT2D{F} <: AbstractFFTPlan{F}
   "plan in all directions"
   plan :: AbstractFFTs.Plan
   "plan in x direction only"
   plan_x :: AbstractFFTs.Plan
   "plan in y direction only"
   plan_y :: AbstractFFTs.Plan
   "ξx x frequencies"
   ξx :: AbstractArray
   "ξy y frequencies"
   ξy :: AbstractArray
   "temporary storage"
   ϕ_hat :: AbstractArray
end

struct PlanFFT3D{F} <: AbstractFFTPlan{F}
   "plan in all directions"
   plan :: AbstractFFTs.Plan
   "plan in x direction only"
   plan_x :: AbstractFFTs.Plan
   "plan in y direction only"
   plan_y :: AbstractFFTs.Plan
   "plan in z direction only"
   plan_z :: AbstractFFTs.Plan
   "ξx x frequencies"
   ξx :: AbstractArray
   "ξy y frequencies"
   ξy :: AbstractArray
   "ξz z frequencies"
   ξz :: AbstractArray
   "temporary storage"
   ϕ_hat :: AbstractArray
end

function ξsquared(ξx::Real, ξy::Real, ξz::Real)
   a = ξx^2 + ξy^2 + ξz^2
   if abs(a) < 1e-8
      return 1
   else
      return a
   end
end

struct PlanFD2D{F} <: AbstractFDPlan{F}
   Δx :: Real
   Δy :: Real
   Δz :: Real
end

struct PlanFD3D{F} <: AbstractFDPlan{F}
   Δx :: Real
   Δy :: Real
   Δz :: Real
end

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D}
   if typeof(t) == FFTPlan
      ξx = fftfreq(f.g.nx,2π/f.g.Δx)
      ξy = fftfreq(f.g.ny,2π/f.g.Δy)
      if typeof(f.ϕ) <: Array
         myArray = Array
      elseif typeof(f.ϕ) <: CuArray
         myArray = CuArray
      end
      ϕ_hat = similar(f.ϕ)
      Ξx = reshape(myArray(ξx),f.g.nx,1)
      Ξy = reshape(myArray(ξy),1,f.g.ny)
      return PlanFFT2D{F}(plan_fft(f.ϕ,(1,2)),
                     plan_fft(f.ϕ,1),
                     plan_fft(f.ϕ,2),
                     Ξx, Ξy,
                     ϕ_hat)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy)
   end
end

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D}
   if typeof(t) == FFTPlan
      ξx = fftfreq(f.g.nx,2π/f.g.Δx)
      ξy = fftfreq(f.g.ny,2π/f.g.Δy)
      ξz = fftfreq(f.g.nz,2π/f.g.Δz)
      if typeof(f.ϕ) <: Array
         myArray = Array
      elseif typeof(f.ϕ) <: CuArray
         myArray = CuArray
      end
      Ξx = reshape(myArray(ξx),f.g.nx,1,1)
      Ξy = reshape(myArray(ξy),1,f.g.ny,1)
      Ξz = reshape(myArray(ξz),1,1,f.g.nz)
      return PlanFFT3D{F}(plan_fft(f.ϕ,(1,2,3)),
                     plan_fft(f.ϕ,1),
                     plan_fft(f.ϕ,2),
                     plan_fft(f.ϕ,3),
                     Ξx, Ξy, Ξz)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy, f.g.Δz)
   end
end