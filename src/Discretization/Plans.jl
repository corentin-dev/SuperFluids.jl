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
   "field"
   f :: AbstractField2D
   pen_x
   pen_y
   "plan in x direction only"
   plan_x :: AbstractFFTs.Plan
   "plan in y direction only"
   plan_y :: AbstractFFTs.Plan
   "ξx x frequencies"
   ξx :: AbstractArray
   "ξy y frequencies"
   ξy :: AbstractArray
   "temporary storage x"
   ϕx_hat :: AbstractArray
   "temporary storage y"
   ϕy_hat :: AbstractArray
   "temporary storage x"
   ϕxtmp_hat :: AbstractArray
   "temporary storage y"
   ϕytmp_hat :: AbstractArray
end

struct PlanFFT3D{F} <: AbstractFFTPlan{F}
   "field"
   f :: AbstractField3D
   pen_x
   pen_y
   pen_z
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
   "temporary storage x"
   ϕx_hat :: AbstractArray
   "temporary storage y"
   ϕy_hat :: AbstractArray
   "temporary storage z"
   ϕz_hat :: AbstractArray
   "temporary storage x"
   ϕxtmp_hat :: AbstractArray
   "temporary storage y"
   ϕytmp_hat :: AbstractArray
   "temporary storage z"
   ϕztmp_hat :: AbstractArray
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
end

struct PlanFD3D{F} <: AbstractFDPlan{F}
   Δx :: Real
   Δy :: Real
   Δz :: Real
end

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D{FT,FFT,A}} where {FT,FFT,A}
   if typeof(t) == FFTPlan
      # frequencies
      ξx = A(fftfreq(f.g.nx, 2π/f.g.Δx))
      ξy = A(fftfreq(f.g.ny, 2π/f.g.Δy))

      # Pencil decompositions
      pen_x = f.decomp.pen_array.pencil
      pen_y = Pencil(pen_x, decomp_dims=(1,), permute = Permutation(2, 1) )

      # create arrays for FFT
      ϕx_hat = PencilArray{FFT}(undef, pen_x)
      ϕy_hat = PencilArray{FFT}(undef, pen_y)

      # temporary arrays
      ϕxtmp_hat = PencilArray{FFT}(undef, pen_x)
      ϕytmp_hat = PencilArray{FFT}(undef, pen_y)

      # create plans
      plan_x = plan_fft(parent(ϕx_hat),1)
      plan_y = plan_fft(parent(ϕy_hat),1)

      return PlanFFT2D{FT}(
                     f,
                     pen_x, pen_y,
                     plan_x, plan_y,
                     ξx, ξy,
                     ϕx_hat, ϕy_hat,
                     ϕxtmp_hat, ϕytmp_hat)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy)
   end
end

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D{FT,FFT,A}} where {FT,FFT,A}
   if typeof(t) == FFTPlan
      # frequencies
      ξx = A(fftfreq(f.g.nx, 2π/f.g.Δx))
      ξy = A(fftfreq(f.g.ny, 2π/f.g.Δy))
      ξz = A(fftfreq(f.g.nz, 2π/f.g.Δz))

      # Pencil decompositions
      pen_x = f.decomp.pen_array.pencil
      pen_y = Pencil(pen_x, decomp_dims=(1, 3), permute = Permutation(2, 1, 3) )
      pen_z = Pencil(pen_x, decomp_dims=(1, 2), permute = Permutation(3, 1, 2) )

      # create arrays for FFT
      ϕx_hat = PencilArray{FFT}(undef, pen_x)
      ϕy_hat = PencilArray{FFT}(undef, pen_y)
      ϕz_hat = PencilArray{FFT}(undef, pen_z)

      # temporary arrays
      ϕxtmp_hat = PencilArray{FFT}(undef, pen_x)
      ϕytmp_hat = PencilArray{FFT}(undef, pen_y)
      ϕztmp_hat = PencilArray{FFT}(undef, pen_z)

      # create plans
      plan_x = plan_fft(parent(ϕx_hat),1)
      plan_y = plan_fft(parent(ϕy_hat),1)
      plan_z = plan_fft(parent(ϕz_hat),1)

      return PlanFFT3D{FT}(
                     f,
                     pen_x, pen_y, pen_z,
                     plan_x, plan_y, plan_z,
                     ξx, ξy, ξz,
                     ϕx_hat, ϕy_hat, ϕz_hat,
                     ϕxtmp_hat, ϕytmp_hat, ϕztmp_hat)
   else
      return PlanFD3D{FT}(f.g.Δx, f.g.Δy, f.g.Δz)
   end
end