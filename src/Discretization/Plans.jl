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
   "x for plan"
   x :: AbstractArray
   "y for plan"
   y :: AbstractArray
   "temporary storage"
   ϕ_hat :: AbstractArray
end

struct PlanFFT3D{F} <: AbstractFFTPlan{F}
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
   "x for plan"
   x :: AbstractArray
   "y for plan"
   y :: AbstractArray
   "temporary storage x"
   ϕx_hat :: AbstractArray
   "temporary storage y"
   ϕy_hat :: AbstractArray
   "temporary storage z"
   ϕz_hat :: AbstractArray
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

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D}
   if typeof(t) == FFTPlan
      ξx = similar(f.x)
      ξy = similar(f.y)
      ξx .= fftfreq(f.g.nx,2π/f.g.Δx)
      ξy .= fftfreq(f.g.ny,2π/f.g.Δy)
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
                     f.g.x, f.g.y,
                     ϕ_hat)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy)
   end
end

function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D}
   if typeof(t) == FFTPlan
      # get array type
      AT = get_array_type(f.ϕ)
      # get type
      T = get_type_array(f.ϕ)

      # frequencies
      ξx = AT(reshape(fftfreq(f.g.nx, 2π/f.g.Δx),f.g.nx,1,1))
      ξy = AT(reshape(fftfreq(f.g.ny, 2π/f.g.Δy),f.g.ny,1,1))
      ξz = AT(reshape(fftfreq(f.g.nz, 2π/f.g.Δz),f.g.nz,1,1))

      # Pencil decompositions
      pen_x = f.decomp.pen_array.pencil
      pen_y = Pencil(pen_x, decomp_dims=(1, 3), permute = Permutation(2, 1, 3) )
      pen_z = Pencil(pen_x, decomp_dims=(2, 1), permute = Permutation(3, 1, 2) )

      # create temporary arrays
      ϕx_hat = PencilArray{T}(undef, pen_x);
      ϕy_hat = PencilArray{T}(undef, pen_y);
      ϕz_hat = PencilArray{T}(undef, pen_z);

      # create plans
      plan_x = plan_fft(parent(ϕx_hat),1)
      plan_y = plan_fft(parent(ϕy_hat),1)
      plan_z = plan_fft(parent(ϕz_hat),1)

      # axes
      println("range local $(range_local(ϕx_hat)) $(range_local(ϕy_hat)) $(size_local(ϕx_hat)) $(size_local(ϕy_hat))")

      x_hat = AT(reshape(f.g.x[range_local(ϕy_hat)[1]],1,size_local(ϕy_hat)[1],1))
      y_hat = AT(reshape(f.g.y[range_local(ϕx_hat)[2]],1,size_local(ϕx_hat)[2],1))

      return PlanFFT3D{F}(plan_x, plan_y, plan_z,
                     ξx, ξy, ξz,
                     x_hat, y_hat,
                     ϕx_hat, ϕy_hat, ϕz_hat)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy, f.g.Δz)
   end
end