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
   "x for plan"
   x :: AbstractArray
   "y for plan"
   y :: AbstractArray
   "z for plan"
   z :: AbstractArray
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
      ξx = fftfreq(f.g.nx, 2π/f.g.Δx)
      ξy = fftfreq(f.g.ny, 2π/f.g.Δy)
      ξz = fftfreq(f.g.nz, 2π/f.g.Δz)

      plan = PencilFFTPlan(f.decomp.pen_array, Transforms.FFT())
      plan_x = PencilFFTPlan(f.decomp.pen_array, (Transforms.FFT(),Transforms.NoTransform(),Transforms.NoTransform()))
      plan_y = PencilFFTPlan(f.decomp.pen_array, (Transforms.NoTransform(),Transforms.FFT(),Transforms.NoTransform()))
      plan_z = PencilFFTPlan(f.decomp.pen_array, (Transforms.NoTransform(),Transforms.NoTransform(),Transforms.FFT()))

      # temporary arrays
      ϕ_hat = allocate_output(plan)

      # axes
      ϕ_hat_glob = global_view(ϕ_hat)
      # ranges
      rx_hat, ry_hat, rz_hat = axes(ϕ_hat_glob)
      nxl_hat, nyl_hat, nzl_hat = size_local(ϕ_hat)

      # range compatible to all types of Array
      rrx_hat = rx_hat[begin]:rx_hat[end]
      rry_hat = ry_hat[begin]:ry_hat[end]
      rrz_hat = rz_hat[begin]:rz_hat[end]
      # reshape hat versions of positions
      x_hat = reshape(f.g.x[rrx_hat],1,1,nxl_hat)
      y_hat = reshape(f.g.y[rry_hat],1,nyl_hat,1)
      z_hat = reshape(f.g.z[rrz_hat],nzl_hat,1,1)
      # reshape versions of ξ
      Ξx = similar(x_hat)
      Ξy = similar(y_hat)
      Ξz = similar(z_hat)
      copyto!(Ξx, reshape(ξx[rrx_hat],1,1,nxl_hat))
      copyto!(Ξy, reshape(ξy[rry_hat],1,nyl_hat,1))
      copyto!(Ξz, reshape(ξz[rrz_hat],nzl_hat,1,1))

      return PlanFFT3D{F}(plan,
                     plan_x, plan_y, plan_z,
                     Ξx, Ξy, Ξz,
                     x_hat, y_hat, z_hat,
                     ϕ_hat)
   else
      return PlanFD2D{F}(f.g.Δx, f.g.Δy, f.g.Δz)
   end
end