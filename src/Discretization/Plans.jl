"Abstract supertype for plan type."
abstract type PlanType end
"FFT plan"
struct FFTPlan <: PlanType end
"Finite difference plan"
struct FiniteDifferencePlan <: PlanType end

export Plan, FFTPlan, FiniteDifferencePlan

"Abstract supertype for plans."
abstract type AbstractPlan{F} end
"Abstract supertype for FFT plans."
abstract type AbstractFFTPlan{F} end
"Abstract supertype for Finite Difference plans."
abstract type AbstractFDPlan{F} end

"""
    PlanFFT2D{F} <: AbstractFFTPlan{F}

Type representing a 2D FFT plan.

It contains the following informations:

- `f`: reference to a field
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `plan_x`: FFT plan in the ``x`` direction.
- `plan_y`: FFT plan in the ``y`` direction.
- `ξx`, `ξy`, `ξz`: frequency (non distributed) along each directions.
- `ϕx_hat`, `ϕy_hat`, `ϕz_hat`: fields to contain transformed fields distributed along each directions.
- `ϕxtmp_hat`, `ϕytmp_hat`, `ϕztmp_hat`: temporary fields distributed along each directions.
"""
struct PlanFFT2D{F} <: AbstractFFTPlan{F}
   f :: AbstractField2D
   pen_x
   pen_y
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
   ξx :: AbstractArray
   ξy :: AbstractArray
   ϕx_hat :: AbstractArray
   ϕy_hat :: AbstractArray
   ϕxtmp_hat :: AbstractArray
   ϕytmp_hat :: AbstractArray
end

"""
    PlanFFT3D{F} <: AbstractFFTPlan{F}

Type representing a 3D FFT plan.

It contains the following informations:

- `f`: reference to a field
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `pen_z`: pencil (or slab) in the ``z`` direction.
- `plan_x`: FFT plan in the ``x`` direction.
- `plan_y`: FFT plan in the ``y`` direction.
- `plan_z`: FFT plan in the ``z`` direction.
- `ξx`, `ξy`, `ξz`: frequency (non distributed) along each directions.
- `ϕx_hat`, `ϕy_hat`, `ϕz_hat`: fields to contain transformed fields distributed along each directions.
- `ϕxtmp_hat`, `ϕytmp_hat`, `ϕztmp_hat`: temporary fields distributed along each directions.
"""
struct PlanFFT3D{F} <: AbstractFFTPlan{F}
   f :: AbstractField3D
   pen_x
   pen_y
   pen_z
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
   plan_z :: AbstractFFTs.Plan
   ξx :: AbstractArray
   ξy :: AbstractArray
   ξz :: AbstractArray
   ϕx_hat :: AbstractArray
   ϕy_hat :: AbstractArray
   ϕz_hat :: AbstractArray
   ϕxtmp_hat :: AbstractArray
   ϕytmp_hat :: AbstractArray
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

"""
   PlanFD2D{F} <: AbstractFFTPlan{F}

Type representing a 2D finite difference plan.

It contains the following informations:

- `f`: reference to a field.
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `Δx`, `Δy`: discretization step.
- `ϕxtmp`, `ϕytmp`: temporary fields distributed along each directions.
"""
struct PlanFD2D{F} <: AbstractFDPlan{F}
   f :: AbstractField2D
   pen_x
   pen_y
   Δx :: Real
   Δy :: Real
   ϕxtmp :: AbstractArray
   ϕytmp :: AbstractArray
end

"""
   PlanFD3D{F} <: AbstractFFTPlan{F}

Type representing a 3D finite difference plan.

It contains the following informations:

- `f`: reference to a field.
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `pen_z`: pencil (or slab) in the ``z`` direction.
- `Δx`, `Δy`, `Δz`: discretization step.
- `ϕxtmp`, `ϕytmp`, `ϕztmp`: temporary fields distributed along each directions.

"""
struct PlanFD3D{F} <: AbstractFDPlan{F}
   f :: AbstractField3D
   pen_x
   pen_y
   pen_z
   Δx :: Real
   Δy :: Real
   Δz :: Real
   ϕxtmp :: AbstractArray
   ϕytmp :: AbstractArray
   ϕztmp :: AbstractArray
end

"""
    Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D{FT,FFT,A}} where {FT,FFT,A}

Returns a 2D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()` or `FiniteDifferencePlan()`)

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FFTPlan());
```

```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FiniteDifferencePlan());
```
"""
function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D{FT,FFT,A}} where {FT,FFT,A}
   # Pencil decompositions
   pen_x = f.pen
   pen_y = Pencil(pen_x, decomp_dims=(1,), permute = Permutation(2, 1) )
   if typeof(t) == FFTPlan
      # frequencies
      ξx = A(fftfreq(f.g.nx, 2π/f.g.Δx))
      ξy = A(fftfreq(f.g.ny, 2π/f.g.Δy))

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
      # create arrays for FFT
      ϕxtmp = PencilArray{FFT}(undef, pen_x)
      ϕytmp = PencilArray{FFT}(undef, pen_y)
      return PlanFD2D{F}(f, pen_x, pen_y, f.g.Δx, f.g.Δy, ϕxtmp, ϕytmp)
   end
end

"""
    Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D{FT,FFT,A}} where {FT,FFT,A}

Returns a 3D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()` or `FiniteDifferencePlan()`)

Example
=======
```jldoctest
julia> grid = Grid((128,128,128), ((-12,12),(-12,12), (-12,12)));
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FFTPlan());
```

```jldoctest
julia> grid = Grid((128,128,128), ((-12,12),(-12,12), (-12,12)));
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FiniteDifferencePlan());
```
"""
function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D{FT,FFT,A}} where {FT,FFT,A}
   # Pencil decompositions
   pen_x = f.pen
   pen_y = Pencil(pen_x, decomp_dims=(1, 3), permute = Permutation(2, 1, 3) )
   pen_z = Pencil(pen_x, decomp_dims=(1, 2), permute = Permutation(3, 1, 2) )
   if typeof(t) == FFTPlan
      # frequencies
      ξx = A(fftfreq(f.g.nx, 2π/f.g.Δx))
      ξy = A(fftfreq(f.g.ny, 2π/f.g.Δy))
      ξz = A(fftfreq(f.g.nz, 2π/f.g.Δz))

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
      # create arrays for FFT
      ϕxtmp = PencilArray{FFT}(undef, pen_x)
      ϕytmp = PencilArray{FFT}(undef, pen_y)
      ϕztmp = PencilArray{FFT}(undef, pen_y)
      return PlanFD3D{F}(f, pen_x, pen_y, pen_z, f.g.Δx, f.g.Δy, f.g.Δz, ϕxtmp, ϕytmp, ϕztmp)
   end
end