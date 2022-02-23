"Abstract supertype for plan type."
abstract type PlanType end
"FFT plan"
struct FFTPlan <: PlanType end
"Finite difference plan"
struct FiniteDifferencePlan <: PlanType end

export Plan, FFTPlan, FiniteDifferencePlan

"Abstract supertype for plans."
abstract type AbstractPlan{N} end
"Abstract supertype for FFT plans."
abstract type AbstractFFTPlan{N} end
"Abstract supertype for Finite Difference plans."
abstract type AbstractFDPlan{N} end

"""
    PlanFFT2D{N} <: AbstractFFTPlan{N}

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
struct PlanFFT2D{N} <: AbstractFFTPlan{N}
   f :: AbstractField2D
   pen_x
   pen_y
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
   ξx :: AbstractArray
   ξy :: AbstractArray
   datax :: Vector{AbstractArray}
   datay :: Vector{AbstractArray}
end

"""
    PlanFFT3D{N} <: AbstractFFTPlan{N}

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
struct PlanFFT3D{N} <: AbstractFFTPlan{N}
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
   datax :: Vector{AbstractArray}
   datay :: Vector{AbstractArray}
   dataz :: Vector{AbstractArray}
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
   PlanFD2D{N} <: AbstractFFTPlan{N}

Type representing a 2D finite difference plan.

It contains the following informations:

- `f`: reference to a field.
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `Δx`, `Δy`: discretization step.
- `ϕxtmp`, `ϕytmp`: temporary fields distributed along each directions.
"""
struct PlanFD2D{N} <: AbstractFDPlan{N}
   f :: AbstractField2D
   pen_x
   pen_y
   Δx :: Real
   Δy :: Real
   ϕxtmp :: AbstractArray
   ϕytmp :: AbstractArray
end

"""
   PlanFD3D{N} <: AbstractFFTPlan{N}

Type representing a 3D finite difference plan.

It contains the following informations:

- `f`: reference to a field.
- `pen_x`: pencil (or slab) in the ``x`` direction.
- `pen_y`: pencil (or slab) in the ``y`` direction.
- `pen_z`: pencil (or slab) in the ``z`` direction.
- `Δx`, `Δy`, `Δz`: discretization step.
- `ϕxtmp`, `ϕytmp`, `ϕztmp`: temporary fields distributed along each directions.

"""
struct PlanFD3D{N} <: AbstractFDPlan{N}
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
    Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D{N,FT,FFT,A}} where {N,FT,FFT,A}

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
function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField2D{N,FT,FFT,A}} where {N,FT,FFT,A}
   # Pencil decompositions
   pen_x = f.pen
   pen_y = Pencil(pen_x, decomp_dims=(1,), permute = Permutation(2, 1) )
   if typeof(t) == FFTPlan
      # frequencies
      ξx = A(fftfreq(f.g.nx, 2π/f.g.Δx))
      ξy = A(fftfreq(f.g.ny, 2π/f.g.Δy))

      # create arrays for FFT
      datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x)]
      datay = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x)]
      for i = 1:N-1
         push!(datax, PencilArray{FFT}(undef, pen_x))
         push!(datax, PencilArray{FFT}(undef, pen_x))
         push!(datay, PencilArray{FFT}(undef, pen_y))
         push!(datay, PencilArray{FFT}(undef, pen_y))
      end

      # create plans
      plan_x = plan_fft(parent(datax[1]),1)
      plan_y = plan_fft(parent(datay[1]),1)

      return PlanFFT2D{N}(
                     f,
                     pen_x, pen_y,
                     plan_x, plan_y,
                     ξx, ξy,
                     datax, datay)
   else
      # create arrays for Finite Difference
      ϕxtmp = PencilArray{FFT}(undef, pen_x)
      ϕytmp = PencilArray{FFT}(undef, pen_y)
      return PlanFD2D{N}(f, pen_x, pen_y, f.g.Δx, f.g.Δy, ϕxtmp, ϕytmp)
   end
end

"""
    Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D{N,FT,FFT,A}} where {N,FT,FFT,A}

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
function Plan(f::F; t::PlanType = FFTPlan()) where {F<:AbstractField3D{N,FT,FFT,A}} where {N,FT,FFT,A}
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
      datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x)]
      datay = [PencilArray{FFT}(undef, pen_y), PencilArray{FFT}(undef, pen_y)]
      dataz = [PencilArray{FFT}(undef, pen_z), PencilArray{FFT}(undef, pen_z)]
      for i = 1:N-1
         push!(datax, PencilArray{FFT}(undef, pen_x))
         push!(datax, PencilArray{FFT}(undef, pen_x))
         push!(datay, PencilArray{FFT}(undef, pen_y))
         push!(datay, PencilArray{FFT}(undef, pen_y))
         push!(dataz, PencilArray{FFT}(undef, pen_z))
         push!(dataz, PencilArray{FFT}(undef, pen_z))
      end

      # create plans
      plan_x = plan_fft(parent(datax[1]),1)
      plan_y = plan_fft(parent(datay[1]),1)
      plan_z = plan_fft(parent(dataz[1]),1)

      return PlanFFT3D{N}(
                     f,
                     pen_x, pen_y, pen_z,
                     plan_x, plan_y, plan_z,
                     ξx, ξy, ξz,
                     datax, datay, dataz)
   else
      # create arrays for FFT
      ϕxtmp = PencilArray{FFT}(undef, pen_x)
      ϕytmp = PencilArray{FFT}(undef, pen_y)
      ϕztmp = PencilArray{FFT}(undef, pen_y)
      return PlanFD3D{N}(f, pen_x, pen_y, pen_z, f.g.Δx, f.g.Δy, f.g.Δz, ϕxtmp, ϕytmp, ϕztmp)
   end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{1}, name::Symbol)
   if name === :ϕx_hat
      return plan.datax[1]
   elseif name === :ϕy_hat
      return plan.datay[1]
   elseif name === :ϕz_hat
      return plan.dataz[1]
   elseif name === :ϕxtmp_hat
      return plan.datax[2]
   elseif name === :ϕytmp_hat
      return plan.datay[2]
   elseif name === :ϕztmp_hat
      return plan.dataz[2]
   else
      return getfield(plan, name)
   end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{N}, name::Symbol) where N
   if name === :ϕx_hat
      return plan.datax[1:N]
   elseif name === :ϕy_hat
      return plan.datay[1:N]
   elseif name === :ϕz_hat
      return plan.dataz[1:N]
   elseif name === :ϕxtmp_hat
      return plan.datax[N+1:2*N]
   elseif name === :ϕytmp_hat
      return plan.datay[N+1:2*N]
   elseif name === :ϕztmp_hat
      return plan.dataz[N+1:2*N]
   else
      return getfield(plan, name)
   end
end