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
$(TYPEDEF)

Type representing a 2D FFT plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFFT2D{N} <: AbstractFFTPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "FFT plan in the ``x`` direction."
    plan_x::AbstractFFTs.Plan
    "FFT plan in the ``y`` direction."
    plan_y::AbstractFFTs.Plan
    "``x`` frequency (non distributed)."
    ξx::AbstractArray
    "``y`` frequency (non distributed)."
    ξy::AbstractArray
    "fields to contain transformed fields distributed along ``x``."
    datax::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``y``."
    datay::Vector{AbstractArray}
end

"""
$(TYPEDEF)

Type representing a 3D FFT plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFFT3D{N} <: AbstractFFTPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "FFT plan in the ``x`` direction."
    plan_x::AbstractFFTs.Plan
    "FFT plan in the ``y`` direction."
    plan_y::AbstractFFTs.Plan
    "FFT plan in the ``z`` direction."
    plan_z::AbstractFFTs.Plan
    "``x`` frequency (non distributed)."
    ξx::AbstractArray
    "``y`` frequency (non distributed)."
    ξy::AbstractArray
    "``z`` frequency (non distributed)."
    ξz::AbstractArray
    "fields to contain transformed fields distributed along ``x``."
    datax::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``y``."
    datay::Vector{AbstractArray}
    "fields to contain transformed fields distributed along ``z``."
    dataz::Vector{AbstractArray}
end

function ξsquared(ξx::Real, ξy::Real, ξz::Real)
    a = ξx^2 + ξy^2 + ξz^2
    if abs(a) < 1e-8
        return 1
    else
        return a
    end
end

function ξsquared2(ξx::Real, ξy::Real)
    a = ξx^2 + ξy^2
    if abs(a) < 1e-8
        return 1
    else
        return a
    end
end

"""
$(TYPEDEF)

Type representing a 2D finite difference plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFD2D{N} <: AbstractFDPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "discretization step along ``x``."
    Δx::Real
    "discretization step along ``y``."
    Δy::Real
    "temporary fields distributed along ``x``."
    ϕxtmp::AbstractArray
    "temporary fields distributed along ``y``."
    ϕytmp::AbstractArray
end

"""
$(TYPEDEF)

Type representing a 3D finite difference plan.

It contains the following informations:

$(TYPEDFIELDS)
"""
struct PlanFD3D{N} <: AbstractFDPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "discretization step along ``x``."
    Δx::Real
    "discretization step along ``y``."
    Δy::Real
    "discretization step along ``z``."
    Δz::Real
    "temporary fields distributed along ``x``."
    ϕxtmp::AbstractArray
    "temporary fields distributed along ``y``."
    ϕytmp::AbstractArray
    "temporary fields distributed along ``z``."
    ϕztmp::AbstractArray
end

"""
$(TYPEDSIGNATURES)

Returns a 2D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()` or `FiniteDifferencePlan()`)

# Example

```jldoctest
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FFTPlan());
```

```jldoctest
julia> field = Field(grid, RealField());
julia> plan = Plan(field, t=FiniteDifferencePlan());
```
"""
function Plan(f::F;
              t::PlanType=FFTPlan()) where {F<:AbstractField2D{N,FT,FFT,A}} where {N,FT,FFT,
                                                                                   A}
    # Pencil decompositions
    pen_x = f.pen
    pen_y = Pencil(pen_x; decomp_dims=(1,), permute=Permutation(2, 1))
    if typeof(t) == FFTPlan
        # frequencies
        ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
        ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))

        # create arrays for FFT
        datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x),
                 PencilArray{FFT}(undef, pen_x)]
        datay = [PencilArray{FFT}(undef, pen_y), PencilArray{FFT}(undef, pen_y),
                 PencilArray{FFT}(undef, pen_y)]
        for i in 1:(N - 1)
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
        end

        # create plans
        plan_x = plan_fft(parent(datax[1]), 1)
        plan_y = plan_fft(parent(datay[1]), 1)

        return PlanFFT2D{N}(f,
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
$(TYPEDSIGNATURES)

Returns a 3D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()` or `FiniteDifferencePlan()`)

# Example

```jldoctest
julia> field3 = Field(grid3, RealField());
julia> plan = Plan(field3, t=FFTPlan());
```

```jldoctest
julia> field3 = Field(grid3, RealField());
julia> plan = Plan(field3, t=FiniteDifferencePlan());
```
"""
function Plan(f::F;
              t::PlanType=FFTPlan()) where {F<:AbstractField3D{N,FT,FFT,A}} where {N,FT,FFT,
                                                                                   A}
    # Pencil decompositions
    pen_x = f.pen
    pen_y = Pencil(pen_x; decomp_dims=(1, 3), permute=Permutation(2, 1, 3))
    pen_z = Pencil(pen_x; decomp_dims=(1, 2), permute=Permutation(3, 1, 2))
    if typeof(t) == FFTPlan
        # frequencies
        ξx = A(fftfreq(f.g.nx, 2π / f.g.Δx))
        ξy = A(fftfreq(f.g.ny, 2π / f.g.Δy))
        ξz = A(fftfreq(f.g.nz, 2π / f.g.Δz))

        # create arrays for FFT
        datax = [PencilArray{FFT}(undef, pen_x), PencilArray{FFT}(undef, pen_x),
                 PencilArray{FFT}(undef, pen_x)]
        datay = [PencilArray{FFT}(undef, pen_y), PencilArray{FFT}(undef, pen_y),
                 PencilArray{FFT}(undef, pen_y)]
        dataz = [PencilArray{FFT}(undef, pen_z), PencilArray{FFT}(undef, pen_z),
                 PencilArray{FFT}(undef, pen_z)]
        for i in 1:(N - 1)
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datax, PencilArray{FFT}(undef, pen_x))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(datay, PencilArray{FFT}(undef, pen_y))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
            push!(dataz, PencilArray{FFT}(undef, pen_z))
        end

        # create plans
        plan_x = plan_fft(parent(datax[1]), 1)
        plan_y = plan_fft(parent(datay[1]), 1)
        plan_z = plan_fft(parent(dataz[1]), 1)

        return PlanFFT3D{N}(f,
                            pen_x, pen_y, pen_z,
                            plan_x, plan_y, plan_z,
                            ξx, ξy, ξz,
                            datax, datay, dataz)
    else
        # create arrays for Finite Difference
        ϕxtmp = PencilArray{FFT}(undef, pen_x)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        ϕztmp = PencilArray{FFT}(undef, pen_z)
        return PlanFD3D{N}(f, pen_x, pen_y, pen_z, f.g.Δx, f.g.Δy, f.g.Δz, ϕxtmp, ϕytmp,
                           ϕztmp)
    end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{1}, name::Symbol)
    if name === :ϕx_hat
        return plan.datax[1]
    elseif name === :ϕy_hat
        return plan.datay[1]
    elseif name === :ϕz_hat
        return plan.dataz[1]
    elseif name === :a_tmpx
        return plan.datax[2]
    elseif name === :a_tmpy
        return plan.datay[2]
    elseif name === :a_tmpz
        return plan.dataz[2]
    elseif name === :a_tmp2x
        return plan.datax[3]
    elseif name === :a_tmp2y
        return plan.datay[3]
    elseif name === :a_tmp2z
        return plan.dataz[3]
    else
        return getfield(plan, name)
    end
end

@inline function Base.getproperty(plan::AbstractFFTPlan{N}, name::Symbol) where {N}
    if name === :ux_hat
        return plan.datax[1:N]
    elseif name === :uy_hat
        return plan.datay[1:N]
    elseif name === :uz_hat
        return plan.dataz[1:N]
    elseif name === :uxtmp_hat
        return plan.datax[(N + 1):(2 * N)]
    elseif name === :uytmp_hat
        return plan.datay[(N + 1):(2 * N)]
    elseif name === :uztmp_hat
        return plan.dataz[(N + 1):(2 * N)]
    elseif name === :a_tmpx
        return plan.datax[2 * N + 1]
    elseif name === :a_tmpy
        return plan.datay[2 * N + 1]
    elseif name === :a_tmpz
        return plan.dataz[2 * N + 1]
    elseif name === :a_tmp2x
        return plan.datax[2 * N + 2]
    elseif name === :a_tmp2y
        return plan.datay[2 * N + 2]
    elseif name === :a_tmp2z
        return plan.dataz[2 * N + 2]
    else
        return getfield(plan, name)
    end
end

function mul_x!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    mul!(parent(a_out), plan.plan_x, parent(a_in))
    return nothing
end

function mul_x!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_x!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_y!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_in_y = plan.a_tmpy
    transpose!(a_in_y, a_in)
    mul!(parent(a_out), plan.plan_y, parent(a_in_y))
    return nothing
end

function mul_y!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_y!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_z!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_in_y = plan.a_tmpy
    a_in_z = plan.a_tmpz
    transpose!(a_in_y, a_in)
    transpose!(a_in_z, a_in_y)
    mul!(parent(a_out), plan.plan_z, parent(a_in_z))
    return nothing
end

function mul_z!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_z!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function mul_all!(a_out::PencilArray, plan::PlanFFT2D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    mul!(parent(a_out_x), plan.plan_x, parent(a_in))
    transpose!(a_out_y, a_out_x)
    mul!(parent(a_out), plan.plan_y, parent(a_out_y))
    return nothing
end

function mul_all!(a_out::PencilArray, plan::PlanFFT3D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    a_out_y2 = plan.a_tmp2y
    a_out_z = plan.a_tmpz
    mul!(parent(a_out_x), plan.plan_x, parent(a_in))
    transpose!(a_out_y, a_out_x)
    mul!(parent(a_out_y2), plan.plan_y, parent(a_out_y))
    transpose!(a_out_z, a_out_y2)
    mul!(parent(a_out), plan.plan_z, parent(a_out_z))
    return nothing
end

function mul_all!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                  u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        mul_all!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_x!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    ldiv!(parent(a_out), plan.plan_x, parent(a_in))
    return nothing
end

function ldiv_x!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_x!(u_out[i], plan.plan_x, u_in[i])
    end
    return nothing
end

function ldiv_y!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_out_y = plan.a_tmpy
    ldiv!(parent(a_out_y), plan.plan_y, parent(a_in))
    transpose!(a_out, a_out_y)
    return nothing
end

function ldiv_y!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_y!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function ldiv_z!(a_out::PencilArray, plan::AbstractFFTPlan, a_in::PencilArray)
    a_out_y = plan.a_tmpy
    a_out_z = plan.a_tmpz
    ldiv!(parent(a_out_z), plan.plan_z, parent(a_in))
    transpose!(a_out_y, a_out_z)
    transpose!(a_out, a_out_y)
    return nothing
end

function ldiv_z!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                 u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_z!(u_out[i], plan.plan_x, u_in[i])
    end
    return nothing
end

function ldiv_all!(a_out::PencilArray, plan::PlanFFT2D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    ldiv!(parent(a_out_y), plan.plan_y, parent(a_in))
    transpose!(a_out_x, a_out_y)
    ldiv!(parent(a_out), plan.plan_x, parent(a_out_x))
    return nothing
end

function ldiv_all!(a_out::PencilArray, plan::PlanFFT3D, a_in::PencilArray)
    a_out_x = plan.a_tmpx
    a_out_y = plan.a_tmpy
    a_out_y2 = plan.a_tmp2y
    a_out_z = plan.a_tmpz
    ldiv!(parent(a_out_z), plan.plan_z, parent(a_in))
    transpose!(a_out_y, a_out_z)
    ldiv!(parent(a_out_y2), plan.plan_y, parent(a_out_y))
    transpose!(a_out_x, a_out_y2)
    ldiv!(parent(a_out), plan.plan_x, parent(a_out_x))
    return nothing
end

function ldiv_all!(u_out::Vector{<:AbstractArray}, plan::AbstractFFTPlan{N},
                   u_in::Vector{<:AbstractArray}) where {N}
    for i in 1:N
        ldiv_all!(u_out[i], plan, u_in[i])
    end
    return nothing
end

function grid_x(plan::PlanFFT2D)
    grid = localgrid(plan.pen_x, (plan.f.g.x, plan.f.g.y))
    x, y = grid.x, grid.y
    gridξ = localgrid(plan.pen_x, (plan.ξx, plan.ξy))
    ξx, ξy = gridξ.x, gridξ.y
    return x, y, ξx, ξy
end

function grid_y(plan::PlanFFT2D)
    grid = localgrid(plan.pen_y, (plan.f.g.x, plan.f.g.y))
    x, y = grid.x, grid.y
    gridξ = localgrid(plan.pen_y, (plan.ξx, plan.ξy))
    ξx, ξy = gridξ.x, gridξ.y
    return x, y, ξx, ξy
end

function grid_x(plan::PlanFFT3D)
    grid = localgrid(plan.pen_x, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_x, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end

function grid_y(plan::PlanFFT3D)
    grid = localgrid(plan.pen_y, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_y, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end

function grid_z(plan::PlanFFT3D)
    grid = localgrid(plan.pen_z, (plan.f.g.x, plan.f.g.y, plan.f.g.z))
    x, y, z = grid.x, grid.y, grid.z
    gridξ = localgrid(plan.pen_z, (plan.ξx, plan.ξy, plan.ξz))
    ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
    return x, y, z, ξx, ξy, ξz
end
