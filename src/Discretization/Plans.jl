"Abstract supertype for plan type."
abstract type PlanType end
"FFT plan"
struct FFTPlan <: PlanType end
"Finite difference plan"
struct FiniteDifferencePlan <: PlanType end
"Compact finite difference plan (6th-order periodic compact scheme)."
struct CompactPlan <: PlanType end

export Plan, FFTPlan, FiniteDifferencePlan, CompactPlan

"Abstract supertype for plans."
abstract type AbstractPlan{N} end
"Abstract supertype for FFT plans."
abstract type AbstractFFTPlan{N} end
"Abstract supertype for Finite Difference plans."
abstract type AbstractFDPlan{N} end
"Abstract supertype for Compact Finite Difference plans."
abstract type AbstractCompactPlan{N} end

"""
$(TYPEDEF)

Precomputed multipliers for a compact finite-difference derivative on one axis.

Built by [`compact_setup`](@ref). The cyclic tridiagonal solve (Thomas
multipliers plus the right-hand-side-independent cyclic correction) is
precomputed here, so each evaluation only performs the stencil plus two sweeps.

$(TYPEDFIELDS)
"""
struct CompactAxis{R}
    "number of grid points along the axis."
    n::Int
    "derivative order (1 or 2)."
    order::Int
    "compact coefficient: 1/3 (1st order) or 2/11 (2nd order)."
    alpha::R
    "stencil coefficient a."
    a::R
    "stencil coefficient b."
    b::R
    "Thomas forward multipliers."
    s::Vector{R}
    "reciprocals of the Thomas pivots."
    w::Vector{R}
    "super-diagonal of the (unchanged) tridiagonal."
    f::Vector{R}
    "precomputed cyclic correction vector."
    t::Vector{R}
    "cyclic correction denominator."
    denom::R
end

"""$(TYPEDSIGNATURES)

Build the precomputed multipliers for a compact finite-difference derivative
along an axis of `n` points, spacing `Δ`, derivative `order` (1 or 2).

The operator is the 6th-order compact derivative
`` α·(g(i-1)+g(i+1)) + g(i) = a·(f(i+1)-f(i-1)) + b·(f(i+2)-f(i-2)) ``
(1st order, `α=1/3`, `a=(7/9)/Δ`, `b=(1/36)/Δ`) or the compact second
derivative (2nd order, `α=2/11`, `a=(12/11)/Δ²`, `b=(3/44)/Δ²`), with periodic
wrap.

The periodic wrap makes the coefficient matrix cyclic pentadiagonal. It is
solved without building the matrix: a Thomas factorization of the leading
tridiagonal part plus a cyclic corner correction. Because the corner
perturbation is rank-one (a single `α` coupling at each end), the correction
vector and its denominator depend only on the matrix, not on the right-hand
side; both are precomputed here so that each evaluation costs one stencil plus
two sweeps.
"""
function compact_setup(n::Int, Δ::Real, order::Int)
    if n < 8
        error("compact finite-difference scheme requires at least 8 points along the axis (got $n)")
    end
    R = eltype(Δ)
    if order == 1
        alpha = R(1)/R(3)
        a = (R(7)/R(9))/Δ
        b = (R(1)/R(36))/Δ
    elseif order == 2
        alpha = R(2)/R(11)
        a = (R(12)/R(11))/Δ^2
        b = (R(3)/R(44))/Δ^2
    else
        error("compact_setup: order must be 1 or 2")
    end
    cv = ones(R, n); cv[1] = R(2); cv[n] = R(1) + alpha^2
    bv = fill(alpha, n); bv[n] = zero(R)
    fv = fill(alpha, n); fv[n] = zero(R)
    # Thomas forward pass (non-cyclic part).
    s = zeros(R, n); w = ones(R, n)
    w .= cv
    for i in 2:n
        s[i] = bv[i-1]/w[i-1]
        w[i] = w[i] - fv[i-1]*s[i]
    end
    for i in 1:n
        w[i] = R(1)/w[i]
    end
    # Right-hand-side-independent cyclic correction (swept once, reused).
    t = zeros(R, n); t[1] = -one(R); t[n] = alpha
    for i in 2:n
        t[i] -= t[i-1]*s[i]
    end
    t[n] *= w[n]
    for i in n-1:-1:1
        t[i] = (t[i] - fv[i]*t[i+1])*w[i]
    end
    denom = one(R) + t[1] - alpha*t[n]
    return CompactAxis{R}(n, order, alpha, a, b, s, w, fv, t, denom)
end

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
$(TYPEDEF)

Type representing a 2D compact finite difference plan.

$(TYPEDFIELDS)
"""
struct PlanCompact2D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField2D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "compact multipliers for the first derivative along ``x``."
    ax::CompactAxis
    "compact multipliers for the second derivative along ``x``."
    a2x::CompactAxis
    "compact multipliers for the first derivative along ``y``."
    ay::CompactAxis
    "compact multipliers for the second derivative along ``y``."
    a2y::CompactAxis
    "temporary field distributed along ``y``."
    ϕytmp::AbstractArray
end

"""
$(TYPEDEF)

Type representing a 3D compact finite difference plan.

$(TYPEDFIELDS)
"""
struct PlanCompact3D{N} <: AbstractCompactPlan{N}
    "reference to a field."
    f::AbstractField3D
    "pencil (or slab) in the ``x`` direction."
    pen_x::Any
    "pencil (or slab) in the ``y`` direction."
    pen_y::Any
    "pencil (or slab) in the ``z`` direction."
    pen_z::Any
    "compact multipliers for the first derivative along ``x``."
    ax::CompactAxis
    "compact multipliers for the second derivative along ``x``."
    a2x::CompactAxis
    "compact multipliers for the first derivative along ``y``."
    ay::CompactAxis
    "compact multipliers for the second derivative along ``y``."
    a2y::CompactAxis
    "compact multipliers for the first derivative along ``z``."
    az::CompactAxis
    "compact multipliers for the second derivative along ``z``."
    a2z::CompactAxis
    "temporary field distributed along ``y``."
    ϕytmp::AbstractArray
    "temporary field distributed along ``z``."
    ϕztmp::AbstractArray
end

"""
$(TYPEDSIGNATURES)

Returns a 2D plan. By default a FFT plan is returned.

Parameters are:

- `f`: a field
- `t`: a type of plan (either `FFTPlan()`, `FiniteDifferencePlan()`, or `CompactPlan()`)

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
    elseif typeof(t) == CompactPlan
        if A !== Array
            error("CompactPlan is not supported on GPU arrays (got $A). " *
                  "The compact Thomas solve uses scalar indexing on the local line, " *
                  "which GPU arrays do not allow from the host. Build the grid with " *
                  "array_type=Array to use the compact scheme, or keep the GPU grid " *
                  "and use FFTPlan()/FiniteDifferencePlan().")
        end
        ax = compact_setup(f.g.nx, f.g.Δx, 1)
        a2x = compact_setup(f.g.nx, f.g.Δx, 2)
        ay = compact_setup(f.g.ny, f.g.Δy, 1)
        a2y = compact_setup(f.g.ny, f.g.Δy, 2)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        return PlanCompact2D{N}(f, pen_x, pen_y, ax, a2x, ay, a2y, ϕytmp)
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
- `t`: a type of plan (either `FFTPlan()`, `FiniteDifferencePlan()`, or `CompactPlan()`)

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
    elseif typeof(t) == CompactPlan
        if A !== Array
            error("CompactPlan is not supported on GPU arrays (got $A). " *
                  "The compact Thomas solve uses scalar indexing on the local line, " *
                  "which GPU arrays do not allow from the host. Build the grid with " *
                  "array_type=Array to use the compact scheme, or keep the GPU grid " *
                  "and use FFTPlan()/FiniteDifferencePlan().")
        end
        ax = compact_setup(f.g.nx, f.g.Δx, 1)
        a2x = compact_setup(f.g.nx, f.g.Δx, 2)
        ay = compact_setup(f.g.ny, f.g.Δy, 1)
        a2y = compact_setup(f.g.ny, f.g.Δy, 2)
        az = compact_setup(f.g.nz, f.g.Δz, 1)
        a2z = compact_setup(f.g.nz, f.g.Δz, 2)
        ϕytmp = PencilArray{FFT}(undef, pen_y)
        ϕztmp = PencilArray{FFT}(undef, pen_z)
        return PlanCompact3D{N}(f, pen_x, pen_y, pen_z, ax, a2x, ay, a2y, az, a2z, ϕytmp, ϕztmp)
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
