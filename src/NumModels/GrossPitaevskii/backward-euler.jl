mutable struct NumModelBackwardEuler{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    plan::Plan
    M::AbstractArray
    b::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a Backward Euler numerical model.

# Details

It solves a linear system with a Krylov solver.

The linear system is preconditionned with the operator ``M^{-1}``:

```math
M=\\dfrac{1}{Δt} + β ∥ϕⁿ∥ + V(x)
````

and the linear operator ``A`` is as follow:
```math
A = 1 + M⁻¹ \\left(
coeffΔ Δ - iΩLₖ
\\right)
```

And the right hand side is ``b``:
```math
b = (MΔt)⁻¹ϕ
```

# Example

```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelBackwardEuler(field, param, 0.01, 1000, 100)
Backward Euler
  ├──────────  krylov: n iterations 70, tolerance 1.0e-8
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelBackwardEuler(f::AbstractField, param::AbstractParameters,
                               Δt::Real, niter::Integer, freqbckp::Integer;
                               nkrylov::Integer=70, tolkrylov::Real=1e-8,
                               plantype::PlanType=FFTPlan())
    gf = GradientField(f; rotation=true)
    plan = Plan(f; t=plantype)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    M = similar(f.ϕ)
    b = similar(f.ϕ)
    return NumModelBackwardEuler{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                                       Δt, niter, freqbckp,
                                                                       nkrylov, tolkrylov,
                                                                       plan, M, b,
                                                                       writers)
end

function timeStep!(n::NumModelBackwardEuler{F,P}) where {F<:AbstractField,
    P<:GrossPitaevskiiParameters}
# compute matrices and vectors
# M⁻¹ = Δt⁻¹ + NL
# NL = V + β × ∥ϕ∥² for GP
@. n.M = 1.0 / (1 / n.Δt + n.param.pot.V + n.param.β * abs2(n.f.ϕ))
# b = M × ϕ × Δt⁻¹
@. n.b = n.M * n.f.ϕ / n.Δt
# solving
nkrylov = krylov!(n, n.f.ϕ)
# normalize
normalize!(n.f)
return nkrylov
end

function prodA(n::NumModelBackwardEuler{F,P},
ϕt) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
return ϕt .- n.M .* lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelBackwardEuler)
    return print(io,
                 "Backward Euler\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

mutable struct NumModelBackwardEulerNoPrecond{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    plan::Plan
    Anl::AbstractArray
    b::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a Backward Euler numerical model without preconditionning.


It solves a linear system with a Krylov solver.

The linear system is not preconditionned. The linear operator ``A`` is as follow:

```math
M=
````

and the linear operator ``A`` is as follow:
```math
A = \\dfrac{1}{Δt} + coeffΔ Δ + β ∥ϕⁿ∥ + V(x) - iΩLₖ
```

And the right hand side is ``b``:
```math
b = Δt⁻¹ϕ
```


# Example

```jldoctest
julia> param = NumModelBackwardEulerNoPrecond(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelBackwardEuler(field, param, 0.01, 1000, 100)
Backward Euler
  ├──────────  krylov: n iterations 70, tolerance 1.0e-8
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelBackwardEulerNoPrecond(f::AbstractField, param::AbstractParameters,
                                        Δt::Real, niter::Integer, freqbckp::Integer;
                                        nkrylov::Integer=70, tolkrylov::Real=1e-8,
                                        plantype::PlanType=FFTPlan())
    gf = GradientField(f; rotation=true)
    plan = Plan(f; t=plantype)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    Anl = similar(f.ϕ)
    b = similar(f.ϕ)
    return NumModelBackwardEulerNoPrecond{typeof(f),typeof(param),typeof(plan)}(f, gf,
                                                                                param,
                                                                                Δt, niter,
                                                                                freqbckp,
                                                                                nkrylov,
                                                                                tolkrylov,
                                                                                plan, Anl,
                                                                                b,
                                                                                writers)
end

function timeStep!(n::NumModelBackwardEulerNoPrecond{F,P}) where {F<:AbstractField,
                                                                  P<:GrossPitaevskiiParameters}
    # compute matrices and vectors
    # Anl = V + β × ∥ϕ∥²
    @. n.Anl = n.param.pot.V + n.param.β * abs2(n.f.ϕ)
    # b = ϕ × Δt⁻¹
    @. n.b = n.f.ϕ / n.Δt
    # solving
    nkrylov = krylov!(n, n.f.ϕ)
    # normalize
    normalize!(n.f)
    return nkrylov
end

function prodA(n::NumModelBackwardEulerNoPrecond, ϕt)
    return ((1.0 / n.Δt) .+ n.Anl) .* ϕt .- lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelBackwardEulerNoPrecond)
    return print(io,
                 "Backward Euler without preconditionning\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end
