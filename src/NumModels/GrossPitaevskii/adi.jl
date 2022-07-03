mutable struct NumModelADI1{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    plan::Plan
    writers::AbstractWriterCollection{F}
end


"""
$(TYPEDSIGNATURES)

Returns a first order splitting scheme (only FFT).

# Details

TODO

# Example

```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelADI1(field, param, 0.01, 1000, 100)
Splitting Order 1
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelADI1(f::AbstractField, param::AbstractParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    return NumModelADI1{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                              Δt, niter, freqbckp,
                                                              plan,
                                                              writers)
end

function Base.show(io::IO, n::NumModelADI1)
    return print(io,
                 "Splitting Order 1\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

function timeStep!(n::NumModelADI1)
    solveLapRot!(n, n.Δt)
    solveNL!(n, n.Δt)
    return 1
end

mutable struct NumModelADI2{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    plan::Plan
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a second order splitting scheme (only FFT).

# Details

TODO

# Example

```jldoctest
julia> param = GrossPitaevskiiParameters(β = 1000, Ω = 0.8, pot = PotentialZero(field));
julia> nummodel = NumModelADI2(field, param, 0.01, 1000, 100)
Splitting Order 2
  ├───────  time step: 0.01
  └──────────── solve: number of iterations 1000, backup frequency 100
```
"""
function NumModelADI2(f::AbstractField, param::AbstractParameters,
                      Δt::Real, niter::Integer, freqbckp::Integer)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    return NumModelADI2{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                              Δt, niter, freqbckp,
                                                              plan,
                                                              writers)
end

function Base.show(io::IO, n::NumModelADI2)
    return print(io,
                 "Splitting Order 2\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

function timeStep!(n::NumModelADI2)
    solveLapRot!(n, n.Δt * 0.5)
    solveNL!(n, n.Δt)
    solveLapRot!(n, n.Δt * 0.5)
    return 2
end

function solveLapRot!(n::AbstractNumModel{F}, Δtl) where {F<:AbstractField2D}
    # field
    ϕ = n.f.ϕ
    # parameters
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    # temporary fields
    ϕxthat = n.plan.ϕx_hat
    ϕythat = n.plan.ϕy_hat
    # FFT x
    x, y, ξx, ξy = grid_x(n.plan)
    mul_x!(ϕxthat, n.plan, ϕ)
    @. ϕxthat = exp(im * (coeffΔ * ξx^2 - Ω * y * ξx) * Δtl) * ϕxthat
    ldiv_x!(ϕ, n.plan, ϕxthat)
    # FFT y
    x, y, ξx, ξy = grid_y(n.plan)
    mul_y!(ϕythat, n.plan, ϕ)
    @. ϕythat = exp(im * (coeffΔ * ξy^2 + Ω * x * ξy) * Δtl) * ϕythat
    ldiv_y!(ϕ, n.plan, ϕythat)
    return nothing
end

function solveLapRot!(n::AbstractNumModel{F,P},
                      Δtl) where {F<:AbstractField3D,P<:GrossPitaevskiiParameters}
    # field
    ϕ = n.f.ϕ
    # parameters
    coeffΔ, Ω = n.param.coeffΔ, n.param.Ω
    # temporary fields
    ϕxthat = n.plan.ϕx_hat
    ϕythat = n.plan.ϕy_hat
    ϕzthat = n.plan.ϕz_hat
    # FFT x
    x, y, z, ξx, ξy, ξz = grid_x(n.plan)
    mul_x!(ϕxthat, n.plan, ϕ)
    @. ϕxthat = exp(im * (coeffΔ * ξx^2 - Ω * y * ξx) * Δtl) * ϕxthat
    ldiv_x!(ϕ, n.plan, ϕxthat)
    # FFT y
    x, y, z, ξx, ξy, ξz = grid_y(n.plan)
    mul_y!(ϕythat, n.plan, ϕ)
    @. ϕythat = exp(im * (coeffΔ * ξy^2 + Ω * x * ξy) * Δtl) * ϕythat
    ldiv_y!(ϕ, n.plan, ϕythat)
    # FFT z
    x, y, z, ξx, ξy, ξz = grid_z(n.plan)
    mul_z!(ϕzthat, n.plan, ϕ)
    @. ϕzthat = exp(im * (coeffΔ * ξz^2) * Δtl) * ϕzthat
    ldiv_z!(ϕ, n.plan, ϕzthat)
    return nothing
end

function solveNL!(n::AbstractNumModel{F,P},
                  Δtl) where {F<:AbstractField,P<:GrossPitaevskiiParameters}
    # ϕ ↦ exp ( -i ( V + ∥ϕ∥² ) Δt ) ϕ
    @.n.f.ϕ = exp(-1im * (n.param.pot.V + n.param.β * abs2(n.f.ϕ)) * Δtl) * n.f.ϕ
end
