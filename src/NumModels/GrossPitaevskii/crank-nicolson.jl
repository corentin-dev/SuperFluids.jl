mutable struct NumModelCrankNicolson{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    nnewton::Integer
    tolnewton::Real
    plan::Plan
    M::AbstractArray
    b::AbstractArray
    Anl::AbstractArray
    Anl2::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a stationnary Crank-Nicolson scheme numerical model.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ + 1/2 Aₙₗ₂ conj(ϕ) - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 2 × β × ∥ψ∥²

Aₙₗ₂ = β × ψ²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function NumModelCrankNicolson(f, param,
                               Δt::Real, niter::Integer, freqbckp::Integer;
                               nkrylov::Integer=70, tolkrylov::Real=1e-8,
                               nnewton::Integer=15, tolnewton::Real=1e-6)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    M = similar(f.ϕ)
    b = similar(f.ϕ)
    Anl = similar(f.ϕ)
    Anl2 = similar(f.ϕ)
    return NumModelCrankNicolson{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                                       Δt, niter, freqbckp,
                                                                       nkrylov, tolkrylov,
                                                                       nnewton, tolnewton,
                                                                       plan, M, b, Anl,
                                                                       Anl2,
                                                                       writers)
end

function timeStep!(n::NumModelCrankNicolson{F,P}) where {F<:AbstractField,
                                                         P<:GrossPitaevskiiParameters}
    # create working vectors
    ψ = similar(n.f.ϕ)
    ϕ₁ = copy(n.f.ϕ)
    # for krylov
    ϕw = similar(n.f.ϕ)
    ϕw .= 0
    nkrylov = 0
    # newton iterations
    for itnewton in 1:(n.nnewton)
        # compute matrices and vectors
        # ψ = 1/2 (ϕ₁+ϕ₀)
        @. ψ = 0.5 * (ϕ₁ + n.f.ϕ)
        # Anls = V + 2 × β × ∥ψ∥²
        @. n.Anl = n.param.pot.V + 2 * n.param.β * abs2(ψ)
        # Anls2 = β × ψ²
        @. n.Anl2 = n.param.β * ψ * ψ
        # M⁻¹ = Δt⁻¹ + V + 3 × β × ∥ψ∥² + 0.5
        @. n.M = 1.0 / (1.0 / n.Δt + (n.param.pot.V + 3 * n.param.β * abs2(ψ)) * 0.5)
        # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
        n.b .= (1 / n.Δt) .* (ϕ₁ .- n.f.ϕ) .+
               (n.param.pot.V .+ n.param.β .* abs2.(ψ)) .* ψ .- lapRot(n, ψ)
        # solving
        nkrylov += krylovPreCond!(n, ϕw)
        # update solution
        ϕ₁ .= ϕ₁ .- ϕw
        # newton residual
        resNewton = sqrt(sum(abs2.(ϕw)))
        if resNewton < n.tolnewton
            println_parallel("Number of Newton iterations: $(itnewton)")
            break
        elseif itnewton == n.nnewton
            println_parallel("warning: Newton algorithm did not converge")
        end
    end
    # update field
    n.f.ϕ .= ϕ₁
    # normalize
    normalize!(n.f)
    return nkrylov
end

"""
$(TYPEDSIGNATURES)

Matrix-vector product for the stationnary Crank-Nicolson Newton-Raphson method :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ + 1/2 Aₙₗ₂ conj(ϕ) - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolson, ϕt)
    return (1 / n.Δt .+ n.Anl) .* ϕt .+ 0.5 .* n.Anl2 .* conj.(ϕt) .- 0.5 .* lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelCrankNicolson)
    return print(io,
                 "Crank-Nicolson Newton-Raphson scheme\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

mutable struct NumModelCrankNicolsonQuasiNewton{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    nnewton::Integer
    tolnewton::Real
    plan::Plan
    M::AbstractArray
    b::AbstractArray
    Anl::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns a stationnary Crank-Nicolson Quasi-Newton scheme numerical model.

# Detail

A quasi-Newton loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 3 × β × ∥ψ∥²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function NumModelCrankNicolsonQuasiNewton(f, param,
                                          Δt::Real, niter::Integer, freqbckp::Integer;
                                          nkrylov::Integer=70, tolkrylov::Real=1e-8,
                                          nnewton::Integer=15, tolnewton::Real=1e-6)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    M = similar(f.ϕ)
    b = similar(f.ϕ)
    Anl = similar(f.ϕ)
    return NumModelCrankNicolsonQuasiNewton{typeof(f),typeof(param),typeof(plan)}(f, gf,
                                                                                  param,
                                                                                  Δt, niter,
                                                                                  freqbckp,
                                                                                  nkrylov,
                                                                                  tolkrylov,
                                                                                  nnewton,
                                                                                  tolnewton,
                                                                                  plan, M,
                                                                                  b, Anl,
                                                                                  writers)
end

function timeStep!(n::NumModelCrankNicolsonQuasiNewton{F,P}) where {F<:AbstractField,
                                                                    P<:GrossPitaevskiiParameters}
    # create working vectors
    ψ = similar(n.f.ϕ)
    ϕ₁ = copy(n.f.ϕ)
    # for krylov
    ϕw = similar(n.f.ϕ)
    ϕw .= 0
    # newton iterations
    nkrylov = 0
    for itnewton in 1:(n.nnewton)
        # compute matrices and vectors
        # ψ = 1/2 (ϕ₁+ϕ₀)
        @. ψ = 0.5 * (ϕ₁ + n.f.ϕ)
        # Anls = V + 3 × β × ∥ψ∥²
        @. n.Anl = n.param.pot.V + 3 * n.param.β * abs2(ψ)
        # M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )
        @. n.M = 1.0 / (1.0 / n.Δt + (n.param.pot.V + 3 * n.param.β * abs2(ψ)) * 0.5)
        # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
        n.b .= (1 / n.Δt) .* (ϕ₁ .- n.f.ϕ) .+
               (n.param.pot.V .+ n.param.β .* abs2.(ψ)) .* ψ .- lapRot(n, ψ)
        # solving
        nkrylov += krylovPreCond!(n, ϕw)
        # update solution
        ϕ₁ .= ϕ₁ .- ϕw
        # newton residual
        resNewton = sqrt(sum(abs2.(ϕw)))
        if resNewton < n.tolnewton
            println_parallel("Number of Newton iterations: $(itnewton)")
            break
        elseif itnewton == n.nnewton
            println_parallel("warning: Newton algorithm did not converge")
        end
    end
    # update field
    n.f.ϕ .= ϕ₁
    # normalize
    normalize!(n.f)
    return nkrylov
end

"""
$(TYPEDSIGNATURES)

Matrix-vector product for the stationnary Crank-Nicolson Quasi-Newton method :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonQuasiNewton, ϕt)
    return (1 / n.Δt .+ n.Anl .* 0.5) .* ϕt .- 0.5 .* lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewton)
    return print(io,
                 "Crank-Nicolson Quasi-Newton scheme\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

mutable struct NumModelCrankNicolsonT{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    nnewton::Integer
    tolnewton::Real
    plan::Plan
    M::AbstractArray
    b::AbstractArray
    Anl::AbstractArray
    Anl2::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns an unstationnary Crank-Nicolson scheme numerical model.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ - 1/2 Aₙₗ₂ conj(ϕ) + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ - 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 2 × β × ∥ψ∥²

Aₙₗ₂ = β × ψ²

b = i Δt⁻¹ × (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function NumModelCrankNicolsonT(f::AbstractField, param::AbstractParameters,
                                Δt::Real, niter::Integer, freqbckp::Integer;
                                nkrylov::Integer=70, tolkrylov::Real=1e-8,
                                nnewton::Integer=15, tolnewton::Real=1e-6)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    M = similar(f.ϕ)
    b = similar(f.ϕ)
    Anl = similar(f.ϕ)
    Anl2 = similar(f.ϕ)
    return NumModelCrankNicolsonT{typeof(f),typeof(param),typeof(plan)}(f, gf, param,
                                                                        Δt, niter, freqbckp,
                                                                        nkrylov, tolkrylov,
                                                                        nnewton, tolnewton,
                                                                        plan, M, b, Anl,
                                                                        Anl2,
                                                                        writers)
end

function timeStep!(n::NumModelCrankNicolsonT{F,P}) where {F<:AbstractField,
                                                          P<:GrossPitaevskiiParameters}
    # create working vectors
    ψ = similar(n.f.ϕ)
    ϕ₁ = copy(n.f.ϕ)
    # for krylov
    ϕw = similar(n.f.ϕ)
    ϕw .= 0
    nkrylov = 0
    # newton iterations
    for itnewton in 1:(n.nnewton)
        # compute matrices and vectors
        # ψ = 1/2 (ϕ₁+ϕ₀)
        @. ψ = 0.5 * (ϕ₁ + n.f.ϕ)
        # Anls = V + 2 × β × ∥ψ∥²
        @. n.Anl = n.param.pot.V + 2 * n.param.β * abs2(ψ)
        # Anls2 = β × ψ²
        @. n.Anl2 = n.param.β * ψ * ψ
        # M⁻¹ = i Δt⁻¹ - 1/2 ( V + 3 × β × ∥ψ∥²)
        @. n.M = 1.0 / (1.0 * im / n.Δt - (n.param.pot.V + 3 * n.param.β * abs2(ψ)) * 0.5)
        # b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
        n.b .= (1.0 * im / n.Δt) .* (ϕ₁ .- n.f.ϕ) .-
               (n.param.pot.V .+ n.param.β .* abs2.(ψ)) .* ψ .+ lapRot(n, ψ)
        # solving
        nkrylov += krylovPreCond!(n, ϕw)
        # update solution
        ϕ₁ .= ϕ₁ .- ϕw
        # newton residual
        resNewton = sqrt(sum(abs2.(ϕw)))
        if resNewton < n.tolnewton
            println_parallel("Number of Newton iterations: $(itnewton)")
            break
        elseif itnewton == n.nnewton
            println_parallel("warning: Newton algorithm did not converge")
        end
    end
    # update field
    n.f.ϕ .= ϕ₁
    return nkrylov
end

"""
$(TYPEDSIGNATURES)

Matrix-vector product for the unstationnary Crank-Nicolson Newton-Raphson method :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ - 1/2 Aₙₗ₂ conj(ϕ) + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonT, ϕt)
    return ((1.0 * im / n.Δt) .- 0.5 .* n.Anl) .* ϕt .- 0.5 .* n.Anl2 .* conj.(ϕt) .+
           0.5 .* lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelCrankNicolsonT)
    return print(io,
                 "Time dependant Crank-Nicolson Newton-Raphson scheme\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end

mutable struct NumModelCrankNicolsonQuasiNewtonT{F,P,Plan} <: AbstractNumModel{F,P,Plan}
    f::F
    gf::Any
    param::P
    Δt::Real
    niter::Integer
    freqbckp::Integer
    nkrylov::Integer
    tolkrylov::Real
    nnewton::Integer
    tolnewton::Real
    plan::Plan
    M::AbstractArray
    b::AbstractArray
    Anl::AbstractArray
    writers::AbstractWriterCollection{F}
end

"""
$(TYPEDSIGNATURES)

Returns an unstationnary Crank-Nicolson Quasi-Newton scheme numerical model.

# Detail

A Newton-Raphson loop using `n.nnewton` iterations until the tolerance `n.tolnewton` is reached, followed by a renormalization step. During the non-linear loop, the linear problem MAϕ=Mb is solved using a krylov solver. The system is built using :

Aϕ = ( Δt⁻¹ + 1/2 Aₙₗ) ϕ - 1/2 (coeffΔ Δϕ + Ω Lz ϕ)

M⁻¹ = Δt⁻¹ + 1/2 × ( V + 3 × β × ∥ψ∥² )

Aₙₗ = V + 3 × β × ∥ψ∥²

b = Δt⁻¹ * (ϕ₁ - ϕ₀) + (V + β × ∥ψ∥²) × ψ + (coeffΔ × Δ + Ω Lz) × ψ
"""
function NumModelCrankNicolsonQuasiNewtonT(f::AbstractField, param::AbstractParameters,
                                           Δt::Real, niter::Integer, freqbckp::Integer;
                                           nkrylov::Integer=70, tolkrylov::Real=1e-8,
                                           nnewton::Integer=15, tolnewton::Real=1e-6)
    gf = GradientField(f; rotation=true)
    plan = Plan(f)
    writer = WriterVTK(f)
    saver = WriterSave(f)
    writers = WriterCollection([writer, saver])
    M = similar(f.ϕ)
    b = similar(f.ϕ)
    Anl = similar(f.ϕ)
    return NumModelCrankNicolsonQuasiNewtonT{typeof(f),typeof(param),typeof(plan)}(f, gf,
                                                                                   param,
                                                                                   Δt,
                                                                                   niter,
                                                                                   freqbckp,
                                                                                   nkrylov,
                                                                                   tolkrylov,
                                                                                   nnewton,
                                                                                   tolnewton,
                                                                                   plan, M,
                                                                                   b, Anl,
                                                                                   writers)
end

function timeStep!(n::NumModelCrankNicolsonQuasiNewtonT{F,P}) where {F<:AbstractField,
                                                                     P<:GrossPitaevskiiParameters}
    # create working vectors
    ψ = similar(n.f.ϕ)
    ϕ₁ = copy(n.f.ϕ)
    # for krylov
    ϕw = similar(n.f.ϕ)
    ϕw .= 0
    nkrylov = 0
    # newton iterations
    for itnewton in 1:(n.nnewton)
        # compute matrices and vectors
        # ψ = 1/2 (ϕ₁+ϕ₀)
        @. ψ = 0.5 * (ϕ₁ + n.f.ϕ)
        # Anls = V + 3 × β × ∥ψ∥²
        @. n.Anl = n.param.pot.V + 3 * n.param.β * abs2(ψ)
        # M⁻¹ = i Δt⁻¹ - 1/2 × ( V + 3 × β × ∥ψ∥² )
        @. n.M = 1.0 / (1.0 * im / n.Δt - (n.param.pot.V + 3 * n.param.β * abs2(ψ)) * 0.5)
        # b = i Δt⁻¹ * (ϕ₁ - ϕ₀) - (V + β × ∥ψ∥²) × ψ + (coeffΔ Δ + Ω Lz) × ψ + coeffΔ Δϕ - Ω Lz Φ
        n.b .= (1.0 * im / n.Δt) .* (ϕ₁ .- n.f.ϕ) .-
               (n.param.pot.V .+ n.param.β .* abs2.(ψ)) .* ψ .+ lapRot(n, ψ)
        # solving
        nkrylov += krylovPreCond!(n, ϕw)
        # update solution
        ϕ₁ .= ϕ₁ .- ϕw
        # newton residual
        resNewton = sqrt(sum(abs2.(ϕw)))
        if resNewton < n.tolnewton
            println_parallel("Number of Newton iterations: $(itnewton)")
            break
        elseif itnewton == n.nnewton
            println_parallel("warning: Newton algorithm did not converge")
        end
    end
    # update field
    n.f.ϕ .= ϕ₁
    return nkrylov
end

"""
$(TYPEDSIGNATURES)

Matrix-vector product for the unstationnary Crank-Nicolson Quasi-Newton method :

Aϕ = ( i Δt⁻¹ - 1/2 Aₙₗ) ϕ + 1/2 (coeffΔ Δϕ + Ω Lz ϕ)
"""
function prodA(n::NumModelCrankNicolsonQuasiNewtonT, ϕt)
    return ((1.0 * im / n.Δt) .- n.Anl .* 0.5) .* ϕt .+ 0.5 .* lapRot(n, ϕt)
end

function Base.show(io::IO, n::NumModelCrankNicolsonQuasiNewtonT)
    return print(io,
                 "Time dependant Crank-Nicolson Quasi-Newton scheme\n",
                 "  ├──────────  krylov: n iterations $(n.nkrylov), tolerance $(n.tolkrylov)\n",
                 "  ├──────────  newton: n iterations $(n.nnewton), tolerance $(n.tolnewton)\n",
                 "  ├───────  time step: $(n.Δt)\n",
                 "  └──────────── solve: number of iterations $(n.niter), backup frequency $(n.freqbckp)")
end
