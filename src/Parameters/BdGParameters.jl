"""
$(TYPEDEF)

Parameters of the Bogoliubov-de Gennes eigenproblem linearized about a
stationary Gross-Pitaevskii state ``\\psi_0``:

```math
\\begin{pmatrix} H & -K \\\\ K^* & -H \\end{pmatrix}
\\begin{pmatrix} u \\\\ v \\end{pmatrix} = \\omega
\\begin{pmatrix} u \\\\ v \\end{pmatrix}
```

with

```math
H = -\\text{coeffΔ}\\,\\nabla^2 + V(\\boldsymbol{x}) + 2\\beta|\\psi_0|^2 - \\mu
- i\\Omega \\mathcal{L}_z,\\qquad
K = \\beta\\,\\psi_0^2,
```

where ``\\mu = \\langle\\psi_0, (-\\text{coeffΔ} + V + \\beta|\\psi_0|^2)\\psi_0
\\rangle`` is the chemical potential of the stationary state and
``\\mathcal{L}_z = y\\,\\partial_x - x\\,\\partial_y`` is the rotation operator.

It contains the following informations:

$(TYPEDFIELDS)
"""
mutable struct BdGParameters <: AbstractParameters
    "coefficient in front of the Δ term (same convention as the GP models)."
    coeffΔ::Real
    "interaction coefficient."
    β::Real
    "potential."
    pot::AbstractPotential
    "rotation along ``z`` axis."
    Ω::Real
end

"""
$(TYPEDSIGNATURES)

Returns a `BdGParameters`.

Parameters are:
- `coeffΔ`: coefficient in front of the ``\\Delta`` operator,
- `β`: interaction coefficient,
- `pot`: potential (of class `AbstractPotential`),
- `Ω`: rotation along the ``z`` axis.

# Example

```jldoctest
julia> param = BdGParameters(β = 1000, Ω = 0.0, pot = PotentialZero(field))
BdGParameters
  └──────────────── coeffΔ: -0.5, β: 1000, Ω: 0.0
```
"""
function BdGParameters(; coeffΔ::Real=-0.5, β::Real=1.0,
                        pot::AbstractPotential, Ω::Real=0.0)
    return BdGParameters(coeffΔ, β, pot, Ω)
end

function Base.show(io::IO, param::BdGParameters)
    return print(io, "BdGParameters\n",
                 "  ├──────  linearized about a stationary GP state:\n",
                 "  ├─────── H = $(param.coeffΔ) Δ + V + 2 $(param.β)|ψ₀|² - μ - i $(param.Ω) Lz\n",
                 "  └──────────────── K = $(param.β) ψ₀²,  μ: chemical potential of ψ₀")
end
