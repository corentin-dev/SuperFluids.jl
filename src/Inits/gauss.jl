"""
    InitGauss{F} <: AbstractInit{F}

Structure describing a Gauss initializator.

It contains the following informations:

- `f`: a reference to a field.
- `γz`: trapping potential in ``z`` direction.
- `Ω`: rotation speed.

"""
struct InitGauss{F} <: AbstractInit{F}
    f::F
    γz::Real
    Ω::Real
end

"""
    InitGauss(f :: F; γz :: Real = 1., Ω :: Real = 0.) where {F<:AbstractField}

Returns an `InitGauss` initializator object.

Parameters are:

- `f`: a reference field.
- `γz`: trapping potential in ``z`` direction.
- `Ω`: rotation speed.

"""
function InitGauss(f::F; γz::Real=1.0, Ω::Real=0.0) where {F<:AbstractField}
    return InitGauss{F}(f, γz, Ω)
end

"""
    initField!(init::InitGauss{F}) where {F<:AbstractField2D}


Initialize a field using a `InitGauss` initializer for 2D grids.

```math
ϕ = (1-Ω)\\dfrac{1}{\\sqrt{π}} \\exp(-0.5(x^2+y^2)) + Ω \\left(\\dfrac{x+i y}{\\sqrt{π}}\\right)\\exp(-0.5(x^2+y^2))
```

!!! info
    ``ϕ`` is normalized in order to have ``∥ϕ∥₂ = 1``.
"""
function initField!(init::InitGauss{F}) where {F<:AbstractField2D}
    @. init.f.ϕ = (1 - init.Ω) * (1 / sqrt(π) * exp(-0.5 * (init.f.x^2 + init.f.y^2))) +
                  init.Ω * (1 * (init.f.x + im * init.f.y) / sqrt(π) *
                            exp(-0.5 * (init.f.x^2 + init.f.y^2)))
    normalize!(init.f)
    return nothing
end

"""
    initField!(init::InitGauss{F}) where {F<:AbstractField3D}


Initialize a field using a `InitGauss` initializer for 3D grids.

```math
ϕ = (1-Ω)\\dfrac{γ_z^{1/4}}{π^{3/4}} \\exp(-0.5(x^2+y^2+γ_z z^2)) + Ω \\left(\\dfrac{(x+i y)(γ_z^{1/4})}{π^{3/4}}\\right)\\exp(-0.5(x^2+y^2+γ_z z^2))
```

!!! info
    ``ϕ`` is normalized in order to have ``∥ϕ∥₂ = 1``.
"""
function initField!(init::InitGauss{F}) where {F<:AbstractField3D}
    x = init.f.x
    y = init.f.y
    z = init.f.z

    @. init.f.ϕ = (1 - init.Ω) * exp(-0.5 * (x^2 + y^2 + init.γz * z^2)) *
                  (init.γz^0.25 / π^0.75) +
                  init.Ω * (x + im * y + 0 * z) * exp(-0.5 * (x^2 + y^2 + init.γz * z^2)) *
                  (init.γz^0.25 / π^0.75)
    normalize!(init.f)
    return nothing
end

function Base.show(io::IO, init::InitGauss{F}) where {F<:AbstractField2D}
    return print(io, "InitGauss\n",
                 "  ├──────  parameters: Ω $(init.Ω)\n",
                 "  ├─────────────  s1 = 1 / π^1/2 × exp(-1/2 × (x²+y²))\n",
                 "  ├─────────────  s2 = (x+iy) / π^1/2 × exp(-1/2 × (x²+y²))\n",
                 "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")
end

function Base.show(io::IO, init::InitGauss{F}) where {F<:AbstractField3D}
    return print(io, "InitGauss\n",
                 "  ├──────  parameters: γz $(init.γz) Ω $(init.Ω)\n",
                 "  ├─────────────  s1 = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+γz z²))\n",
                 "  ├─────────────  s2 = γz^1/4 (x+iy) / π^3/4 × exp(-1/2 × (x²+y²+γz z²))\n",
                 "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")
end
