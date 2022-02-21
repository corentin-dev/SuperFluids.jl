"""
    InitThomasFermi{F} <: AbstractInit{F}

Structure describing a Thomas-Fermi initializator.

It contains the following informations:

- `f`: a reference to a field.
- `β`: interaction coefficient.
- `γx`, `γy`, `γz`: quadratic trapping potential coefficients.

"""
struct InitThomasFermi{F} <: AbstractInit{F}
   f :: F
   β :: Real
   γx :: Real
   γy :: Real
   γz :: Real
end

"""
    InitThomasFermi(f :: F, β :: Real; γx :: Real = 1., γy :: Real = 1., γz :: Real = 1.) where {F<:AbstractField}

Returns an `InitThomasFermi` initializator object.

Parameters are:

- `f`: a reference field.
- `β`: interaction coefficient.
- `γx`, `γy`, `γz`: quadratic trapping potential coefficients.

"""
function InitThomasFermi(f :: F, β :: Real; γx :: Real = 1., γy :: Real = 1., γz :: Real = 1.) where {F<:AbstractField}
   return InitThomasFermi{F}(f, β, γx, γy, γz)
end

"""
    initField!(init::InitThomasFermi{F}) where {F<:AbstractField2D}


Initialize a field using a `InitThomasFermi` initializer for 2D grids.

```math
ϕ = \\left\\lbrace
\\begin{aligned}
0&\\text{ if }x^2 + y^2 > ρ0\\\\
ρ_0 - (x^2 + y^2 )&\\text{ else}
\\end{aligned}
\\right.
```

with ``ρ_0 = √( 4β √(γ_x γ_y) / π )``

!!! info
    ``ϕ`` is normalized in order to have ``∥ϕ∥_2 = 1``.
"""
function initField!(init::InitThomasFermi{F}) where {F<:AbstractField2D}
   x = init.f.x
   y = init.f.y
   ρ0 = √( 4* init.β * √(init.γx * init.γy) / π )
   function TF(x,y)
      if x^2 + y^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 )
      end
   end
   @. init.f.ϕ = TF(x,y)
   normalize!(init.f)
   return nothing
end

"""
    initField!(init::InitThomasFermi{F}) where {F<:AbstractField2D}


Initialize a field using a `InitThomasFermi` initializer for 2D grids.

```math
ϕ = \\left\\lbrace
\\begin{aligned}
0&\\text{ if }x^2 + y^2 + z^2> ρ_0\\\\
ρ_0 - (x^2 + y^2 + z^2)&\\text{ else}
\\end{aligned}
\\right.
```

with ``ρ_0 = (30 β √(γ_x γ_y γ_z) / (8π))^{2/5}``

!!! info
    ``ϕ`` is normalized in order to have ``∥ϕ∥₂ = 1``.
"""
function initField!(init::InitThomasFermi{F}) where {F<:AbstractField3D}
   x = init.f.x
   y = init.f.y
   z = init.f.z
   ρ0 = ( 30 * init.β * √(init.γx * init.γy * init.γz) / ( 8 * π ) )^(2/5)
   function TF(x,y,z)
      if x^2 + y^2 + z^2 > ρ0
         return 0
      else
         return ρ0 .- (x^2 + y^2 + z^2)
      end
   end
   @. init.f.ϕ = TF(x,y,z)
   normalize!(init.f)
   return nothing
end

Base.show(io::IO, init::InitThomasFermi{F}) where {F<:AbstractField3D} =
      print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) γz $(init.γz) β $(init.β)\n",
         "  ├──────────────  μ = γz^1/4 / π^3/4 × exp(-1/2 × (x²+y²+z²))\n",
         "  └──────────────  ϕ = (1-Ω) × s1 + Ω × s2")

Base.show(io::IO, init::InitThomasFermi{F}) where {F<:AbstractField2D} =
      print(io, "InitThomasFermi\n",
         "  ├──────  parameters: γx $(init.γx) γy $(init.γy) β $(init.β)\n",
         "  ├──────────────  μ = √(β γx γy)\n",
         "  └──────────────  ϕ = √( √μ - V )")