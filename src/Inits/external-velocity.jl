"""
    InitExternalVelocity{F} <: AbstractInit{F}

Structure describing an external velocity initializator.

!!! warning
    For the moment, only Taylor-Green is supported.

It contains the following informations:

- `f`: a reference to a field.
- `ξ`: healing length.
- `coeffΔ`: coefficient used in front of the ``\\Delta`` operator.
- `type`: a string describing the kind of initialization.

"""
struct InitExternalVelocity{F} <: AbstractInit{F}
   f :: F
   ξ :: Real
   coeffΔ :: Real
   type :: String
end

"""
    InitExternalVelocity(f :: F, coeffΔ :: Real, β :: Real) where {F<:AbstractField}

Returns an `InitExternalVelocity` initializator object.

Parameters are:

- `f`: a reference field.
- `coeffΔ`: coefficient used in front of the ``\\Delta`` operator.
- `β`: interaction coefficient.

"""
function InitExternalVelocity(f :: F, coeffΔ :: Real, β :: Real) where {F<:AbstractField}
   ξ = √(-coeffΔ / β)
   return InitExternalVelocity{F}(f, ξ, coeffΔ, "TG")
end

"""
    initField!(init::InitExternalVelocity{F}) where {F<:AbstractField2D}


Initialize a field using a `InitExternalVelocity` initializer for 2D grids.

```math
\\phi = (v_1 v_2 v_3 v_4)^i
```

where:
- ``i = \\lfloor \\frac{1}{-2 \\pi \\text{coeff}\\Delta} \\rfloor``
- ``\\lambda = \\sqrt{2} \\cos(x)``
- ``\\mu = \\sqrt{2} \\cos(y)``
- ``v_1 = (\\lambda + i(\\mu - 0.7)) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{\\lambda^2 + (\\mu-0.7)^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{\\lambda^2 + (\\mu-0.7)^2}} ``
- ``v_2 = (\\lambda + i(\\mu + 0.7)) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{\\lambda^2 + (\\mu+0.7)^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{\\lambda^2 + (\\mu+0.7)^2}} ``
- ``v_3 = (\\lambda-0.7 + i\\mu) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{(\\lambda-0.7)^2 + \\mu^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{(\\lambda-0.7)^2 + \\mu^2}} ``
- ``v_4 = (\\lambda+0.7 + i\\mu) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{(\\lambda+0.7)^2 + \\mu^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{(\\lambda+0.7)^2 + \\mu^2}} ``

!!! warning
    initField! on `InitExternalVelocity` is only for a specific geometry at the moment, and only for Taylor-Green.

"""
function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField2D}
   ξ = init.ξ
   coeffΔ = init.coeffΔ
   function TG(x,y)
      λ=cos(x)*√(2.)
      μ=cos(y)*√(2.)
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end

   @. init.f.ϕ = TG(init.f.x, init.f.y)
   return nothing
end

"""
    initField!(init::InitExternalVelocity{F}) where {F<:AbstractField3D}


Initialize a field using a `InitExternalVelocity` initializer for 3D grids.

```math
\\phi = (v_1 v_2 v_3 v_4)^i
```

where:
- ``i = \\lfloor \\frac{1}{-2 \\pi \\text{coeff}\\Delta} \\rfloor``
- ``\\lambda = \\sqrt{2} \\cos(x) \\sqrt{2\\vert\\cos(z)\\vert}}``
- ``\\mu = \\sqrt{2} \\cos(y) \\sqrt{2\\vert\\cos(z)\\vert}} \\sign(\\cos(z))``
- ``v_1 = (\\lambda + i(\\mu - 0.7)) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{\\lambda^2 + (\\mu-0.7)^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{\\lambda^2 + (\\mu-0.7)^2}} ``
- ``v_2 = (\\lambda + i(\\mu + 0.7)) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{\\lambda^2 + (\\mu+0.7)^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{\\lambda^2 + (\\mu+0.7)^2}} ``
- ``v_3 = (\\lambda-0.7 + i\\mu) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{(\\lambda-0.7)^2 + \\mu^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{(\\lambda-0.7)^2 + \\mu^2}} ``
- ``v_4 = (\\lambda+0.7 + i\\mu) \\dfrac{ \\tanh\\left( \\dfrac{ \\sqrt{(\\lambda+0.7)^2 + \\mu^2} }{(\\sqrt{2} \\xi)} \\right) }{\\sqrt{(\\lambda+0.7)^2 + \\mu^2}} ``

!!! warning
    initField! on `InitExternalVelocity` is only for a specific geometry at the moment, and only for Taylor-Green.

"""
function initField!(init::InitExternalVelocity{F}) where {F<:AbstractField3D}
   ξ = init.ξ
   coeffΔ = init.coeffΔ
   function TG(x,y,z)
      λ=cos(x)*√(2*abs(cos(z)))
      μ=cos(y)*√(2*abs(cos(z)))*sign(cos(z))
      vortex1=(λ + im*(μ-0.7))*tanh(√(λ^2 + (μ-0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ-0.7)^2)
      vortex2=(λ + im*(μ+0.7))*tanh(√(λ^2 + (μ+0.7)^2)/ (√(2)*ξ))/√(λ^2 + (μ+0.7)^2)
      vortex3=(λ-0.7 + im*μ)*tanh(√((λ-0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ-0.7)^2 + μ^2)
      vortex4=(λ+0.7 + im*μ)*tanh(√((λ+0.7)^2 + μ^2)/ (√(2)*ξ))/√((λ+0.7)^2 + μ^2)
      return (vortex1*vortex2*vortex3*vortex4)^Int(floor(1 / (-2*π*coeffΔ)))
   end

   @. init.f.ϕ = TG(init.f.x, init.f.y, init.f.z)
   return nothing
end

Base.show(io::IO, init::InitExternalVelocity) =
   print(io, "InitExternalVelocity\n",
      "  └───────────── type: $(init.type)")
