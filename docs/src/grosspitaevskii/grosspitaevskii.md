# Gross-Pitaevskii equation

This section discuss a particular class of equation that can be used for superfluids. The [Gross-Pitaevskii equation](https://en.wikipedia.org/wiki/Gross%E2%80%93Pitaevskii_equation) is an idealized equation for particles at $0K$. It can be viewed as a particular non-linear Shrödinger equation.

$$
i \dfrac{dϕ}{dt} = αΔϕ +  β |ϕ|²ϕ  + V(x⃗)ϕ - i Ω L_z ϕ
$$

with:

- $α$: `coeffΔ` by default it is $-\frac{1}{2}$, but can be anything (negative),
- $β$: `β` it is the interaction coefficient,
- $V(x⃗)$: `pot` a [potential](@ref Potential),
- $Ω$: `Ω` the rotation coefficient along $z$ axis.

A particular equation is solved in the cas of stationary cases. A particular set of
numerical scheme deals with the _imaginary_ time step Gross-Pitaevskii equation.

## Parameters

### Constructor

```@docs
GrossPitaevskiiParameters(; :: Real, :: Real, :: AbstractPotential, :: Real)
```

### Structure

```@docs
SuperFluids.GrossPitaevskiiParameters
```

## Energy

```@docs
SuperFluids.energy
```

## Utilities

```@docs
SuperFluids.lapRot
```