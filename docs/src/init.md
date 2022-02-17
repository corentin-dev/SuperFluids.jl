# Initialization

## Gauss

### Initializer functions

```@docs
initField!(::SuperFluids.InitGauss2D{F}) where {F<:SuperFluids.AbstractField2D}
initField!(::SuperFluids.InitGauss3D{F}) where {F<:SuperFluids.AbstractField3D}
```

### Constructors

```@docs
InitGauss(:: F; :: Real) where {F<:SuperFluids.AbstractField2D}
InitGauss(::F; ::Real, :: Real) where {F<:SuperFluids.AbstractField3D}
```

### Structures

```@docs
SuperFluids.InitGauss2D
SuperFluids.InitGauss3D
```

## External velocity

### Initializer functions

```@docs
initField!(::InitExternalVelocity{F}) where {F<:SuperFluids.AbstractField2D}
```

### Constructors

```@docs
InitExternalVelocity(:: F,:: Real,:: Real) where {F<:SuperFluids.AbstractField}
```

### Structures

```@docs
SuperFluids.InitExternalVelocity
```