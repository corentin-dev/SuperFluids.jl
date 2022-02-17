# Initialization

## Gauss

### Initializer functions

```@docs
initField!(::SuperFluids.InitGauss{F}) where {F<:SuperFluids.AbstractField2D}
initField!(::SuperFluids.InitGauss{F}) where {F<:SuperFluids.AbstractField3D}
```

### Constructors

```@docs
InitGauss(::F; ::Real, :: Real) where {F<:SuperFluids.AbstractField}
```

### Structures

```@docs
SuperFluids.InitGauss
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