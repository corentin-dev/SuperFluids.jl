# Initialization

## Thomas-Fermi

### Initializer functions

```@docs
initField!(::SuperFluids.InitThomasFermi{F}) where {F<:SuperFluids.AbstractField2D}
initField!(::SuperFluids.InitThomasFermi{F}) where {F<:SuperFluids.AbstractField3D}
```

### Constructors

```@docs
InitThomasFermi(:: F,:: Real;:: Real, :: Real, :: Real) where {F<:SuperFluids.AbstractField}
```

### Structures

```@docs
SuperFluids.InitThomasFermi
```

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