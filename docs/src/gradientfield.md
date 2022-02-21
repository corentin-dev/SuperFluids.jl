# Gradient Field

Gradient field is used to contain derivatives of a field by the simulation. It depends on a [field](@ref Fields). It may or may not compute rotation along the ``z`` axis.

!!! info
    There exists two kinds of `GradientField` classes, with or without rotation.

## Constructors

```@docs
GradientField
```

## Structures

```@docs
SuperFluids.GradientField2D
```

```@docs
SuperFluids.GradientRotField2D
```

```@docs
SuperFluids.GradientField3D
```

```@docs
SuperFluids.GradientRotField3D
```