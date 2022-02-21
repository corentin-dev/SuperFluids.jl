# Fields

`Field` objects are used to contain the data used by the simulation. It depends on a [Grid](@ref Grids).

!!! info
    While [`Grid`](@ref Grids) is not distributed, `Field` is distributed.


For the moment, there exists two kinds of fields: [`SuperFluids.Field2D`](@ref) and [`SuperFluids.Field3D`](@ref).

## [Constructors](@id field.constructors)

```@docs
Field
```

## [Structures](@id field.structures)

```@docs
SuperFluids.Field2D
```

```@docs
SuperFluids.Field3D
```