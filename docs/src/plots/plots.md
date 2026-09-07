# Plots

`SuperfluidDynamics.jl` produces plots with [`Makie.jl`](https://makie.org/). Only 2D
plots are currently available, aimed at Bose-Einstein-condensate applications
(`plot = true` in `solve!` drives them).

## Precompiling Makie (optional)

Makie is slow to precompile. To avoid the cost on every run, build a
self-contained system image once:

```
pkg> add PackageCompiler
```

```julia
using PackageCompiler
```

```
pkg> activate .
pkg> add GLMakie
```

```julia
create_sysimage(:GLMakie; sysimage_path = "GLMakie.so")
exit()
```

then launch Julia with that image:

```bash
julia -q -JGLMakie.so --project=.
```
