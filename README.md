# SuperFluids.jl

This package is authored by Corentin Lothodé, and largely inspired by GPS a Fortran program by Philippe Parnaudeau.

## Get package

```
git clone git@plmlab.math.cnrs.fr:lmrs/num/SuperFluids.jl.git
```

Start Julia :
```
julia --project=.
```

Import package :
```
using SuperFluids
```

## Precompile Makie

To save time, you can precompile Makie :
```
# precompilation de Makie
] add PackageCompiler
using PackageCompiler
] activate .
] add GLMakie
create_sysimage(:GLMakie; sysimage_path="GLMakie.so")
exit()
```

Then, you can start julia using :
```
julia -q -JGLMakie.so --project=.
```
