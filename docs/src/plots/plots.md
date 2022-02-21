# Plots in SuperFluids.jl

`SuperFluids.jl` uses `Makie.jl` in order to produce plots. For the moment, only 2D plots has been developped, with BEC applications in mind.

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
