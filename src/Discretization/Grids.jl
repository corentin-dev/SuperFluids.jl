"Abstract supertype for grids."
abstract type AbstractGrid{N,FT<:Real,A} end
"Alias for 1D abstract grids."
const AbstractGrid1D{FT,A} = AbstractGrid{1,FT,A}
"Alias for 2D abstract grids."
const AbstractGrid2D{FT,A} = AbstractGrid{2,FT,A}
"Alias for 3D abstract grids."
const AbstractGrid3D{FT,A} = AbstractGrid{3,FT,A}

export AbstractGrid, AbstractGrid1D, AbstractGrid2D, AbstractGrid3D
export Grid, Grid1D, Grid2D, Grid3D

"""
$(TYPEDEF)

Type representing a grid.

It contains the following informations:

$(TYPEDFIELDS)

See also [`Field`](@ref).
"""
struct Grid{N,FT,A} <: AbstractGrid{N,FT,A}
    "a vector containing the positions along each direction."
    data::NTuple{N,A}
    "dimension of the grid along each direction."
    n::NTuple{N,Integer}
    "minimum bounds of the grid."
    min::NTuple{N,FT}
    "maximum bounds of the grid."
    max::NTuple{N,FT}
    "physical length along each direction."
    L::NTuple{N,FT}
    "corresponds to the minimum distance between each point of the grid in each direction."
    Δ::NTuple{N,FT}
end

"Alias for 1D grids."
const Grid1D{FT,A} = Grid{1,FT,A}
"Alias for 2D grids."
const Grid2D{FT,A} = Grid{2,FT,A}
"Alias for 3D grids."
const Grid3D{FT,A} = Grid{3,FT,A}

"""
$(TYPEDSIGNATURES)

Returns a Grid2D of size `size = (nx,ny)` ranging from `bounds = ((xmin,xmax),(ymin,ymax))` .

# Example

```jldoctest
julia> grid = Grid((32,32), ((-12,12), (-12,12)))
Grid2D
  ├──────  resolution: 32×32
  ├───────  mesh size: 1024
  ├────  grid spacing: 0.75×0.75
  └──────────  domain: [-12.0,12.0]×[-12.0,12.0]
```

See also [`Field`](@ref).
"""
function Grid(size::Tuple{Integer,Integer},
              bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real}};
              FT=Float64, array_type=Array)
    # size
    nx, ny = size
    @assert nx > 0
    @assert ny > 0
    # bounds
    (xmin, xmax), (ymin, ymax) = bounds
    # length
    Lx, Ly = xmax - xmin, ymax - ymin
    @assert Lx > 0
    @assert Ly > 0
    # discretization
    x, y = LinRange(xmin, xmax, nx + 1), LinRange(ymin, ymax, ny + 1)
    # spacing
    Δx, Δy = Lx / nx, Ly / ny
    x = array_type(x[1:(end - 1)])
    y = array_type(y[1:(end - 1)])
    return Grid{2,FT,array_type}((x, y), (nx, ny), (xmin, ymin), (xmax, ymax), (Lx, Ly),
                                 (Δx, Δy))
end

"""
$(TYPEDSIGNATURES)

Returns a Grid3D with of size `size = (nx,ny,nz)` ranging from `bounds = ((xmin,xmax),(ymin,ymax),(zmin,zmax))` .

# Example

```jldoctest
julia> grid = Grid((32,32,32), ((-12,12),(-12,12),(-12,12)))
Grid3D
  ├──────  resolution: 32×32×32
  ├───────  mesh size: 32768
  ├────  grid spacing: 0.75×0.75×0.75
  └──────────  domain: [-12.0,12.0]×[-12.0,12.0]×[-12.0,12.0]
```
"""
function Grid(size::Tuple{Integer,Integer,Integer},
              bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real},Tuple{Real,Real}};
              FT=Float64, array_type=Array)
    # size
    nx, ny, nz = size
    @assert nx > 0
    @assert ny > 0
    @assert nz > 0
    # bounds
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    # length
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    @assert Lx > 0
    @assert Ly > 0
    @assert Lz > 0
    # discretization
    x, y, z = LinRange(xmin, xmax, nx + 1), LinRange(ymin, ymax, ny + 1),
              LinRange(zmin, zmax, nz + 1)
    # spacing
    Δx, Δy, Δz = Lx / nx, Ly / ny, Lz / nz
    x = array_type(x[1:(end - 1)])
    y = array_type(y[1:(end - 1)])
    z = array_type(z[1:(end - 1)])
    return Grid{3,FT,array_type}((x, y, z), (nx, ny, nz), (xmin, ymin, zmin),
                                 (xmax, ymax, zmax),
                                 (Lx, Ly, Lz), (Δx, Δy, Δz))
end

Base.eltype(::AbstractGrid{FT}) where {FT} = FT
Base.size(grid::AbstractGrid) = Tuple(grid.n)
Base.length(grid::AbstractGrid) = Tuple(grid.L)

function Base.show(io::IO, g::Grid2D)
    return print(io, "Grid2D\n",
                 "  ├──────  resolution: $(g.nx)×$(g.ny)\n",
                 "  ├───────  mesh size: $(g.nx*g.ny)\n",
                 "  ├────  grid spacing: $(g.Δx)×$(g.Δy)\n",
                 "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]")
end

function Base.show(io::IO, g::Grid3D)
    return print(io, "Grid3D\n",
                 "  ├──────  resolution: $(g.nx)×$(g.ny)×$(g.nz)\n",
                 "  ├───────  mesh size: $(g.nx*g.ny*g.nz)\n",
                 "  ├────  grid spacing: $(g.Δx)×$(g.Δy)×$(g.Δz)\n",
                 "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]×[$(g.zmin),$(g.zmax)]")
end

@inline function Base.getproperty(f::AbstractGrid, name::Symbol)
    if name === :x
        f.data[1]
    elseif name === :y
        f.data[2]
    elseif name === :z
        f.data[3]
    elseif name === :nx
        f.n[1]
    elseif name === :ny
        f.n[2]
    elseif name === :nz
        f.n[3]
    elseif name === :xmin
        f.min[1]
    elseif name === :ymin
        f.min[2]
    elseif name === :zmin
        f.min[3]
    elseif name === :xmax
        f.max[1]
    elseif name === :ymax
        f.max[2]
    elseif name === :zmax
        f.max[3]
    elseif name === :Lx
        f.L[1]
    elseif name === :Ly
        f.L[2]
    elseif name === :Lz
        f.L[3]
    elseif name === :Δx
        f.Δ[1]
    elseif name === :Δy
        f.Δ[2]
    elseif name === :Δz
        f.Δ[3]
    else
        getfield(f, name)
    end
end
