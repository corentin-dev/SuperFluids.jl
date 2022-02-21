"Abstract supertype for grids."
abstract type AbstractGrid{FT<:Real,A} end
"Abstract supertype for 2D grids."
abstract type AbstractGrid2D{FT<:Real,A} <: AbstractGrid{FT,A} end
"Abstract supertype for 3D grids."
abstract type AbstractGrid3D{FT<:Real,A} <: AbstractGrid{FT,A} end

export Grid

"""
    Grid2D{FT<:Real} <: AbstractGrid2D{FT}

Type representing a 2D grid.

A `Grid2D` contains the following informations:

- `x`, `y`: a vector containing the positions along each direction.
- `nx`, `ny`: dimension of the grid along each direction.
- `xmin`, `xmax`, `ymin`, `ymax`: bounds of the grid.
- `Lx`, `Ly`: physical length along each direction.
- `Δx`, `Δy`: corresponds to the minimum distance between each point of the grid in each direction.
"""
struct Grid2D{FT<:Real,A} <: AbstractGrid2D{FT,A}
   x :: A
   y :: A
   nx :: Integer
   ny :: Integer
   xmin :: FT
   xmax :: FT
   ymin :: FT
   ymax :: FT
   Lx :: FT
   Ly :: FT
   Δx :: FT
   Δy :: FT
end

"""
    Grid3D{FT<:Real} <: AbstractGrid3D{FT}

Type representing a 2D grid.

A `Grid2D` contains the following informations:

- `x`, `y`, `z`: a vector containing the positions along each direction.
- `nx`, `ny`, `nz`: dimension of the grid along each direction.
- `xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax`: bounds of the grid.
- `Lx`, `Ly`, `Lz`: physical length along each direction.
- `Δx`, `Δy`, `Δz`: corresponds to the minimum distance between each point of the grid in each direction.

"""
struct Grid3D{FT,A} <: AbstractGrid3D{FT,A}
   x :: A
   y :: A
   z :: A
   nx :: Integer
   ny :: Integer
   nz :: Integer
   xmin :: FT
   xmax :: FT
   ymin :: FT
   ymax :: FT
   zmin :: FT
   zmax :: FT
   Lx :: FT
   Ly :: FT
   Lz :: FT
   Δx :: FT
   Δy :: FT
   Δz :: FT
end

"""
    Grid(size::Tuple{Integer,Integer},
      bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real}};
      FT=Float64,device=CPU())

Returns a Grid2D with of size `size = (nx,ny)` ranging from `bounds = ((xmin,xmax),(ymin,ymax))` .

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)))
Grid2D
  ├──────  resolution: 128×128
  ├───────  mesh size: 16384
  ├────  grid spacing: 0.1875×0.1875
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
   (xmin,xmax),(ymin,ymax) = bounds
   # length
   Lx, Ly = xmax - xmin, ymax - ymin
   @assert Lx > 0
   @assert Ly > 0
   # discretization
   x, y = LinRange(xmin,xmax,nx+1), LinRange(ymin,ymax,ny+1)
   # spacing
   Δx, Δy = Lx / nx, Ly / ny
   x = array_type(x[1:end-1])
   y = array_type(y[1:end-1])
   return Grid2D{FT,array_type}(x, y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
end

"""
    Grid(size::Tuple{Integer,Integer,Integer},
      bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real},Tuple(Real,Real};
      FT=Float64,device=CPU())

Returns a Grid3D with of size `size = (nx,ny,nz)` ranging from `bounds = ((xmin,xmax),(ymin,ymax),(zmin,zmax))` .

Example
=======
```jldoctest
julia> grid = Grid((128,128,128), ((-12,12),(-12,12),(-12,12)))
Grid3D
  ├──────  resolution: 128×128×128
  ├───────  mesh size: 2097152
  ├────  grid spacing: 0.1875×0.1875×0.1875
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
   (xmin,xmax),(ymin,ymax), (zmin,zmax) = bounds
   # length
   Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
   @assert Lx > 0
   @assert Ly > 0
   @assert Lz > 0
   # discretization
   x, y, z = LinRange(xmin,xmax,nx+1), LinRange(ymin,ymax,ny+1), LinRange(zmin,zmax,nz+1)
   # spacing
   Δx, Δy, Δz = Lx / nx, Ly / ny, Lz / nz
   x = array_type(x[1:end-1])
   y = array_type(y[1:end-1])
   z = array_type(z[1:end-1])
   return Grid3D{FT,array_type}(x, y, z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz)
end

Base.eltype(::AbstractGrid{FT}) where FT = FT

Base.size(grid::AbstractGrid2D) = (grid.nx, grid.ny)
Base.length(grid::AbstractGrid2D) = (grid.Lx, grid.Ly)

Base.show(io::IO, g::Grid2D) =
      print(io, "Grid2D\n",
         "  ├──────  resolution: $(g.nx)×$(g.ny)\n",
         "  ├───────  mesh size: $(g.nx*g.ny)\n",
         "  ├────  grid spacing: $(g.Δx)×$(g.Δy)\n",
         "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]")

Base.size(grid::AbstractGrid3D) = (grid.nx, grid.ny, grid.nz)
Base.length(grid::AbstractGrid3D) = (grid.Lx, grid.Ly, grid.Lz)

Base.show(io::IO, g::Grid3D) =
     print(io, "Grid3D\n",
         "  ├──────  resolution: $(g.nx)×$(g.ny)×$(g.nz)\n",
         "  ├───────  mesh size: $(g.nx*g.ny*g.nz)\n",
         "  ├────  grid spacing: $(g.Δx)×$(g.Δy)×$(g.Δz)\n",
         "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]×[$(g.zmin),$(g.zmax)]")