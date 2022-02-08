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
"""
struct Grid2D{FT<:Real,A} <: AbstractGrid2D{FT,A}
   "x range"
   x :: A
   "y range"
   y :: A
   "nx size in x direction"
   nx :: Integer
   "ny size in y direction"
   ny :: Integer
   "xmin minimum x boundary"
   xmin :: FT
   "xmax maximum x boundary"
   xmax :: FT
   "ymin minimum y boundary"
   ymin :: FT
   "ymax maximum y boundary"
   ymax :: FT
   "Lx x length"
   Lx :: FT
   "Ly y length"
   Ly :: FT
   "Δx x discretization"
   Δx :: FT
   "Δy y discretization"
   Δy :: FT
end

"""
    Grid3D{FT<:Real} <: AbstractGrid3D{FT}

Type representing a grid.
"""
struct Grid3D{FT,A} <: AbstractGrid3D{FT,A}
   "x range"
   x :: A
   "y range"
   y :: A
   "z range"
   z :: A
   "nx size in x direction"
   nx :: Integer
   "ny size in y direction"
   ny :: Integer
   "nz size in z direction"
   nz :: Integer
   "xmin minimum x boundary"
   xmin :: FT
   "xmax maximum x boundary"
   xmax :: FT
   "ymin minimum y boundary"
   ymin :: FT
   "ymax maximum y boundary"
   ymax :: FT
   "zmin minimum z boundary"
   zmin :: FT
   "zmax maximum z boundary"
   zmax :: FT
   "Lx x length"
   Lx :: FT
   "Ly y length"
   Ly :: FT
   "Lz z length"
   Lz :: FT
   "Δx x discretization"
   Δx :: FT
   "Δy y discretization"
   Δy :: FT
   "Δz z discretization"
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
   # reshaping arrays for broadcast
   x = array_type(x[1:end-1])
   y = array_type(y[1:end-1])
   return Grid2D{FT,array_type}(device, x, x, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
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