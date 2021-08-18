"Abstract supertype for grids."
abstract type AbstractGrid{FT<:Real,A} end
"Abstract supertype for 2D grids."
abstract type AbstractGrid2D{FT<:Real,A} <: AbstractGrid{FT,A} end
"Abstract supertype for 3D grids."
abstract type AbstractGrid3D{FT<:Real,A} <: AbstractGrid{FT,A} end

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
   # range
   x :: A
   y :: A
   z :: A
   # Size
   nx :: Integer
   ny :: Integer
   nz :: Integer
   # Bounds
   xmin :: FT
   xmax :: FT
   ymin :: FT
   ymax :: FT
   zmin :: FT
   zmax :: FT
   # Length.
   Lx :: FT
   Ly :: FT
   Lz :: FT
   # Discretization.
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
```
julia> grid = Grid((128,128), ((-12,12),(-12,12)))
```
"""
function Grid(size::Tuple{Real,Real},
      bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real}};
      FT=Float64,device=CPU())
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
   X = reshape(x[1:end-1],nx,1)
   Y = reshape(y[1:end-1],1,ny)
   if typeof(device) == CPU
      return Grid2D{FT,Array{FT,2}}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
   elseif typeof(device) == GPU
      X = CuArray(X)
      Y = CuArray(Y)
      return Grid2D{FT,typeof(X)}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
   end
end

"""
    Grid(size::Tuple{Integer,Integer,Integer},
      bounds::Tuple{Tuple{Real,Real,Real},Tuple{Real,Real,Real},Tuple(Real,Real,Real)};
      FT=Float64,device=CPU())

Returns a Grid3D with of size `size = (nx,ny,nz)` ranging from `bounds = ((xmin,xmax),(ymin,ymax),(zmin,zmax))` .

Example
=======
```
julia> grid = Grid((128,128,128), ((-12,12),(-12,12),(-12,12)))
```
"""
function Grid(size::Tuple{Integer,Integer,Integer},
      bounds::Tuple{Tuple{Real,Real},Tuple{Real,Real},Tuple{Real,Real}};
      FT=Float64,device=CPU())
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
   # reshaping arrays for broadcast
   X = reshape(x[1:end-1],nx,1,1)
   Y = reshape(y[1:end-1],1,ny,1)
   Z = reshape(z[1:end-1],1,1,nz)
   if typeof(device) == CPU
      return Grid3D{FT,Array{FT,3}}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz)
   elseif typeof(device) == GPU
      X = CuArray(X)
      Y = CuArray(Y)
      Z = CuArray(Z)
      return Grid3D{FT,typeof(X)}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz)
   end
end

"""
    Grid(conf::AbstractConfig;FT=Float64,device=CPU())

Returns a Grid3D with of size `size = (nx,ny,nz)` ranging from `xbounds` × `ybounds` × `zbounds`, or Grid2D if `nz = 1`.

Example
=======
```
julia> conf = Config("GPS_input.init")
julia> grid = Grid(conf)
```
"""
function Grid(conf::AbstractConfig;FT=Float64,device=CPU())
   nx = retrieve(conf, "discretization", "nx", Int64)
   ny = retrieve(conf, "discretization", "ny", Int64)
   nz = retrieve(conf, "discretization", "nz", Int64)
   # bounds
   xmin = retrieve(conf, "geometry", "xmin", FT)
   xmax = retrieve(conf, "geometry", "xmax", FT)
   ymin = retrieve(conf, "geometry", "ymin", FT)
   ymax = retrieve(conf, "geometry", "ymax", FT)
   # length
   Lx = xmax - xmin
   Ly = ymax - ymin
   # discretization
   x = LinRange(xmin,xmax,nx+1)
   y = LinRange(xmin,xmax,ny+1)
   # spacing
   Δx = Lx / nx
   Δy = Ly / ny
   if nz > 1
      # bounds
      zmin = retrieve(conf, "geometry", "zmin", FT)
      zmax = retrieve(conf, "geometry", "zmax", FT)
      # length
      Lz = zmax - zmin
      # discretization
      z = LinRange(zmin,zmax,nz+1)
      # spacing
      Δz = Lz / nz
      # reshape for broadcast
      X = reshape(x[1:end-1],nx,1,1)
      Y = reshape(y[1:end-1],1,ny,1)
      Z = reshape(z[1:end-1],1,1,nz)
      if typeof(device) == CPU
         return Grid3D{FT,Array{FT,3}}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz)
      elseif typeof(device) == GPU
         X = CuArray(X)
         Y = CuArray(Y)
         Z = CuArray(Z)
         return Grid3D{FT,typeof(X)}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz)
      end
   else
      X = reshape(x[1:end-1],nx,1)
      Y = reshape(y[1:end-1],1,ny)
      if typeof(device) == CPU
         return Grid2D{FT,Array{FT,2}}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
      elseif typeof(device) == GPU
         X = CuArray(X)
         Y = CuArray(Y)
         return Grid2D{FT,typeof(X)}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy)
      end
   end
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
