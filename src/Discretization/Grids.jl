"Abstract supertype for grids."
abstract type AbstractGrid{FT<:Real} end
"Abstract supertype for 2D grids."
abstract type AbstractGrid2D{FT<:Real} <: AbstractGrid{FT} end
"Abstract supertype for 3D grids."
abstract type AbstractGrid3D{FT<:Real} <: AbstractGrid{FT} end

"""
    Grid2D{FT<:Real} <: AbstractGrid2D{FT}

Type representing a 2D grid.
"""
struct Grid2D{FT<:Real} <: AbstractGrid2D{FT}
   "x range"
   x :: Array{FT}
   "y range"
   y :: Array{FT}
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
   "ξx x frequencies"
   ξx :: Array{FT}
   "ξy y frequencies"
   ξy :: Array{FT}
end

"""
    Grid2D{FT<:Real} <: AbstractGrid2D{FT}

Type representing a 2D grid.
"""
struct Grid2DGPU{FT<:Real} <: AbstractGrid2D{FT}
   "x range"
   x :: CuArray{FT}
   "y range"
   y :: CuArray{FT}
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
   "ξx x frequencies"
   ξx :: CuArray{FT}
   "ξy y frequencies"
   ξy :: CuArray{FT}
end

"""
    Grid3D{FT<:Real} <: AbstractGrid3D{FT}

Type representing a grid.
"""
struct Grid3D{FT} <: AbstractGrid3D{FT}
   # range
   x :: Array{FT}
   y :: Array{FT}
   z :: Array{FT}
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
   # Frequencies
   ξx :: Array{FT}
   ξy :: Array{FT}
   ξz :: Array{FT}
end

struct Grid3DGPU{FT} <: AbstractGrid3D{FT}
   # range
   x :: CuArray{FT}
   y :: CuArray{FT}
   z :: CuArray{FT}
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
   # Frequencies
   ξx :: CuArray{FT}
   ξy :: CuArray{FT}
   ξz :: CuArray{FT}
end

"""
    Grid3D(size::Tuple{Real,Real,Real},xbounds::Tuple{Real,Real},ybounds::Tuple{Real,Real},zbounds::Tuple{Real,Real},FT=Float64)

Returns a Grid3D with of size `size = (nx,ny,nz)` ranging from `xbounds` × `ybounds` × `zbounds`, or Grid2D if `nz = 1`.

Example
=======
```
julia> conf = ConfParse("GPS_input.init")
julia> parse_conf!(conf)
julia> grid = Grid(conf)
```
"""
function Grid(conf::AbstractConfig;FT=Float64,D=CPU())
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
   # frequencies
   ξx = fftfreq(nx,2π/Δx)
   ξy = fftfreq(ny,2π/Δy)
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
      # frequencies
      ξz = fftfreq(nz,2π/Δz)
      # reshape for broadcast
      X = reshape(x[1:end-1],nx,1,1)
      Y = reshape(y[1:end-1],1,ny,1)
      Z = reshape(z[1:end-1],1,1,nz)
      Ξx = reshape(ξx,nx,1,1)
      Ξy = reshape(ξy,1,ny,1)
      Ξz = reshape(ξz,1,1,nz)
      if typeof(D) == CPU
         return Grid3D{FT}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz, Ξx, Ξy, Ξz)
      elseif typeof(D) == GPU
         X = CuArray(X)
         Y = CuArray(Y)
         Z = CuArray(Z)
         Ξx = CuArray(Ξx)
         Ξy = CuArray(Ξy)
         Ξz = CuArray(Ξz)
         return Grid3DGPU{FT}(X, Y, Z, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, Lx, Ly, Lz, Δx, Δy, Δz, Ξx, Ξy, Ξz)
      end
   else
      X = reshape(x[1:end-1],nx,1)
      Y = reshape(y[1:end-1],1,ny)
      Ξx = reshape(ξx,nx,1)
      Ξy = reshape(ξy,1,ny)
      if typeof(D) == CPU
         return Grid2D{FT}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy, Ξx, Ξy)
      elseif typeof(D) == GPU
         X = CuArray(X)
         Y = CuArray(Y)
         Ξx = CuArray(Ξx)
         Ξy = CuArray(Ξy)
         return Grid2DGPU{FT}(X, Y, nx, ny, xmin, xmax, ymin, ymax, Lx, Ly, Δx, Δy, Ξx, Ξy)
      end
   end
end

Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.size(grid::AbstractGrid) = (grid.nx, grid.ny, grid.nz)
Base.length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)

Base.show(io::IO, g::Grid2D) =
     print(io, "Grid2D\n",
         "  ├──────  resolution: $(g.nx)×$(g.ny)\n",
         "  ├───────  mesh size: $(g.nx*g.ny)\n",
         "  ├────  grid spacing: $(g.Δx)×$(g.Δy)\n",
         "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]")

Base.show(io::IO, g::Grid3D) =
     print(io, "Grid3D\n",
         "  ├──────  resolution: $(g.nx)×$(g.ny)×$(g.nz)\n",
         "  ├───────  mesh size: $(g.nx*g.ny*g.nz)\n",
         "  ├────  grid spacing: $(g.Δx)×$(g.Δy)×$(g.Δz)\n",
         "  └──────────  domain: [$(g.xmin),$(g.xmax)]×[$(g.ymin),$(g.ymax)]×[$(g.zmin),$(g.zmax)]")
