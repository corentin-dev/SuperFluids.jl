"Abstract supertype for field type."
abstract type FieldType end
"Complex field"
struct ComplexField <: FieldType end
"Real field"
struct RealField <: FieldType end

export ComplexField, RealField
export Field

"Abstract supertype for numerical models."
abstract type AbstractField{FT,FFT,A,PA,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{FT,FFT,A,PA,G,D} <: AbstractField{FT,FFT,A,PA,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{FT,FFT,A,PA,G,D} <: AbstractField{FT,FFT,A,PA,G,D} end

"""
    Field2D{FT,FFT,A,PA,G,D} <: AbstractField2D{FT,FFT,A,PA,G,D}

Type representing a 2D field on a 2D grid.

- `decomp`: informations concerning the decomposition.
- `x`, `y`: local grid (relative to the `ϕ` decomposition).
- `ndims`: number of dimension (additional) of the field.
- `g`: reference to the grid.
- `ϕ`: distributed containing the data.

"""
mutable struct Field2D{FT,FFT,A,PA,G,D} <: AbstractField2D{FT,FFT,A,PA,G,D}
   "decomposition"
   decomp :: D
   "local x"
   x :: LocalGrids.RectilinearGridComponent
   "local y"
   y :: LocalGrids.RectilinearGridComponent
   "number of dimensions"
   ndims :: Integer
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: PA
end

"""
    Field3D{FT,FFT,A,PA,G,D} <: AbstractField3D{FT,FFT,A,PA,G,D}

Type representing a 3D field on a 3D grid.

- `decomp`: informations concerning the decomposition.
- `x`, `y`, `z`: local grid (relative to the `ϕ` decomposition).
- `ndims`: number of dimension (additional) of the field.
- `g`: reference to the grid.
- `ϕ`: distributed containing the data.

"""
mutable struct Field3D{FT,FFT,A,PA,G,D} <: AbstractField3D{FT,FFT,A,PA,G,D}
   "decomposition"
   decomp :: D
   "local x"
   x :: LocalGrids.RectilinearGridComponent
   "local y"
   y :: LocalGrids.RectilinearGridComponent
   "local z"
   z :: LocalGrids.RectilinearGridComponent
   "number of dimensions"
   ndims :: Integer
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: PA
end

"""
    Field(g::AbstractGrid2D{FT,A,D},::RealField) where {FT<:Real,A<:Array,D}

Returns a 2D field.

Example
=======
```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, RealField())
Field2D
  ├──────  Array type: Matrix{Float64}
  └──────────  memory: 0.5 MB
```

```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField())
Field2D
  ├──────  Array type: Matrix{ComplexF64}
  └──────────  memory: 0.5 MB
```
"""
function Field(
      g::AbstractGrid2D{FT,A}, t::FieldType;
      ndims::Integer=1, mpi_topo::AbstractDomainDecomposition=MPITopo1D()
   ) where {FT<:Real, A}

   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end

   if ndims == 1
      dims = (g.nx,g.ny)
   else
      dims = (g.nx,g.ny,ndims)
   end

   pen_x = Pencil(A, mpi_topo.topo, dims)
   local_dims = size_local(pen_x)
   pen_array = PencilArray(pen_x, A{myT}(undef, local_dims))
   ϕ = pen_array

   pen_array_glob = global_view(pen_array)
   r = Tuple([ minimum(a):maximum(a) for a in axes(pen_array_glob) ])

   grid = localgrid(pen_x, (g.x,g.y))
   x, y = grid.x, grid.y

   decomp = MPIFieldDecomposition(mpi_topo, pen_array, r, local_dims)
   return Field2D{FT,myT,A,typeof(ϕ),typeof(g),typeof(decomp)}(decomp, x, y, ndims, g, ϕ)
end

"""
    Field(g::AbstractGrid3D{FT,A,D},t::FieldType;ndims::Integer=1) where {FT<:Real,A,D}

Returns a 3D field.

Example
=======
```jldoctest
julia> grid = Grid((128,128,128), ((-12,12), (-12,12), (-12,12)));
julia> field = Field(grid, RealField())
Field3D
  ├───────  FloatType: Float64
  └──────────  memory: 64.0 MB
```

```jldoctest
julia> grid = Grid((128,128,128), ((-12,12), (-12,12), (-12,12)));
julia> field = Field(grid, ComplexField())
Field3D
  ├───────  FloatType: ComplexF64
  └──────────  memory: 64.0 MB
```
"""
function Field(
      g::AbstractGrid3D{FT,A}, t::FieldType;
      ndims::Integer=1, mpi_topo::AbstractDomainDecomposition=MPITopo2D()
   ) where {FT<:Real, A}

   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end

   if ndims == 1
      dims = (g.nx,g.ny,g.nz)
   else
      dims = (g.nx,g.ny,g.nz,ndims)
   end

   pen_x = Pencil(A, mpi_topo.topo, dims, (2,3))
   local_dims = size_local(pen_x)
   pen_array = PencilArray(pen_x, A{myT}(undef, local_dims))
   ϕ = pen_array

   pen_array_glob = global_view(pen_array)
   r = Tuple([ minimum(a):maximum(a) for a in axes(pen_array_glob) ])

   grid = localgrid(pen_x, (g.x,g.y,g.z))
   x, y, z = grid.x, grid.y, grid.z

   decomp = MPIFieldDecomposition(mpi_topo, pen_array, r, local_dims)
   return Field3D{FT,myT,A,typeof(ϕ),typeof(g),typeof(decomp)}(decomp, x, y, z, ndims, g, ϕ)
end

function norm(f::Field2D)
   normϕ = sum(abs2.(f.ϕ))
   normϕ = sqrt(normϕ) * sqrt(f.g.Δx*f.g.Δy)
end

function norm(f::Field3D)
   normϕ = sum(abs2.(f.ϕ))
   normϕ = sqrt(normϕ) * sqrt(f.g.Δx*f.g.Δy*f.g.Δz)
end

"""
    normalize!(f::AbstractField)

Normalize a field.

Example
=======
```jldoctest
julia> grid = Grid((128,128,128), ((-12,12), (-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> field.ϕ .= 2;
julia> norm(field)
235.15101530718513
julia> normalize!(field)
julia> norm(field)
1.000000000000001
```
"""
function normalize!(f::AbstractField)
   normϕ = norm(f)
   f.ϕ ./= normϕ
   return nothing
end

Base.show(io::IO, f::Field2D{FT,FFT}) where {FT,FFT} =
      print(io, "Field2D\n",
         "  ├──────  Array type: $(FFT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")

Base.show(io::IO, f::Field3D{FT,FFT}) where {FT,FFT} =
      print(io, "Field3D\n",
         "  ├───────  FloatType: $(FFT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ.data)/1024^2) MB")