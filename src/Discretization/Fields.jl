using Base: @propagate_inbounds

"Abstract supertype for field type."
abstract type FieldType end
"Complex field"
struct ComplexField <: FieldType end
"Real field"
struct RealField <: FieldType end

export ComplexField, RealField
export Field

"Abstract supertype for numerical models."
abstract type AbstractField{FT,FFT,A,PA,G,P} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{FT,FFT,A,PA,G,P} <: AbstractField{FT,FFT,A,PA,G,P} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{FT,FFT,A,PA,G,P} <: AbstractField{FT,FFT,A,PA,G,P} end

"""
    Field2D{FT,FFT,A,PA,G,P} <: AbstractField2D{FT,FFT,A,PA,G,P}

Type representing a 2D field on a 2D grid.

- `pen`: informations concerning the decomposition.
- `x`, `y`: local grid (relative to the `ϕ` decomposition).
- `ndims`: number of dimension (additional) of the field.
- `g`: reference to the grid.
- `data`: distributed containing the data.

"""
mutable struct Field2D{FT,FFT,A,PA,G,P} <: AbstractField2D{FT,FFT,A,PA,G,P}
   pen :: P
   x :: LocalGrids.RectilinearGridComponent
   y :: LocalGrids.RectilinearGridComponent
   ndims :: Integer
   g :: G
   data :: Vector{PA}
end

"""
    Field3D{FT,FFT,A,PA,G,P} <: AbstractField3D{FT,FFT,A,PA,G,P}

Type representing a 3D field on a 3D grid.

- `pen`: informations concerning the decomposition.
- `x`, `y`, `z`: local grid (relative to the `ϕ` decomposition).
- `ndims`: number of dimension (additional) of the field.
- `g`: reference to the grid.
- `data`: distributed containing the data.

"""
mutable struct Field3D{FT,FFT,A,PA,G,P} <: AbstractField3D{FT,FFT,A,PA,G,P}
   pen :: P
   x :: LocalGrids.RectilinearGridComponent
   y :: LocalGrids.RectilinearGridComponent
   z :: LocalGrids.RectilinearGridComponent
   ndims :: Integer
   g :: G
   data :: Vector{PA}
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

   dims = (g.nx,g.ny)

   pen_x = Pencil(A, mpi_topo.topo, dims, (2,))
   local_dims = size_local(pen_x)
   data = [PencilArray(pen_x, A{myT}(undef, local_dims))]
   for i = 1:ndims-1
      push!(data, PencilArray(pen_x, A{myT}(undef, local_dims)))
   end

   pen_array_glob = global_view(data[1])
   r = Tuple([ minimum(a):maximum(a) for a in axes(pen_array_glob) ])

   grid = localgrid(pen_x, (g.x,g.y))
   x, y = grid.x, grid.y

   return Field2D{FT,myT,A,typeof(data[1]),typeof(g),typeof(pen_x)}(pen_x, x, y, ndims, g, data)
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

   dims = (g.nx,g.ny,g.nz)
   pen_x = Pencil(A, mpi_topo.topo, dims, (2,3))
   local_dims = size_local(pen_x)
   data = [PencilArray(pen_x, A{myT}(undef, local_dims))]
   for i = 1:ndims-1
      push!(data, PencilArray(pen_x, A{myT}(undef, local_dims)))
   end

   pen_array_glob = global_view(data[1])
   r = Tuple([ minimum(a):maximum(a) for a in axes(pen_array_glob) ])

   grid = localgrid(pen_x, (g.x,g.y,g.z))
   x, y, z = grid.x, grid.y, grid.z

   return Field3D{FT,myT,A,typeof(data[1]),typeof(g),typeof(pen_x)}(pen_x, x, y, z, ndims, g, data)
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

@inline function Base.getproperty(f::AbstractField, name::Symbol)
   if name === :ϕ
      f.data[1]
   elseif name === :vx
      f.data[1]
   elseif name === :vy
      f.data[2]
   elseif name === :vz
      f.data[3]
   else
      getfield(f, name)
   end
end

function similar_data(data::Vector{A}) where A
   newdata = [similar(data[1])]
   for i = 1:length(data)-1
      push!(newdata, similar(data[1]))
   end
   return newdata
end

Base.show(io::IO, f::Field2D{FT,FFT}) where {FT,FFT} =
      print(io, "Field2D\n",
         "  ├──────  Array type: $(FFT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")

Base.show(io::IO, f::Field3D{FT,FFT}) where {FT,FFT} =
      print(io, "Field3D\n",
         "  ├───────  FloatType: $(FFT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ.data)/1024^2) MB")
