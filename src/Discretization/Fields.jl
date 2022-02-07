"Abstract supertype for field type."
abstract type FieldType end
"Complex field"
struct ComplexField <: FieldType end
"Real field"
struct RealField <: FieldType end

export ComplexField, RealField
export Field
export norm, normalize!

"Abstract supertype for numerical models."
abstract type AbstractField{FT,A,PA,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{FT,A,PA,G,D} <: AbstractField{FT,A,PA,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{FT,A,PA,G,D} <: AbstractField{FT,A,PA,G,D} end

"""
    Field2D{A,G<:AbstractGrid} <: AbstractField2D{A,G}

Type representing a 2D field on a 2D grid.
"""
mutable struct Field2D{FT,A,PA,G,D} <: AbstractField2D{FT,A,PA,G,D}
   "decomposition"
   decomp :: D
   "local x"
   x :: A
   "local y"
   y :: A
   "number of dimensions"
   ndims :: Integer
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: PA
end

"""
    Field3D{A,G<:AbstractGrid,D} <: AbstractField3D{A,G,D}

Type representing a 3D field on a 3D grid.
"""
mutable struct Field3D{FT,A,PA,G,D} <: AbstractField3D{FT,A,PA,G,D}
   "decomposition"
   decomp :: D
   "local x"
   x :: A
   "local y"
   y :: A
   "local z"
   z :: A
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
      ndims::Integer=1, mpi_topo::AbstractDomainDecomposition=MPINone()
   ) where {FT<:Real, A}
   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end
   myArray = get_array_type(A)
   if ndims == 1
      dims = (g.nx,g.ny)
   else
      dims = (g.nx,g.ny,ndims)
   end
   ϕ = myArray{myT}(undef, dims...)
   r = axes(ϕ)
   x = g.x
   y = g.y
   decomp = NoFieldDecomposition(dd, r, nl)
   return Field2D{myT,A,typeof(ϕ),typeof(g),typeof(decomp)}(decomp, x, y, ndims, g, ϕ)
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
  ├───────  FloatType: Array{Float64, 3}
  └──────────  memory: 64.0 MB
```

```jldoctest
julia> grid = Grid((128,128,128), ((-12,12), (-12,12), (-12,12)));
julia> field = Field(grid, ComplexField())
Field3D
  ├───────  FloatType: Array{ComplexF64, 3}
  └──────────  memory: 64.0 MB
```
"""
function Field(
      g::AbstractGrid3D{FT,A}, t::FieldType;
      ndims::Integer=1, mpi_topo::AbstractDomainDecomposition=MPINone()
   ) where {FT<:Real, A}

   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end

   myArray = get_array_type(A)
   if ndims == 1
      dims = (g.nx,g.ny,g.nz)
   else
      dims = (g.nx,g.ny,g.nz,ndims)
   end

   pen_x = Pencil(myArray, mpi_topo.topo, dims, (2,3))
   local_dims = size_local(pen_x)
   pen_array = PencilArray(pen_x, myArray{myT}(undef, local_dims))
   ϕ = pen_array#.data

   pen_array_glob = global_view(pen_array)
   r = Tuple([ minimum(a):maximum(a) for a in axes(pen_array_glob) ])

   grid = localgrid(pen_x, (g.x,g.y,g.z))
   x, y, z = grid.x, grid.y, grid.z

   decomp = MPIFieldDecomposition(mpi_topo, pen_array, r, local_dims)

   return Field3D{myT,Any,typeof(ϕ),typeof(g),typeof(decomp)}(decomp, x, y, z, ndims, g, ϕ)
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

function get_array_type(A::DataType)
   if A <: Array
      myArray = Array
   elseif A <: CuArray
      myArray = CuArray
   end
   return myArray
end

function get_array_type(a::A) where (A<:AbstractArray)
   if A <: Array
      myArray = Array
   elseif A <: CuArray
      myArray = CuArray
   elseif A<:PencilArray
      if typeof(a.data) <: Array
         myArray = Array
      elseif typeof(a.data) <: CuArray
         myArray = CuArray
      end
   end
   return myArray
end

get_type_array(A::AbstractArray{T}) where T = T

get_real_type_array(A::AbstractArray{T}) where T = T

get_real_type_array(A::AbstractArray{Complex{T}}) where T = T

Base.show(io::IO, f::Field2D{FT}) where FT =
     print(io, "Field2D\n",
         "  ├──────  Array type: $(FT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")

Base.show(io::IO, f::Field3D{FT}) where FT =
     print(io, "Field3D\n",
         "  ├───────  FloatType: $(FT)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")