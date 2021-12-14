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
abstract type AbstractField{A,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{A,G,D} <: AbstractField{A,G,D} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{A,G,D} <: AbstractField{A,G,D} end

"""
    Field2D{A,G<:AbstractGrid} <: AbstractField2D{A,G}

Type representing a 2D field on a 2D grid.
"""
mutable struct Field2D{A,G,D} <: AbstractField2D{A,G,D}
   "decomposition"
   decomp :: D
   "number of dimensions"
   ndims :: Integer
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: A
end

"""
    Field3D{A,G<:AbstractGrid,D} <: AbstractField3D{A,G,D}

Type representing a 3D field on a 3D grid.
"""
mutable struct Field3D{A,G,D} <: AbstractField3D{A,G,D}
   "decomposition"
   decomp :: D
   "local x"
   x :: AbstractArray
   "local y"
   y :: AbstractArray
   "local z"
   z :: AbstractArray
   "number of dimensions"
   ndims :: Integer
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: A
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
# function Field(g::AbstractGrid2D{FT,A,D},t::FieldType;ndims::Integer=1) where {FT<:Real,A,D}
#    if typeof(t) == RealField
#       myT = FT
#    elseif typeof(t) == ComplexField
#       myT = Complex{FT}
#    end
#    if A <: Array
#       myArray = Array
#    elseif A <: CuArray
#       myArray = CuArray
#    end
#    if ndims == 1
#       ϕ = myArray{myT}(undef,g.nx,g.ny)
#    else
#       ϕ = myArray{myT}(undef,g.nx,g.ny,ndims)
#    end
#    return Field2D{typeof(ϕ),typeof(g)}(ndims,g,ϕ)
# end

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
   ) where {FT<:Real,A}
   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end
   if A <: Array
      myArray = Array
   elseif A <: CuArray
      myArray = CuArray
   end
   if ndims == 1
      dims = (g.nx,g.ny,g.nz)
   else
      dims = (g.nx,g.ny,g.nz,ndims)
   end
   if typeof(mpi_topo) <: AbstractNoMPIDecomposition
      ϕ = myArray{myT}(undef, dims...)
      r = axes(ϕ)
      x = g.x
      y = g.y
      z = g.z
      decomp = NoFieldDecomposition(dd, r, nl)
   elseif typeof(mpi_topo) <: AbstractMPIDecomposition
      pen_x = Pencil(mpi_topo.topo, dims, (2,3))
      nl = size_local(pen_x)
      ϕ = PencilArray(pen_x, myArray{myT}(undef, dims))
      #ϕ = PencilArray{myT}(undef, pen_x)
      ϕglob = global_view(ϕ)
      r = axes(ϕglob)
      @assert size_local(ϕ) == nl

      # function PencilArray(pencil::Pencil{Np, Mp} where {Np, Mp},
      #    data::AbstractArray{T, N}) where {T, N}
      # dims = (size_local(pencil, MemoryOrder())..., extra_dims...)
      # PencilArray(pencil, Array{T}(init, dims))

      x = reshape(g.x[r[1]],nl[1],1,1)
      y = reshape(g.y[r[2]],1,nl[2],1)
      z = reshape(g.z[r[3]],1,1,nl[3])
      decomp = MPIFieldDecomposition(mpi_topo, pen_x, r, nl)
   end
   return Field3D{typeof(ϕ),typeof(g),typeof(decomp)}(decomp, x, y, z, ndims, g, ϕ)
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

Base.show(io::IO, f::Field2D{A}) where A =
     print(io, "Field2D\n",
         "  ├──────  Array type: $(A)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")

Base.show(io::IO, f::Field3D{A}) where A =
     print(io, "Field3D\n",
         "  ├───────  FloatType: $(A)", '\n',
         "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")