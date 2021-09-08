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
abstract type AbstractField{A,G} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{A,G} <: AbstractField{A,G} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{A,G} <: AbstractField{A,G} end

"""
    Field2D{A,G<:AbstractGrid} <: AbstractField2D{A,G}

Type representing a 2D field on a 2D grid.
"""
mutable struct Field2D{A,G} <: AbstractField2D{A,G}
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: A
end

"""
    Field3D{A,G<:AbstractGrid} <: AbstractField3D{A,G}

Type representing a 3D field on a 3D grid.
"""
mutable struct Field3D{A,G} <: AbstractField3D{A,G}
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: A
end

"""
    Field(g::AbstractGrid2D{FT,A},::RealField) where {FT<:Real,A<:Array}

Returns a 2D field with real values.

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
function Field(g::AbstractGrid2D{FT,A},t::FieldType;ndims=1) where {FT<:Real,A}
   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end
   if typeof(A) == Array
      myArray = Array
   elseif typeof(A) == CuArray
      myArray = CuArray
   end
   if ndims == 1
      ϕ = Array{myT}(undef,g.nx,g.ny)
   else
      ϕ = Array{myT}(undef,g.nx,g.ny,ndims)
   end
   return Field2D{typeof(ϕ),typeof(g)}(g,ϕ)
end

"""
    Field(g::AbstractGrid3D{FT,A},::RealField) where {FT<:Real,A<:Array}

Returns a 3D field with real values.

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
function Field(g::AbstractGrid3D{FT,A},::RealField) where {FT<:Real,A}
   if typeof(t) == RealField
      myT = FT
   elseif typeof(t) == ComplexField
      myT = Complex{FT}
   end
   if typeof(A) == Array
      myArray = Array
   elseif typeof(A) == CuArray
      myArray = CuArray
   end
   if ndims == 1
      ϕ = myArray{myT}(undef,g.nx,g.ny)
   else
      ϕ = myArray{myT}(undef,g.nx,g.ny,g.nz,ndims)
   end
   return Field3D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function norm(f::Field2D)
   normϕ = real(sum(f.ϕ.*conj(f.ϕ)))
   normϕ = sqrt(normϕ) * sqrt(f.g.Δx*f.g.Δy)
end

function norm(f::Field3D)
   normϕ = real(sum(f.ϕ.*conj.(f.ϕ)))
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
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*16/1024^2) MB")

Base.show(io::IO, f::Field3D{A}) where A =
     print(io, "Field3D\n",
         "  ├───────  FloatType: $(A)", '\n', 
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*16/1024^2) MB")
