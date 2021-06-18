export AbstractField, AbstractField2D, AbstractField3D

"Abstract supertype for field type."
abstract type FieldType end
"Complex field"
struct ComplexField <: FieldType end
"Real field"
struct RealField <: FieldType end

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

function Field(g::AbstractGrid2D{FT,A},::RealField) where {FT<:Real,A<:Array}
   ϕ = Array{FT}(undef,g.nx,g.ny)
   return Field2D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT,A},::RealField) where {FT<:Real,A<:Array}
   ϕ = Array{FT}(undef,g.nx,g.ny,g.nz)
   return Field3D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid2D{FT,A},::ComplexField) where {FT<:Real,A<:Array}
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny)
   return Field2D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT,A},::ComplexField) where {FT<:Real,A<:Array}
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny,g.nz)
   return Field3D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid2D{FT,A},::RealField) where {FT<:Real,A<:CuArray}
   ϕ = CuArray{FT}(undef,g.nx,g.ny)
   return Field2D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT,A},::RealField) where {FT<:Real,A<:CuArray}
   ϕ = CuArray{FT}(undef,g.nx,g.ny,g.nz)
   return Field3D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid2D{FT,A},::ComplexField) where {FT<:Real,A<:CuArray}
   ϕ = CuArray{Complex{FT}}(undef,g.nx,g.ny)
   return Field2D{typeof(ϕ),typeof(g)}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT,A},::ComplexField) where {FT<:Real,A<:CuArray}
   ϕ = CuArray{Complex{FT}}(undef,g.nx,g.ny,g.nz)
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
