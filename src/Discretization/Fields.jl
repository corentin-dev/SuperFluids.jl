export AbstractField, AbstractField2D, AbstractField3D

"Abstract supertype for field type."
abstract type FieldType end
"Complex field"
struct ComplexField <: FieldType end
"Real field"
struct RealField <: FieldType end

"Abstract supertype for numerical models."
abstract type AbstractField{FT,G} end
"Abstract supertype for numerical models."
abstract type AbstractField2D{FT,G} <: AbstractField{FT,G} end
"Abstract supertype for numerical models."
abstract type AbstractField3D{FT,G} <: AbstractField{FT,G} end

"""
    Field2D{FT<:Number,G<:AbstractGrid} <: AbstractField2D{FT,G}

Type representing a 2D field on a 2D grid.
"""
struct Field2D{FT<:Number,G} <: AbstractField2D{FT,G}
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: Array{FT}
end

"""
    Field3D{FT<:Number,G<:AbstractGrid} <: AbstractField3D{FT,G}

Type representing a 3D field on a 3D grid.
"""
struct Field3D{FT,G} <: AbstractField3D{FT,G}
   "g Grid"
   g :: G
   "ϕ array containing data"
   ϕ :: Array{FT}
end

function Field(g::AbstractGrid2D{FT},::RealField) where {FT<:Real}
   ϕ = Array{FT}(undef,g.nx,g.ny)
   return Field2D{FT,AbstractGrid2D{FT}}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT},::RealField) where {FT<:Real}
   ϕ = Array{FT}(undef,g.nx,g.ny,g.nz)
   return Field3D{FT,AbstractGrid3D{FT}}(g,ϕ)
end

function Field(g::AbstractGrid2D{FT},::ComplexField) where {FT<:Real}
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny)
   return Field2D{Complex{FT},AbstractGrid2D{FT}}(g,ϕ)
end

function Field(g::AbstractGrid3D{FT},::ComplexField) where {FT<:Real}
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny,g.nz)
   return Field3D{Complex{FT},AbstractGrid3D{FT}}(g,ϕ)
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

Base.eltype(g::Field3D{FT,G}) where {FT,G} = FT

Base.show(io::IO, f::Field2D{FT}) where FT =
     print(io, "Field2D\n",
         "  ├───────  FloatType: $(FT)", '\n', 
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*sizeof(FT)/1024^2) MB")

Base.show(io::IO, f::Field3D{FT}) where FT =
     print(io, "Field3D\n",
         "  ├───────  FloatType: $(FT)", '\n', 
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*sizeof(FT)/1024^2) MB")
