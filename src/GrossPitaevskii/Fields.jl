using AbstractFFTs: fftfreq

abstract type AbstractField{FT,G} end
abstract type AbstractField2D{FT,G} <: AbstractField{FT,G} end
abstract type AbstractField3D{FT,G} <: AbstractField{FT,G} end

struct Field2D{FT,G} <: AbstractField2D{FT,G}
   g :: G
   ϕ :: Array{Complex{FT}}
   ϕhat :: Array{Complex{FT}}
end

struct Field3D{FT,G} <: AbstractField3D{FT,G}
   g :: G
   ϕ :: Array{Complex{FT}}
   ϕhat :: Array{Complex{FT}}
end

function Field(g::AbstractGrid2D{FT}) where FT
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny)
   ϕ_hat = similar(ϕ)
   return Field2D{FT,AbstractGrid2D{FT}}(g,ϕ,ϕ_hat)
end

function Field(g::AbstractGrid3D{FT}) where FT
   ϕ = Array{Complex{FT}}(undef,g.nx,g.ny,g.nz)
   ϕ_hat = similar(ϕ)
   return Field3D{FT,AbstractGrid3D{FT}}(g,ϕ,ϕ_hat)
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
         "  ├───────  FloatType: $(Complex{FT})", '\n', 
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*sizeof(Complex{FT})/1024^2) MB")

Base.show(io::IO, f::Field3D{FT}) where FT =
     print(io, "Field3D\n",
         "  ├───────  FloatType: $(Complex{FT})", '\n', 
         "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*sizeof(Complex{FT})/1024^2) MB")
