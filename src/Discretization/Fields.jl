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
abstract type AbstractField{N,ND,FT,FFT,A,PA,G,P} end
"Alias for 1D abstract field."
const AbstractField1D{ND,FT,FFT,A,PA,G,P} = AbstractField{1,ND,FT,FFT,A,PA,G,P}
"Alias for 2D abstract field."
const AbstractField2D{ND,FT,FFT,A,PA,G,P} = AbstractField{2,ND,FT,FFT,A,PA,G,P}
"Alias for 3D abstract field."
const AbstractField3D{ND,FT,FFT,A,PA,G,P} = AbstractField{3,ND,FT,FFT,A,PA,G,P}

"""
$(TYPEDEF)

Type representing a field on a grid.

$(TYPEDFIELDS)
"""
mutable struct Field{N,ND,FT,FFT,A,PA,G,P} <: AbstractField{N,ND,FT,FFT,A,PA,G,P}
    "informations concerning the decomposition."
    pen::P
    "local grid (relative to the `ϕ` decomposition)."
    grid::LocalGrids.AbstractLocalGrid
    "reference to the grid."
    g::G
    "distributed containing the data."
    data::Array{PA,ND}
end

"Alias for 1D field."
const Field1D{ND,FT,FFT,A,PA,G,P} = Field{1,ND,FT,FFT,A,PA,G,P}
"Alias for 2D field."
const Field2D{ND,FT,FFT,A,PA,G,P} = Field{2,ND,FT,FFT,A,PA,G,P}
"Alias for 3D field."
const Field3D{ND,FT,FFT,A,PA,G,P} = Field{3,ND,FT,FFT,A,PA,G,P}

"""
$(TYPEDSIGNATURES)

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
function Field(g::AbstractGrid2D{FT,A}, t::FieldType;
               ndims::Integer=1,
               mpi_topo::AbstractDomainDecomposition=MPITopo1D()) where {FT<:Real,A}
    if typeof(t) == RealField
        myT = FT
    elseif typeof(t) == ComplexField
        myT = Complex{FT}
    end

    dims = (g.nx, g.ny)

    pen_x = Pencil(A, mpi_topo.topo, dims, (2,))
    local_dims = size_local(pen_x)
    data = [PencilArray(pen_x, A{myT}(undef, local_dims))]
    for i in 1:(ndims - 1)
        push!(data, PencilArray(pen_x, A{myT}(undef, local_dims)))
    end

    pen_array_glob = global_view(data[1])
    r = Tuple([minimum(a):maximum(a) for a in axes(pen_array_glob)])

    grid = localgrid(pen_x, (g.x, g.y))

    return Field2D{ndims,FT,myT,A,typeof(data[1]),typeof(g),typeof(pen_x)}(pen_x, grid, g,
                                                                           data)
end

"""
$(TYPEDSIGNATURES)

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
function Field(g::AbstractGrid3D{FT,A}, t::FieldType;
               ndim::Integer=1,
               mpi_topo::AbstractDomainDecomposition=MPITopo2D()) where {FT<:Real,A}
    if typeof(t) == RealField
        myT = FT
    elseif typeof(t) == ComplexField
        myT = Complex{FT}
    end

    dims = (g.nx, g.ny, g.nz)
    pen_x = Pencil(A, mpi_topo.topo, dims, (2, 3))
    local_dims = size_local(pen_x)
    data = [PencilArray(pen_x, A{myT}(undef, local_dims))]
    for i in 1:(ndim - 1)
        push!(data, PencilArray(pen_x, A{myT}(undef, local_dims)))
    end

    pen_array_glob = global_view(data[1])
    r = Tuple([minimum(a):maximum(a) for a in axes(pen_array_glob)])

    grid = localgrid(pen_x, (g.x, g.y, g.z))

    return Field3D{ndims,FT,myT,A,typeof(data[1]),typeof(g),typeof(pen_x)}(pen_x, grid, g,
                                                                           data)
end

function norm(f::Field2D)
    normϕ = sum(abs2.(f.ϕ))
    return normϕ = sqrt(normϕ) * sqrt(f.g.Δx * f.g.Δy)
end

function norm(f::Field3D)
    normϕ = sum(abs2.(f.ϕ))
    return normϕ = sqrt(normϕ) * sqrt(f.g.Δx * f.g.Δy * f.g.Δz)
end

"""
$(TYPEDSIGNATURES)

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
    elseif name === :u
        f.data
    elseif name === :ux
        f.data[1]
    elseif name === :uy
        f.data[2]
    elseif name === :uz
        f.data[3]
    elseif name === :x
        f.grid.x
    elseif name === :y
        f.grid.y
    elseif name === :z
        f.grid.z
    else
        getfield(f, name)
    end
end

function similar_data(data::Vector{A}) where {A}
    newdata = [similar(data[1])]
    for i in 1:(length(data) - 1)
        push!(newdata, similar(data[1]))
    end
    return newdata
end

function Base.show(io::IO, f::Field2D{ND,FT,FFT}) where {ND,FT,FFT}
    return print(io, "Field2D\n",
                 "  ├──────  Array type: $(FFT)", '\n',
                 "  └──────────  memory: $(sizeof(f.ϕ)/1024^2) MB")
end

function Base.show(io::IO, f::Field3D{ND,FT,FFT}) where {ND,FT,FFT}
    return print(io, "Field3D\n",
                 "  ├───────  FloatType: $(FFT)", '\n',
                 "  └──────────  memory: $(sizeof(f.ϕ.data)/1024^2) MB")
end
