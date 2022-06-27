"Abstract type for gradient field."
abstract type AbstractGradientField{N,F,A} end
"Alias for 1D abstract gradient field."
const AbstractGradientField1D{F,A} = AbstractGradientField{1,F,A}
"Alias for 2D abstract gradient field."
const AbstractGradientField2D{F,A} = AbstractGradientField{2,F,A}
"Alias for 3D abstract gradient field."
const AbstractGradientField3D{F,A} = AbstractGradientField{3,F,A}

export GradientField

"""
$(TYPEDEF)

Type representing a gradient of a 2D field.

It contains the following informations:

$(TYPEDFIELDS)
"""
mutable struct GradientField{N,F,A} <: AbstractGradientField{N,F,A}
    "reference to a field."
    f::F
    "first derivatives of the given field."
    ∇data::A
    "second derivatives of the given field."
    Δdata::A
end

"Alias for 1D gradient field with gradient"
const GradientField1D{F,A} = GradientField{1,F,A}
"Alias for 2D gradient field with gradient"
const GradientField2D{F,A} = GradientField{2,F,A}
"Alias for 3D gradient field with gradient"
const GradientField3D{F,A} = GradientField{3,F,A}

"""
$(TYPEDEF)

Type representing a gradient of a 2D field with rotation along z axis.

It contains the following informations:

$(TYPEDFIELDS)
"""
mutable struct GradientRotField{N,F,A} <: AbstractGradientField{N,F,A}
    "reference to a field."
    f::F
    "first derivatives of the given field."
    ∇data::A
    "second derivatives of the given field."
    Δdata::A
    "rotation along the `z` axis."
    rdata::A
end

"Alias for 1D gradient field with gradient and rotation."
const GradientRotField1D{F,A} = GradientRotField{1,F,A}
"Alias for 2D gradient field with gradient and rotation."
const GradientRotField2D{F,A} = GradientRotField{2,F,A}
"Alias for 3D gradient field with gradient and rotation."
const GradientRotField3D{F,A} = GradientRotField{3,F,A}

"""
$(TYPEDEF)

Type representing a gradient of a 3D field, laplacian and curl.

It contains the following informations:

$(TYPEDFIELDS)
"""
mutable struct GradientCurlField{N,F,A} <: AbstractGradientField{N,F,A}
    "reference to a field."
    f::F
    "first derivatives of the given field."
    ∇data::A
    "curl derivatives of given field."
    ωdata::A
end

"Alias for 1D gradient field with gradient and curl."
const GradientCurlField1D{F,A} = GradientCurlField{1,F,A}
"Alias for 2D gradient field with gradient and curl."
const GradientCurlField2D{F,A} = GradientCurlField{2,F,A}
"Alias for 3D gradient field with gradient and curl."
const GradientCurlField3D{F,A} = GradientCurlField{3,F,A}

"""
$(TYPEDSIGNATURES)

Returns a GradientField2D or GradientRotField2D.

Parameters are:

- `f`: a field
- `rotation`: specify if a rotation should be computed or not.
- `laplacian`: specify if a rotation should be computed or not.
- `vorticity`: specify if a rotation should be computed or not.
"""
function GradientField(f::F; rotation::Bool=false, laplacian::Bool=true,
                       vorticity::Bool=false) where {F<:AbstractField2D}
    ∇data = similar_data(f.data) # dx
    push!(∇data, similar_data(f.data)...) # dy
    if laplacian
        Δdata = similar_data(f.data) # dx
        push!(Δdata, similar_data(f.data)...) # ddx
        if rotation
            rdata = similar_data(f.data) # rx
            push!(rdata, similar_data(f.data)...) # ry
            return GradientRotField2D{typeof(f),typeof(∇data)}(f, ∇data, Δdata, rdata)
        else
            return GradientField2D{typeof(f),typeof(∇data)}(f, ∇data, Δdata)
        end
    elseif vorticity == true && rotation == false
        ωdata = similar_data(f.data) # ωz
        return GradientCurlField2D{typeof(f),typeof(∇data)}(f, ∇data, ωdata)
    else
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Returns a GradientField3D or GradientRotField3D.

Parameters are:

- `f`: a field
- `laplacian`: specify if a rotation should be computed or not.
- `rotation`: specify if a rotation should be computed or not.
- `vorticity`: specify if a rotation should be computed or not.
"""
function GradientField(f::F; rotation::Bool=false, laplacian::Bool=true,
                       vorticity::Bool=false) where {F<:AbstractField3D}
    if laplacian
        # grad
        ∇data = similar_data(f.data) # dx
        push!(∇data, similar_data(f.data)...) # dy
        push!(∇data, similar_data(f.data)...) # dz
        # lap
        Δdata = similar_data(f.data) # dx
        push!(Δdata, similar_data(f.data)...) # ddx
        push!(Δdata, similar_data(f.data)...) # ddz
        if rotation
            rdata = similar_data(f.data) # rx
            push!(rdata, similar_data(f.data)...) # ry
            return GradientRotField3D{typeof(f),typeof(∇data)}(f, ∇data, Δdata, rdata)
        else
            return GradientField3D{typeof(f),typeof(∇data)}(f, ∇data, Δdata)
        end
    elseif vorticity == true && rotation == false
        ωdata = [similar(f.data[1])] # ωx
        push!(ωdata, similar_data(f.data)...) # ωy
        push!(ωdata, similar_data(f.data)...) # ωz
        return GradientCurlField3D{typeof(f),typeof(ωdata)}(f, ωdata)
    else
        return nothing
    end
end

@inline function Base.getproperty(gf::AbstractGradientField{N,F},
                                  name::Symbol) where {N,F<:AbstractField{N,1}}
    if name === :dx
        return gf.∇data[1]
    elseif name === :dy
        return gf.∇data[2]
    elseif name === :dz
        return gf.∇data[3]
    elseif name === :ddx
        return gf.Δdata[1]
    elseif name === :ddy
        return gf.Δdata[2]
    elseif name === :ddz
        return gf.Δdata[3]
    elseif name === :rx
        return gf.rdata[1]
    elseif name === :ry
        return gf.rdata[2]
    elseif name === :ω
        return gf.ω[1]
    else
        return getfield(gf, name)
    end
end

@inline function Base.getproperty(gf::AbstractGradientField{N,F},
                                  name::Symbol) where {N,F<:AbstractField{N,ND}} where {ND}
    if name === :dx
        return gf.∇data[1:ND]
    elseif name === :dy
        return gf.∇data[(1 + ND):(2 * ND)]
    elseif name === :dz
        return gf.∇data[(1 + 2 * ND):(3 * ND)]
    elseif name === :ddx
        return gf.Δdata[1:ND]
    elseif name === :ddy
        return gf.Δdata[(1 + ND):(2 * ND)]
    elseif name === :ddz
        return gf.Δdata[(1 + 2 * ND):(3 * ND)]
    elseif name === :rx
        return gf.rdata[1]
    elseif name === :ry
        return gf.rdata[2]
    elseif name === :ω
        return gf.ωdata[1:ND]
    else
        return getfield(gf, name)
    end
end

function Base.show(io::IO, f::GradientField2D{F,A}) where {F,A}
    return print(io, "GradientField2D\n",
                 "  ├──────  Array type: $(A)", '\n',
                 "  └──────────  memory: $(2*f.g.nx*f.g.ny*16/1024^2) MB")
end

function Base.show(io::IO, f::GradientField3D{F,A}) where {F,A}
    return print(io, "GradientField3D\n",
                 "  ├───────  FloatType: $(A)", '\n',
                 "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*16/1024^2) MB")
end
