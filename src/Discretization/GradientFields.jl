"Abstract supertype for numerical models."
abstract type AbstractGradientField{F,A} end
"Abstract supertype for numerical models."
abstract type AbstractGradientField2D{F,A} <: AbstractGradientField{F,A} end
"Abstract supertype for numerical models."
abstract type AbstractGradientField3D{F,A} <: AbstractGradientField{F,A} end

export GradientField

"""
     GradientField2D{F,A} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 2D field.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`: first derivatives of the given field.
- `ddx`, `ddy`: second derivatives of the given field.

"""
mutable struct GradientField2D{F,A} <: AbstractGradientField2D{F,A}
    f :: F
    ∇data :: A
    Δdata :: A
end

"""
     GradientRotField2D{F,A} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 2D field with rotation along z axis.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`: first derivatives of the given field.
- `ddx`, `ddy`: second derivatives of the given field.
- `rx`, `ry`: rotation along the `z` axis.

"""
mutable struct GradientRotField2D{F,A} <: AbstractGradientField2D{F,A}
   f :: F
   ∇data :: A
   Δdata :: A
   rdata :: A
end

"""
     GradientCurlField2D{F,A} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 3D field, laplacian and curl.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`: first derivatives of the given field.
- `ddx`, `ddy`: second derivatives of the given field.
- `rx`, `ry`: rotation along the `z` axis.

"""
mutable struct GradientCurlField3D{F,A} <: AbstractGradientField3D{F,A}
    f :: F
    ∇data :: A
    ωdata :: A
end

"""
     GradientField3D{F,A} <: AbstractGradientField3D{F,A}

Type representing a gradient of a 3D field.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`, `dz`: first derivatives of the given field.
- `ddx`, `ddy`, `ddz`: second derivatives of the given field.

"""
mutable struct GradientField3D{F,A} <: AbstractGradientField3D{F,A}
    f :: F
    ∇data :: A
    Δdata :: A
end

"""
     GradientRotField3D{F,A} <: AbstractGradientField3D{F,A}

Type representing a gradient of a 3D field with rotation along z axis.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`, `dz`: first derivatives of the given field.
- `ddx`, `ddy`, `ddz`: second derivatives of the given field.
- `rx`, `ry`: rotation along the `z` axis.

"""
mutable struct GradientRotField3D{F,A} <: AbstractGradientField3D{F,A}
    f :: F
    ∇data :: A
    Δdata :: A
    rdata :: A
end

"""
     GradientCurlField2D{F,A} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 2D field, laplacian and curl.

It contains the following informations:

- `f`: reference to a field.
- `dx`, `dy`: first derivatives of the given field.
- `ω`: rotation along the `z` axis.

"""
mutable struct GradientCurlField2D{F,A} <: AbstractGradientField2D{F,A}
    f :: F
    ∇data :: A
    ωdata :: A
end

"""
     GradientField(f::F; rotation::Bool = true) where {F<:AbstractField2D}

Returns a GradientField2D or GradientRotField2D.

Parameters are:

- `f`: a field
- `rotation`: specify if a rotation should be computed or not.
- `laplacian`: specify if a rotation should be computed or not.
- `vorticity`: specify if a rotation should be computed or not.
"""
function GradientField(f::F; rotation::Bool = false, laplacian::Bool = true, vorticity::Bool = false) where {F<:AbstractField2D}
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
        ωdata = [similar(f.data[1])] # ω
        return GradientCurlField2D{typeof(f),typeof(∇data)}(f, ∇data, ωdata)
    else
        return nothing
    end
end

"""
     GradientField(f::F; rotation::Bool = true) where {F<:AbstractField3D}

Returns a GradientField3D or GradientRotField3D.

Parameters are:

- `f`: a field
- `laplacian`: specify if a rotation should be computed or not.
- `rotation`: specify if a rotation should be computed or not.
- `vorticity`: specify if a rotation should be computed or not.
"""
function GradientField(f::F; rotation::Bool = false, laplacian::Bool = true, vorticity::Bool = false) where {F<:AbstractField3D}
    ∇data = similar_data(f.data) # dx
    push!(∇data, similar_data(f.data)...) # dy
    push!(∇data, similar_data(f.data)...) # dz
    if laplacian
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
        ωdata = [similar(f.data[1])] # ω
        return GradientCurlField3D{typeof(f),typeof(∇data)}(f, ∇data, ωdata)
    else
        return nothing
    end
end

@inline function Base.getproperty(gf::AbstractGradientField, name::Symbol)
    f = getfield(gf, :f)
    ndims = f.ndims
    if ndims == 1
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
        end
    else
        if name === :dx
            return gf.∇data[1:ndims]
        elseif name === :dy
            return gf.∇data[1+ndims:2*ndims]
        elseif name === :dz
            return gf.∇data[1+2*ndims:3*ndims]
        elseif name === :ddx
            return gf.Δdata[1:ndims]
        elseif name === :ddy
            return gf.Δdata[1+ndims:2*ndims]
        elseif name === :ddz
            return gf.Δdata[1+2*ndims:3*ndims]
        elseif name === :rx
            return gf.rdata[1]
        elseif name === :ry
            return gf.rdata[2]
        elseif name === :ω
            gf.ω[1]
        end
    end
    return getfield(gf, name)
 end

Base.show(io::IO, f::GradientField2D{F,A}) where {F,A} =
    print(io, "GradientField2D\n",
        "  ├──────  Array type: $(A)", '\n',
        "  └──────────  memory: $(2*f.g.nx*f.g.ny*16/1024^2) MB")

Base.show(io::IO, f::GradientField3D{F,A}) where {F,A} =
    print(io, "GradientField3D\n",
        "  ├───────  FloatType: $(A)", '\n',
        "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*16/1024^2) MB")