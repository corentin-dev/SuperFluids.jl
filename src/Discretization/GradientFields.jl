"Abstract supertype for numerical models."
abstract type AbstractGradientField{F,A} end
"Abstract supertype for numerical models."
abstract type AbstractGradientField2D{F,A} <: AbstractGradientField{F,A} end
"Abstract supertype for numerical models."
abstract type AbstractGradientField3D{F,A} <: AbstractGradientField{F,A} end

"""
    GradientField2D{F<:AbstractGrid,A<:AbstractArrayd} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 2D field.
"""
mutable struct GradientField2D{F,A} <: AbstractGradientField2D{F,A}
    f :: F
    dx :: A
    dy :: A
    ddx :: A
    ddy :: A
end

"""
    GradientRotField2D{F<:AbstractGrid,A<:AbstractArray} <: AbstractGradientField2D{F,A}

Type representing a gradient of a 2D field with rotation along z axis.
"""
mutable struct GradientRotField2D{F,A} <: AbstractGradientField2D{F,A}
   f :: F
   dx :: A
   dy :: A
   ddx :: A
   ddy :: A
   rx :: A
   ry :: A
end

function GradientField(f::F; rotation::Bool = true) where {F<:AbstractField2D}
    dx = similar(f.ϕ)
    dy = similar(f.ϕ)
    ddx = similar(f.ϕ)
    ddy = similar(f.ϕ)
    if rotation
        rx = similar(f.ϕ)
        ry = similar(f.ϕ)
        return GradientRotField2D{typeof(f),typeof(dx)}(f, dx, dy, ddx, ddy, rx, ry)
    else
        return GradientField2D{typeof(f),typeof(dx)}(f, dx, dy, ddx, ddy)
    end
end

"""
    GradientField3D{F<:AbstractGrid,A<:AbstractArrayd} <: AbstractGradientField3D{F,A}

Type representing a gradient of a 3D field.
"""
mutable struct GradientField3D{F,A} <: AbstractGradientField3D{F,A}
    f :: F
    dx :: A
    dy :: A
    dz :: A
    ddx :: A
    ddy :: A
    ddz :: A
end

"""
    GradientRotField3D{F<:AbstractGrid,A<:AbstractArrayd} <: AbstractGradientField3D{F,A}

Type representing a gradient of a 3D field.
"""
mutable struct GradientRotField3D{F,A} <: AbstractGradientField3D{F,A}
    f :: F
    dx :: A
    dy :: A
    dz :: A
    ddx :: A
    ddy :: A
    ddz :: A
    rx :: A
    ry :: A
end

function GradientField(f::F; rotation::Bool = true) where {F<:AbstractField3D}
    dx = similar(f.ϕ)
    dy = similar(f.ϕ)
    dz = similar(f.ϕ)
    ddx = similar(f.ϕ)
    ddy = similar(f.ϕ)
    ddz = similar(f.ϕ)
    if rotation
        rx = similar(f.ϕ)
        ry = similar(f.ϕ)
        return GradientRotField3D{typeof(f),typeof(dx)}(f, dx, dy, dz, ddx, ddy, ddz, dz, rx, ry)
    else
        return GradientField3D{typeof(f),typeof(dx)}(f, dx, dy, dz, ddx, ddy, ddz)
    end
end

# Base.show(io::IO, f::GradientField2D{A}) where A =
#      print(io, "GradientField2D\n",
#          "  ├──────  Array type: $(A)", '\n',
#          "  └──────────  memory: $(2*f.g.nx*f.g.ny*16/1024^2) MB")

# Base.show(io::IO, f::GradientField3D{A}) where A =
#      print(io, "GradientField3D\n",
#          "  ├───────  FloatType: $(A)", '\n',
#          "  └──────────  memory: $(2*f.g.nx*f.g.ny*f.g.nz*16/1024^2) MB")
