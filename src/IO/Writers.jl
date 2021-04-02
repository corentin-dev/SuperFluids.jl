using WriteVTK

export Writer, addFile!, finishWriter!

"Abstract supertype for writer."
abstract type AbstractWriter{F} end
"Abstract supertype for 2D writer."
abstract type AbstractWriter2D{F} <: AbstractWriter{F} end
"Abstract supertype for 3D writer."
abstract type AbstractWriter3D{F} <: AbstractWriter{F} end

"""
    Writer2D{F<:AbstractField2D} <: AbstractWriter2D{F}

Type representing a 2D writer on a 2D grid.
"""
struct Writer2D{F<:AbstractField2D} <: AbstractWriter2D{F}
   f :: F
   pvd :: WriteVTK.CollectionFile
end

function Writer(f::AbstractField2D, name="GPS_2D.pvd"::AbstractString)
   pvd = paraview_collection(name)
   return Writer2D{typeof(f)}(nf,modϕ,modϕsurf,realϕ,imagϕ)
end

function addFile!(w::AbstractWriter2D,
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1.::Real)
   f = w.f
   ϕ = parent(f.ϕ)
   x = f.g.x
   y = f.g.y
   vtkfile = vtk_grid("res-$(icpu)-$(istep*Δt)", x, y)
   vtkfile["Re_Phi", VTKPointData()] = real(parent(ϕ))
   vtkfile["Im_Phi", VTKPointData()] = imag(parent(ϕ))
   vtkfile["Module", VTKPointData()] = real(parent(ϕ).*conj(parent(ϕ)))
   vtkfile["Time"] = Δt * istep
   outfiles = vtk_save(vtkfile)
   w.pvd[ Δt * i] = vtkfile
   return nothing
end

function finishWriter!(w::AbstractWriter)
   vtk_save(w.pvd)
   return nothing
end
