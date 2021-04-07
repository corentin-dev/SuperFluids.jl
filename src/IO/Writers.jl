using WriteVTK

"Abstract supertype for writer."
abstract type AbstractWriter{F} end

"""
    Writer{F<:AbstractField} <: AbstractWriter{F}

Type representing a writer on a grid.
"""
struct Writer{F<:AbstractField} <: AbstractWriter{F}
   f :: F
   pvd :: WriteVTK.CollectionFile
end

function Writer(f::AbstractField, filename="GPS.pvd"::AbstractString)
   pvd = paraview_collection(filename)
   return Writer{typeof(f)}(f,pvd)
end

function addFile!(w::AbstractWriter{F},
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1.::Real) where {F <: AbstractField2D}
   f = w.f
   ϕ = parent(f.ϕ)
   x = vec(f.g.x)
   y = vec(f.g.y)
   vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep*Δt).vtr", x, y)
   vtkfile["Re_Phi", VTKPointData()] = real(parent(ϕ))
   vtkfile["Im_Phi", VTKPointData()] = imag(parent(ϕ))
   vtkfile["Module", VTKPointData()] = real(parent(ϕ).*conj(parent(ϕ)))
   vtkfile["Time"] = Δt * istep
   outfiles = vtk_save(vtkfile)
   w.pvd[ Δt * istep] = vtkfile
   return nothing
end

function addFile!(w::AbstractWriter{F},
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1.::Real) where {F <: AbstractField3D}
   f = w.f
   ϕ = parent(f.ϕ)
   x = vec(f.g.x)
   y = vec(f.g.y)
   z = vec(f.g.z)
   vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep*Δt).vtr", x, y, z)
   vtkfile["Re_Phi", VTKPointData()] = real(parent(ϕ))
   vtkfile["Im_Phi", VTKPointData()] = imag(parent(ϕ))
   vtkfile["Module", VTKPointData()] = real(parent(ϕ).*conj(parent(ϕ)))
   vtkfile["Time"] = Δt * istep
   outfiles = vtk_save(vtkfile)
   w.pvd[ Δt * istep] = vtkfile
   return nothing
end

function finishWriter!(w::AbstractWriter)
   vtk_save(w.pvd)
   return nothing
end
