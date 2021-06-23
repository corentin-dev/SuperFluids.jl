using WriteVTK
using HDF5

"Abstract supertype for writer."
abstract type AbstractWriter{T} end

"""
    WriterSave{T<:AbstractField} <: AbstractWriter{T}

Type representing a saving writer for a given field.
"""
struct WriterSave{T<:AbstractField} <: AbstractWriter{T}
   f :: T
end

function write!(w::WriterSave{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T

   h5open("$(prefix)-$(icpu)-$(istep).h5", "w") do h5file
      h5file["field"] = Array(w.f.ϕ)
   end

   return nothing
end

function read!(w::WriterSave{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T

   h5open("$(prefix)-$(icpu)-$(istep).h5", "r") do h5file
      ϕ = read(h5file,"field")
      copyto!(w.f.ϕ, ϕ)
   end

   return nothing
end

"""
    WriterVTK{T<:AbstractField} <: AbstractWriter{T}

Type representing a VTK writer for a given field.
"""
struct WriterVTK{T<:AbstractField} <: AbstractWriter{T}
   f :: T
   filename :: AbstractString
end

function WriterVTK(f::AbstractField, filename="GPS.pvd"::AbstractString)
   return WriterVTK{typeof(f)}(f,filename)
end

function write!(w::WriterVTK{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {T <: AbstractField2D}
   # references
   f = w.f
   plan = Plan(f)
   plan_x, plan_y = plan.plan_x, plan.plan_y
   ϕ = parent(f.ϕ)
   ξx, ξy = f.g.ξx, f.g.ξy
   x, y = vec(f.g.x), vec(f.g.y)

   ϕhat = plan_x * f.ϕ
   ϕhat .= im .* ξx .* ϕhat
   ∇ϕ_x = plan_x \ ϕhat
   ϕhat = plan_y * f.ϕ
   ϕhat .= im .* ξy .* ϕhat
   ∇ϕ_y = plan_y \ ϕhat

   # create vtk
   vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep).vtr", Array(x), Array(y))
   vtkfile["Re_Phi", VTKPointData()] = Array(real.(ϕ))
   vtkfile["Im_Phi", VTKPointData()] = Array(imag.(ϕ))
   vtkfile["Module", VTKPointData()] = Array(real.(ϕ.*conj.(ϕ)))
   vtkfile["Phase", VTKPointData()] = Array(angle.(ϕ.*conj.(ϕ)))
   vtkfile["VelocityX", VTKPointData()] = Array(imag.(conj.(ϕ).*∇ϕ_x))
   vtkfile["VelocityY", VTKPointData()] = Array(imag.(conj.(ϕ).*∇ϕ_y))
   vtkfile["Time"] = Δt * istep

   # write to file
   outfiles = vtk_save(vtkfile)

   # add to pvd
   pvd = paraview_collection(w.filename, append=true)
   pvd[ Δt * istep] = vtkfile
   vtk_save(pvd)

   return nothing
end

function write!(w::WriterVTK{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {T <: AbstractField3D}
   # references
   f = w.f
   plan = Plan(f)
   plan_x, plan_y, plan_z = plan.plan_x, plan.plan_y, plan.plan_z
   ϕ = parent(f.ϕ)
   ξx, ξy, ξz = f.g.ξx, f.g.ξy, f.g.ξz
   x, y, z = vec(f.g.x), vec(f.g.y), vec(f.g.z)

   ϕhat = plan_x * f.ϕ
   ϕhat .= im .* ξx .* ϕhat
   ∇ϕ_x = plan_x \ ϕhat
   ϕhat = plan_y * f.ϕ
   ϕhat .= im .* ξy .* ϕhat
   ∇ϕ_y = plan_y \ ϕhat
   ϕhat = plan_z * f.ϕ
   ϕhat .= im .* ξz .* ϕhat
   ∇ϕ_z = plan_z \ ϕhat

   vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep).vtr", x, y, z)
   vtkfile["Re_Phi", VTKPointData()] = Array(real.(ϕ))
   vtkfile["Im_Phi", VTKPointData()] = Array(imag.(ϕ))
   vtkfile["Module", VTKPointData()] = Array(real.(ϕ.*conj.(ϕ)))
   vtkfile["Phase", VTKPointData()] = Array(angle.(ϕ.*conj.(ϕ)))
   vtkfile["VelocityX", VTKPointData()] = Array(imag.(conj.(ϕ).*∇ϕ_x))
   vtkfile["VelocityY", VTKPointData()] = Array(imag.(conj.(ϕ).*∇ϕ_y))
   vtkfile["VelocityZ", VTKPointData()] = Array(imag.(conj.(ϕ).*∇ϕ_z))
   vtkfile["Time"] = Δt * istep

   outfiles = vtk_save(vtkfile)

   w.pvd[ Δt * istep] = vtkfile

   return nothing
end

function read!(w::WriterVTK{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T

   return nothing
end

"Abstract supertype for a collection of writers."
abstract type AbstractWriterCollection{T} end

mutable struct WriterCollection{T} <:AbstractWriterCollection{T}
   writerList :: Array{AbstractWriter{T},1}
end

function write!(wc::WriterCollection{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T
   for writer in wc.writerList
      write!(writer,prefix=prefix,icpu=icpu,istep=istep)
   end
end

function read!(wc::WriterCollection{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T
   for writer in wc.writerList
      read!(writer,prefix=prefix,icpu=icpu,istep=istep,Δt=Δt)
   end
end
