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
      Δt=1::Real) where {F <: AbstractField2D}
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
   vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep).vtr", x, y)
   vtkfile["Re_Phi", VTKPointData()] = real(ϕ)
   vtkfile["Im_Phi", VTKPointData()] = imag(ϕ)
   vtkfile["Module", VTKPointData()] = real(ϕ.*conj(ϕ))
   vtkfile["VelocityX", VTKPointData()] = imag.(conj.(ϕ).*∇ϕ_x)
   vtkfile["VelocityY", VTKPointData()] = imag.(conj.(ϕ).*∇ϕ_y)
   vtkfile["Time"] = Δt * istep
   outfiles = vtk_save(vtkfile)
   w.pvd[ Δt * istep] = vtkfile
   return nothing
end

function addFile!(w::AbstractWriter{F},
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {F <: AbstractField3D}
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
   vtkfile["Re_Phi", VTKPointData()] = real(ϕ)
   vtkfile["Im_Phi", VTKPointData()] = imag(ϕ)
   vtkfile["Module", VTKPointData()] = real(ϕ.*conj(ϕ))
   vtkfile["VelocityX", VTKPointData()] = imag.(conj.(ϕ).*∇ϕ_x)./real.(ϕ.*conj.(ϕ))
   vtkfile["VelocityY", VTKPointData()] = imag.(conj.(ϕ).*∇ϕ_y)./real.(ϕ.*conj.(ϕ))
   vtkfile["VelocityZ", VTKPointData()] = imag.(conj.(ϕ).*∇ϕ_z)./real.(ϕ.*conj.(ϕ))
   vtkfile["Time"] = Δt * istep
   outfiles = vtk_save(vtkfile)
   w.pvd[ Δt * istep] = vtkfile
   return nothing
end

function finishWriter!(w::AbstractWriter)
   vtk_save(w.pvd)
   return nothing
end
