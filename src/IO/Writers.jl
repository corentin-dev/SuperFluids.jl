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

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", w.f.ϕ.pencil.topology.comm, write=true) do h5file
      tmp = PencilArray(w.f.ϕ.pencil, Array{Float64}(undef, size_local(w.f.ϕ)))
      @. tmp = real(w.f.ϕ)
      h5file["re"] = tmp
      @. tmp = imag(w.f.ϕ)
      h5file["im"] = tmp
      # @. tmp = icpu
      # h5file["icpu"] = tmp
   end

   return nothing
end

function read!(w::WriterSave{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where T

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", w.f.ϕ.pencil.topology.comm, read=true) do h5file
      tmp = PencilArray(w.f.ϕ.pencil, Array{Float64}(undef, size_local(w.f.ϕ)))
      read!(h5file, tmp, "re")
      w.f.ϕ .= tmp
      read!(h5file, tmp, "im")
      w.f.ϕ .+= 1j .* tmp
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

function write!(w::WriterVTK{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField{FT,FFT,A}} where {FT,FFT,A}

   ϕ = w.f.ϕ
   comm = w.f.ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)

   r_local = range_local(ϕ)
   r_local = (r_local[1][begin]:r_local[1][end]+1, r_local[2][begin]:r_local[2][end]+1)
   x, y = A(LinRange(w.f.g.xmin,w.f.g.xmax,w.f.g.nx+1)[r_local[1]]), A(LinRange(w.f.g.ymin,w.f.g.ymax,w.f.g.ny+1)[r_local[2]])
   extents = MPI.Allgather(r_local,comm)

   tmp = A{FT}(undef, size_local(w.f.ϕ))
   saved_files = pvtk_grid("$(prefix)-$(istep)", x, y, extents = extents, part = mpi_rank, nparts = mpi_size) do pvtk
      tmp .= real.(parent(ϕ))
      pvtk["re", VTKCellData()] = tmp
      tmp .= imag.(parent(ϕ))
      pvtk["im", VTKCellData()] = tmp
      tmp .= sqrt.(abs2.(parent(ϕ)))
      pvtk["module", VTKCellData()] = tmp
   end

   return nothing
end

function write!(w::WriterVTK{T};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {T <: AbstractField3D}


   ϕ = w.f.ϕ
   comm = w.f.ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)
   grid = localgrid(w.f.ϕ.pencil, (w.f.g.x, w.f.g.y, w.f.g.z))
   x, y, z = grid.x, grid.y, grid.z

   tmp = similar(parent(ϕ))
   saved_files = pvtk_grid("$(prefix)-$(istep)", x, y, z; part = mpi_rank, nparts = mpi_size) do pvtk
      tmp .= real.(parent(ϕ))
      pvtk["re"] = tmp
      pvtk["im"] = tmp
      # pvtk["Processor"] = rand(2)
   end
   # outfiles = vtk_save(vtkfile)

#    # create vtk
#    vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep).vtr", x, y, z)
#    vtkfile["Re_Phi", VTKPointData()] = Array(real.(ϕ))
#    vtkfile["Im_Phi", VTKPointData()] = Array(imag.(ϕ))
#    vtkfile["Module", VTKPointData()] = Array(real.(ϕ.*conj.(ϕ)))
#    vtkfile["Time"] = Δt * istep
#    # else
#    # vtkfile = vtk_grid("$(prefix)-$(icpu)-$(istep).vtr", Array(x), Array(y))
#    # vtkfile["VelocityX", VTKPointData()] = real.(parent(f.ϕ)[:,:,:,1])
#    # vtkfile["VelocityY", VTKPointData()] = real.(parent(f.ϕ)[:,:,:,2])
#    # vtkfile["VelocityZ", VTKPointData()] = real.(parent(f.ϕ)[:,:,:,3])
#    # vtkfile["Time"] = Δt * istep
#    # end

#    # write to file
#    outfiles = vtk_save(vtkfile)

   # add to pvd
   # if istep == 0
   #    pvd = paraview_collection(w.filename, append=false)
   # else
   #    pvd = paraview_collection(w.filename, append=true)
   # end
   # pvd[ Δt * istep] = vtkfile
   # vtk_save(pvd)

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
