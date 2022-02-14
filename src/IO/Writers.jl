using WriteVTK
using HDF5

"Abstract supertype for writer."
abstract type AbstractWriter{F} end

"""
    WriterSave{F<:AbstractField} <: AbstractWriter{F}

Type representing a saving writer for a given field.
"""
struct WriterSave{F<:AbstractField} <: AbstractWriter{F}
   f :: F
end

function write!(w::WriterSave{F};
   prefix="res"::AbstractString,
   icpu=0::Integer,
   istep=0::Integer,
   Δt=1::Real) where {F<:AbstractField{FT,FFT,A}} where {FT,FFT,A<:Array}

   tmp = w.f.ϕ

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", tmp.pencil.topology.comm, write=true) do h5file
      tmpr = PencilArray(tmp.pencil, Array{FT}(undef, size_local(tmp)))
      @. tmpr = real(tmp)
      h5file["re"] = tmp
      @. tmpr = imag(tmp)
      h5file["im"] = tmp
   end

   return nothing
end

function write!(w::WriterSave{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField{FT,FFT,A}} where {FT,FFT,A}

   pen = Pencil(size_global(w.f.ϕ), MPI.COMM_WORLD)
   tmp = PencilArray{FFT}(undef, pen)
   copyto!(parent(tmp),parent(w.f.ϕ))

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", tmp.pencil.topology.comm, write=true) do h5file
      tmpr = PencilArray(tmp.pencil, Array{FT}(undef, size_local(tmp)))
      @. tmpr = real(tmp)
      h5file["re"] = tmp
      @. tmpr = imag(tmp)
      h5file["im"] = tmp
   end

   return nothing
end

function read!(w::WriterSave{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where F

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
    WriterVTK{F<:AbstractField} <: AbstractWriter{F}

Type representing a VTK writer for a given field.
"""
struct WriterVTK{F<:AbstractField} <: AbstractWriter{F}
   f :: F
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
   comm = ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)

   r_local = range_local(ϕ)
   r_local = (r_local[1][begin]:r_local[1][end]+1, r_local[2][begin]:r_local[2][end]+1)
   x, y = A(LinRange(w.f.g.xmin,w.f.g.xmax,w.f.g.nx+1)[r_local[1]]), A(LinRange(w.f.g.ymin,w.f.g.ymax,w.f.g.ny+1)[r_local[2]])
   extents = MPI.Allgather(r_local,comm)

   tmp = A{FT}(undef, size_local(ϕ))
   vtkfile = pvtk_grid("$(prefix)-$(istep)", x, y, extents = extents, part = mpi_rank, nparts = mpi_size)
   tmp .= real.(parent(ϕ))
   vtkfile["re", VTKCellData()] = tmp
   tmp .= imag.(parent(ϕ))
   vtkfile["im", VTKCellData()] = tmp
   tmp .= sqrt.(abs2.(parent(ϕ)))
   vtkfile["module", VTKCellData()] = tmp

   # write to file
   outfiles = vtk_save(vtkfile)

   # add to pvd
   if istep == 0
      pvd = paraview_collection(w.filename, append=false)
   else
      pvd = paraview_collection(w.filename, append=true)
   end
   pvd[ Δt * istep] = vtkfile
   vtk_save(pvd)

   return nothing
end

function write!(w::WriterVTK{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField{FT,FFT,A}} where {FT,FFT,A}


   ϕ = w.f.ϕ
   comm = ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)

   r_local = range_local(ϕ)
   r_local = (r_local[1][begin]:r_local[1][end]+1, r_local[2][begin]:r_local[2][end]+1)
   x, y, z = A(LinRange(w.f.g.xmin,w.f.g.xmax,w.f.g.nx+1)[r_local[1]]), A(LinRange(w.f.g.ymin,w.f.g.ymax,w.f.g.ny+1)[r_local[2]]), A(LinRange(w.f.g.zmin,w.f.g.zmax,w.f.g.nz+1)[r_local[2]])
   extents = MPI.Allgather(r_local,comm)

   tmp = A{FT}(undef, size_local(ϕ))
   vtkfile = pvtk_grid("$(prefix)-$(istep)", x, y, z, extents = extents, part = mpi_rank, nparts = mpi_size)
   tmp .= real.(parent(ϕ))
   vtkfile["re", VTKCellData()] = tmp
   tmp .= imag.(parent(ϕ))
   vtkfile["im", VTKCellData()] = tmp
   tmp .= sqrt.(abs2.(parent(ϕ)))
   vtkfile["module", VTKCellData()] = tmp

   # write to file
   outfiles = vtk_save(vtkfile)

   # add to pvd
   if istep == 0
      pvd = paraview_collection(w.filename, append=false)
   else
      pvd = paraview_collection(w.filename, append=true)
   end
   pvd[ Δt * istep] = vtkfile
   vtk_save(pvd)

   return nothing
end

function read!(w::WriterVTK{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where F

   return nothing
end

"Abstract supertype for a collection of writers."
abstract type AbstractWriterCollection{F} end

mutable struct WriterCollection{F} <:AbstractWriterCollection{F}
   writerList :: Array{AbstractWriter{F},1}
end

function write!(wc::WriterCollection{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where F
   for writer in wc.writerList
      write!(writer,prefix=prefix,icpu=icpu,istep=istep)
   end
end

function read!(wc::WriterCollection{F};
      prefix="res"::AbstractString,
      icpu=0::Integer,
      istep=0::Integer,
      Δt=1::Real) where F
   for writer in wc.writerList
      read!(writer,prefix=prefix,icpu=icpu,istep=istep,Δt=Δt)
   end
end