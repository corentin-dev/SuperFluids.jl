using WriteVTK
using HDF5

export WriterSave, WriterVTK
export write!, read!

"Abstract supertype for writer."
abstract type AbstractWriter{F} end

"""
   WriterSave{F<:AbstractField} <: AbstractWriter{F}

Type representing a saving writer for a given field.
"""
struct WriterSave{F<:AbstractField} <: AbstractWriter{F}
   f :: F
end

"""
   write!(w::WriterSave{AbstractField};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real)

Write a given field in a HDF5 format.

Parameters:

- `w`: a `WriterSave` with a `Field` to save,
- `prefix`: a string that is put in front of the saved file (default: `"res"`),
- `istep`: an integer representing the current `istep` value (default: `0`),
- `Δt` : a real representing the time step value (default: `1.`).

The HDF5 handles parallel files through a `PencilArray`.

```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> writer = WriterSave(field);
julia> write!(writer)
```
"""
function write!(w::WriterSave{F};
   prefix="res"::AbstractString,
   istep=0::Integer,
   Δt=1::Real) where {F<:AbstractField{N,FT,FFT,A}} where {N,FT,FFT,A<:Array}

   tmp = w.f.data

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", tmp[1].pencil.topology.comm, write=true) do h5file
      tmpr = PencilArray(tmp[1].pencil, Array{FT}(undef, size_local(tmp[1])))
      if N == 1
         @. tmpr = real(tmp[1])
         h5file["re"] = tmpr
         @. tmpr = imag(tmp[1])
         h5file["im"] = tmpr
      else
         for i = 1:N
            @. tmpr = real(tmp[i])
            h5file["U$(i)"] = tmpr
         end
      end
   end

   return nothing
end

function write!(w::WriterSave{F};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField{N,FT,FFT,A}} where {N,FT,FFT,A}

   pen = Pencil(size_global(w.f.ϕ), MPI.COMM_WORLD)
   tmp = PencilArray{FFT}(undef, pen)

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", tmp.pencil.topology.comm, write=true) do h5file
      tmpr = PencilArray(tmp.pencil, Array{FT}(undef, size_local(tmp)))
      if N == 1
         copyto!(parent(tmp),parent(w.f.ϕ))
         @. tmpr = real(tmp[1])
         h5file["re"] = tmpr
         @. tmpr = imag(tmp[1])
         h5file["im"] = tmpr
      else
         for i = 1:N
            copyto!(parent(tmp),parent(w.f.data[i]))
            @. tmpr = real(tmp)
            h5file["U$(i)"] = tmpr
         end
      end
   end

   return nothing
end

"""
   Base.read!(w::WriterSave{AbstractField};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real)

Read a given field in the HDF5 format.

Parameters:

- `w`: a `WriterSave` with a `Field` to read,
- `prefix`: a string that is put in front of the saved file (default: `"res"`),
- `istep`: an integer representing the current `istep` value (default: `0`),
- `Δt` : a real representing the time step value (default: `1.`).

The HDF5 handles parallel files through a `PencilArray`.

```jldoctest
julia> grid = Grid((128,128), ((-12,12), (-12,12)));
julia> field = Field(grid, ComplexField());
julia> writer = WriterSave(field);
julia> write!(writer)
julia> read!(writer)
```
"""
function Base.read!(w::WriterSave{F};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real) where F

   open(PencilArrays.PencilIO.PHDF5Driver(), "$(prefix)-$(istep).h5", w.f.ϕ.pencil.topology.comm, read=true) do h5file
      tmp = PencilArray(w.f.ϕ.pencil, Array{Float64}(undef, size_local(w.f.ϕ)))
      read!(h5file, tmp, "re")
      w.f.ϕ .= tmp
      read!(h5file, tmp, "im")
      w.f.ϕ .+= im .* tmp
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
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField2D{N,FT,FFT,A}} where {N,FT,FFT,A}

   ϕ = w.f.ϕ
   comm = ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)

   r_local = range_local(ϕ)
   r_local = (r_local[1][begin]:r_local[1][end]+1, r_local[2][begin]:r_local[2][end]+1)
   x, y = A(LinRange(w.f.g.xmin,w.f.g.xmax,w.f.g.nx+1)[r_local[1]]), A(LinRange(w.f.g.ymin,w.f.g.ymax,w.f.g.ny+1)[r_local[2]])
   extents = MPI.Allgather(r_local,comm)

   vtkfile = pvtk_grid("$(prefix)-$(istep)", x, y, extents = extents, part = mpi_rank, nparts = mpi_size)
   if N == 1
      tmp = A{FT}(undef, size_local(ϕ))
      tmp .= real.(parent(ϕ))
      vtkfile["re", VTKCellData()] = tmp
      tmp .= imag.(parent(ϕ))
      vtkfile["im", VTKCellData()] = tmp
      tmp .= sqrt.(abs2.(parent(ϕ)))
      vtkfile["module", VTKCellData()] = tmp
   else
      tmp = A{FT}(undef, (2, size_local(ϕ)...))
      tmp[1] .= real.(parent(w.f.ux))
      tmp[2] .= real.(parent(w.f.uy))
      vtkfile["U", VTKCellData()] = tmp
   end

   # write to file
   outfiles = vtk_save(vtkfile)

   # add to pvd
   if mpi_rank == 1
      if istep == 0
         pvd = paraview_collection(w.filename, append=false)
      else
         pvd = paraview_collection(w.filename, append=true)
      end
      pvd[ Δt * istep] = vtkfile
      vtk_save(pvd)
   end

   return nothing
end

function write!(w::WriterVTK{F};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real) where {F<:AbstractField3D{N,FT,FFT,A}} where {N,FT,FFT,A}


   ϕ = w.f.ϕ
   comm = ϕ.pencil.topology.comm
   mpi_rank = MPI.Comm_rank(comm) + 1
   mpi_size = MPI.Comm_size(comm)

   r_local = range_local(ϕ)
   r_local = (r_local[1][begin]:r_local[1][end]+1, r_local[2][begin]:r_local[2][end]+1, r_local[3][begin]:r_local[3][end]+1)
   x, y, z = A(LinRange(w.f.g.xmin,w.f.g.xmax,w.f.g.nx+1)[r_local[1]]), A(LinRange(w.f.g.ymin,w.f.g.ymax,w.f.g.ny+1)[r_local[2]]), A(LinRange(w.f.g.zmin,w.f.g.zmax,w.f.g.nz+1)[r_local[3]])
   extents = MPI.Allgather(r_local,comm)

   tmp = A{FT}(undef, size_local(ϕ))
   vtkfile = pvtk_grid("$(prefix)-$(istep)", x, y, z, extents = extents, part = mpi_rank, nparts = mpi_size)
   if N == 1
      tmp .= real.(parent(ϕ))
      vtkfile["re", VTKCellData()] = tmp
      tmp .= imag.(parent(ϕ))
      vtkfile["im", VTKCellData()] = tmp
      tmp .= sqrt.(abs2.(parent(ϕ)))
      vtkfile["module", VTKCellData()] = tmp
   else
      tmp = A{FT}(undef, (3, size_local(ϕ)...))
      tmp[1,:,:,:] .= real.(parent(w.f.ux))
      tmp[2,:,:,:] .= real.(parent(w.f.uy))
      tmp[3,:,:,:] .= real.(parent(w.f.uz))
      vtkfile["U", VTKCellData()] = tmp
   end

   # write to file
   outfiles = vtk_save(vtkfile)

   # add to pvd
   if mpi_rank == 1
      if istep == 0
         pvd = paraview_collection(w.filename, append=false)
      else
         pvd = paraview_collection(w.filename, append=true)
      end
      pvd[ Δt * istep] = vtkfile
      vtk_save(pvd)
   end

   return nothing
end

function Base.read!(w::WriterVTK{F};
      prefix="res"::AbstractString,
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
      istep=0::Integer,
      Δt=1::Real) where F
   for writer in wc.writerList
      write!(writer,prefix=prefix,istep=istep)
   end
end

function Base.read!(wc::WriterCollection{F};
      prefix="res"::AbstractString,
      istep=0::Integer,
      Δt=1::Real) where F
   for writer in wc.writerList
      read!(writer,prefix=prefix,istep=istep,Δt=Δt)
   end
end