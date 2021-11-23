"Abstract MPI device."
abstract type MPIDevice <: Device end

"Abstract MPI 1D"
struct MPI1D <: MPIDevice
    comm :: MPI.Comm
    rank :: Integer
    size :: Integer
    topo :: MPITopology
    function MPI1D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0]
        MPI.Dims_create!(size,topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        new(comm, rank, size, topo)
    end
end

"Abstract MPI 2D"
struct MPI2D <: MPIDevice
    comm :: MPI.Comm
    rank :: Integer
    size :: Integer
    topo :: MPITopology
    function MPI2D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0,0]
        MPI.Dims_create!(size,topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        new(comm, rank, size, topo)
    end
end