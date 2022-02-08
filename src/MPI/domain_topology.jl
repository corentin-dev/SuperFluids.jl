"Abstract domain decomposition"
abstract type AbstractDomainDecomposition end

"Abstract MPI decomposition."
abstract type AbstractMPIDecomposition <: AbstractDomainDecomposition end

"MPI 1D"
struct MPITopo1D <: AbstractMPIDecomposition
    comm :: MPI.Comm
    rank :: Integer
    size :: Integer
    topo :: MPITopology
    function MPITopo1D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0]
        MPI.Dims_create!(size,topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        global _rank
        _rank = rank
        new(comm, rank, size, topo)
    end
end

"MPI 2D"
struct MPITopo2D <: AbstractMPIDecomposition
    comm :: MPI.Comm
    rank :: Integer
    size :: Integer
    topo :: MPITopology
    function MPITopo2D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0,0]
        MPI.Dims_create!(size,topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        global _rank
        _rank = rank
        new(comm, rank, size, topo)
    end
end

"MPI 3D"
struct MPITopo3D <: AbstractMPIDecomposition
    comm :: MPI.Comm
    rank :: Integer
    size :: Integer
    topo :: MPITopology
    function MPITopo3D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0,0,0]
        MPI.Dims_create!(size,topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        new(comm, rank, size, topo)
    end
end