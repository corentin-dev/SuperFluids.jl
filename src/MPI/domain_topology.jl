"Abstract domain decomposition"
abstract type AbstractDomainDecomposition end

"Abstract MPI decomposition."
abstract type AbstractMPIDecomposition <: AbstractDomainDecomposition end

"MPI 1D"
struct MPITopo1D <: AbstractMPIDecomposition
    comm::MPI.Comm
    rank::Integer
    size::Integer
    topo::MPITopology
    function MPITopo1D()
        MPI.Init()
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = MPI.Comm_rank(mpi_comm)
        mpi_size = MPI.Comm_size(mpi_comm)
        topo_dims = [0]
        MPI.Dims_create!(mpi_size, topo_dims)
        topo = MPITopology(mpi_comm, Tuple(topo_dims))
        global _rank
        _rank = mpi_rank
        return new(mpi_comm, mpi_rank, mpi_size, topo)
    end
end

"MPI 2D"
struct MPITopo2D <: AbstractMPIDecomposition
    comm::MPI.Comm
    rank::Integer
    size::Integer
    topo::MPITopology
    function MPITopo2D()
        MPI.Init()
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = MPI.Comm_rank(mpi_comm)
        mpi_size = MPI.Comm_size(mpi_comm)
        topo_dims = [0, 0]
        MPI.Dims_create!(mpi_size, topo_dims)
        topo = MPITopology(mpi_comm, Tuple(topo_dims))
        global _rank
        _rank = mpi_rank
        return new(mpi_comm, mpi_rank, mpi_size, topo)
    end
end

"MPI 3D"
struct MPITopo3D <: AbstractMPIDecomposition
    comm::MPI.Comm
    rank::Integer
    size::Integer
    topo::MPITopology
    function MPITopo3D()
        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        size = MPI.Comm_size(comm)
        topo_dims = [0, 0, 0]
        MPI.Dims_create!(size, topo_dims)
        topo = MPITopology(comm, Tuple(topo_dims))
        return new(comm, rank, size, topo)
    end
end
