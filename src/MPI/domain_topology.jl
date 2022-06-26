"""
$(TYPEDEF)

Abstract domain decomposition.
"""
abstract type AbstractDomainDecomposition{N} end

"""

Abstract MPI decomposition.
"""
abstract type AbstractMPIDecomposition{N} <: AbstractDomainDecomposition{N} end

"""
$(TYPEDEF)

MPI topology.

Describes the topology of multiple MPI processes.

$(TYPEDFIELDS)

Standard constructor is:

$(METHODLIST)
"""
struct MPITopo{N} <: AbstractMPIDecomposition{N}
    "communicator"
    comm::MPI.Comm
    "rank of current process"
    rank::Integer
    "number of processes in communicator"
    size::Integer
    "topology of communicator"
    topo::MPITopology

    """
    $(SIGNATURES)

    Final constructor for [`MPITopo`](@ref).
    """
    function MPITopo(N::Integer, comm::MPI.Comm, rank::Integer, size::Integer,
                     topo::MPITopology)
        _rank = rank
        return new{N}(comm, rank, size, topo)
    end
end

"""
$(SIGNATURES)

Constructs a 1D MPI topology.
"""
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
    return MPITopo(1, mpi_comm, mpi_rank, mpi_size, topo)
end

"""
$(SIGNATURES)

Constructs a 2D MPI topology.
"""
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
    return MPITopo(2, mpi_comm, mpi_rank, mpi_size, topo)
end

"""
$(SIGNATURES)

Constructs a 3D MPI topology.
"""
function MPITopo3D()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    topo_dims = [0, 0, 0]
    MPI.Dims_create!(size, topo_dims)
    topo = MPITopology(comm, Tuple(topo_dims))
    return MPITopo(3, mpi_comm, mpi_rank, mpi_size, topo)
end
