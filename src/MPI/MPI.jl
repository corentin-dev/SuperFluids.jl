include("domain_topology.jl")
include("domain_decomposition.jl")

"Global variable for print"
_rank = 0

"""
    print_parallel(arg...)

Only prints for rank 0.
"""
function print_parallel(arg...)
    global _rank
    if _rank == 0
        print(arg...)
    end
end

"""
    println_parallel(arg...)

Only prints for rank 0.
"""
function println_parallel(arg...)
    global _rank
    if _rank == 0
        println(arg...)
    end
end