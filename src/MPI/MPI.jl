include("domain_topology.jl")

"Global variable for print"
global _rank = -1

"""
    print_parallel(arg...)

Only prints for rank 0.
"""
function print_parallel(arg...)
    if _rank == 0
        print(arg...)
    elseif _rank == -1
        print("error", arg...)
    end
end

"""
    println_parallel(arg...)

Only prints for rank 0.
"""
function println_parallel(arg...)
    if _rank == 0
        println(arg...)
    elseif _rank == -1
        print("error", arg...)
    end
end
