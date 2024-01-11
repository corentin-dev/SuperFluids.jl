include("domain_topology.jl")

export print_parallel, println_parallel

"Global variable for print"
global _rank = -1

"""
$(SIGNATURES)

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
$(SIGNATURES)

Only prints for rank 0.
"""
function println_parallel(arg...)
    if _rank == 0
        println(arg...)
    elseif _rank == -1
        print("error", arg...)
    end
end
