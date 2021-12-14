"Abstract field decomposition"
abstract type AbstractFieldDecomposition{DD} end

"Field decomposition"
struct NoFieldDecomposition{DD} <: AbstractFieldDecomposition{DD}
    dd :: DD
    r :: Tuple
    n :: Tuple
    function NoFieldDecomposition(dd, r::Tuple, n::Tuple)
        new{typeof(dd)}(dd, r, n)
    end
end

"Field decomposition"
struct MPIFieldDecomposition{DD} <: AbstractFieldDecomposition{DD}
    dd :: DD
    r :: Tuple
    n :: Tuple
    function MPIFieldDecomposition(dd, r::Tuple, n::Tuple)
        new{typeof(dd)}(dd, r, n)
    end
end