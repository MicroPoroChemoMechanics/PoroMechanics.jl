## Head-and-tail recursion over the per-unknown boundary tuples, for the reason
## `apply_dirichlet!` gives: a heterogeneous tuple iterated in a plain loop boxes, and the
## allocations land inside the assembly loop, per facet and per Newton iteration.
_transport_dirichlet!(f, u, bnode, ::Tuple{}, i) = nothing
function _transport_dirichlet!(f, u, bnode, d::Tuple, i)
    apply_dirichlet!(f, u, bnode, first(d); species = i)
    return _transport_dirichlet!(f, u, bnode, Base.tail(d), i + 1)
end
