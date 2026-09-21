"""
    solve_biot(K1, K2, constraints; inival, times, load=nothing, on_step=nothing, linsolve=\\)
        -> Vector

Solve `K2 * dx/dt + K1 * x = load(t)` by backward Euler at each supplied time.
`times` includes the initial time and must be finite and strictly increasing.
The matrices are constant, unconstrained sparse matrices with the same sparsity
pattern, including any affine-constraint couplings. [`assemble_biot_matrices`](@ref)
builds this pattern when given `constraints`.

`load` is a constant vector, a function `t -> vector` evaluated at each new time,
or `nothing` for zero loading. Allocate the matrices and initial values in a type
that can hold all coefficients and loads. The default linear solver is `\\`;
`linsolve(A, rhs)` can supply a different solver (for example, a dense solve for
small systems carrying `ForwardDiff.Dual` values).

Initial Dirichlet and affine constraints are applied to a copy of `inival`.
At every step the constraint values are updated, the matrix is rebuilt as
`K1 + K2 / dt`, and the constrained system is solved. Affine slave values are
reconstructed after solving. Factorizations are not cached.

`on_step(x, t, step)` runs after each solve, with `step` starting at 1. It can
record diagnostics or save `copy(x)` for a time history; treat `x` as read-only,
since its buffer is reused. Only the final vector is returned. The input matrices,
initial vector, and load vectors are preserved; the constraint handler's time
and values are updated in place.
"""
function solve_biot(
        K1, K2, constraints; inival, times, load = nothing, on_step = nothing, linsolve = \,
    )
    ts = collect(times)
    isempty(ts) && throw(ArgumentError("times must contain at least the initial time"))
    all(isfinite, ts) || throw(ArgumentError("times must be finite"))
    all(>(0), diff(ts)) || throw(ArgumentError("times must be strictly increasing"))
    n = length(inival)
    size(K1) == size(K2) == (n, n) || throw(DimensionMismatch("matrices must match inival"))
    T = promote_type(eltype(K1), eltype(K2), eltype(inival))
    x = Vector{T}(inival)
    A = similar(K1, T)
    rhs = similar(x)
    Ferrite.update!(constraints, first(ts))
    Ferrite.apply!(x, constraints)
    for step in 1:(length(ts) - 1)
        t = ts[step + 1]
        inv_dt = inv(t - ts[step])
        combine!(A, K1, K2, inv_dt)
        if load === nothing
            fill!(rhs, zero(T))
        else
            f = load isa AbstractVector ? load : load(t)
            length(f) == n || throw(DimensionMismatch("load must match inival"))
            copyto!(rhs, f)
        end
        mul!(rhs, K2, x, inv_dt, one(inv_dt))
        Ferrite.update!(constraints, t)
        Ferrite.apply!(A, rhs, constraints)
        copyto!(x, linsolve(A, rhs))
        Ferrite.apply!(x, constraints)
        on_step === nothing || on_step(x, t, step)
    end
    return x
end
