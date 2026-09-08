using ChemistryLab
using OptimaSolver
using DynamicQuantities

"""
    certified_initial_equilibrium(state; initial_guess = nothing) -> (equilibrium, certificate)

Initialize an OPC composition using ChemistryLab's certified solver. The interior-point
solution is only a starting guess; a state is returned only if the KKT certificate passes
against the element totals of the original input. This is an initialization adapter, not
a differentiable replacement for the transient chemistry callbacks.
"""
function certified_initial_equilibrium(state; initial_guess = nothing)
    des = ChemistryLab.DualEquilibriumSolver(state.system)
    b = des.A * [ustrip(us"mol", n) for n in state.n]
    guess = initial_guess === nothing ?
        equilibrate(state, OptimaOptimizer(; tol = 1.0e-10, verbose = false)) : initial_guess
    eq, certificate = ChemistryLab.solve_certified(des, [guess, state]; b)
    certificate !== nothing && certificate.optimal && certificate.n_interior > 0 ||
        error("OPC initial equilibrium is not certified: $certificate")
    return eq, certificate
end
