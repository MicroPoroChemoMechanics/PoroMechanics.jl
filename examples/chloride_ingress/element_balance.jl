# Element balance across a chemical equilibration.
#
# `equilibrate` redistributes matter between species; it must not create or destroy
# elements. Nothing was checking that, and the omission is not hypothetical: on
# ChemistryLab 0.13.0 the same call loses exactly half the calcium of an ordinary
# portlandite/monosulphate/ettringite assemblage — see
# `CLAUDE/chemistrylab_maxiters_mwe.jl`.
#
# The check is deliberately *physical* rather than solver-internal. Reading the
# optimizer's return code would be the obvious alternative, and it is not available:
# on OptimaSolver 0.2.7, the version this package pins, `OptimaOptimizer._cache[]` is
# still `nothing` after `equilibrate`. A balance computed from the states themselves
# needs no such cooperation and survives a change of solver.

"""
    element_amounts(state, system) -> Dict{Symbol, Float64}

Moles of each chemical element in `state`, summed over every species.

Species are deduplicated by symbol. A system built by appending a solid to a seeded
speciation can list the same species twice, and a double-counted element balances
against itself — which hides exactly the discrepancy this function exists to find.
"""
function element_amounts(state, system)
    seen = Set{String}()
    totals = Dict{Symbol, Float64}()
    for sp in system.species
        name = string(symbol(sp))
        name in seen && continue
        push!(seen, name)
        n = ustrip(moles(state, name))
        (isfinite(n) && n != 0) || continue
        for (element, ν) in composition(sp.formula)
            totals[element] = get(totals, element, 0.0) + n * ν
        end
    end
    return totals
end

"""
    element_balance_error(before, after, system) -> (element, relative_error)

Worst relative change of any element across an equilibration, and which element it is.

Elements absent from `before` are skipped rather than reported as an infinite error:
a species set can legitimately gain an element that was seeded at exactly zero.
"""
function element_balance_error(before, after, system)
    a, b = element_amounts(before, system), element_amounts(after, system)
    worst_element, worst = :none, 0.0
    for (element, n_in) in a
        n_in == 0 && continue
        err = abs(get(b, element, 0.0) - n_in) / abs(n_in)
        err > worst && ((worst_element, worst) = (element, err))
    end
    return worst_element, worst
end

"""
    check_element_balance(before, after, system; rtol = 1e-6, label = "")

Warn when an equilibration failed to conserve the elements it was given.

`rtol` is loose on purpose: an equilibrium solve is iterative and a balance closed to
1e-6 is a converged one. The failure this guards against is not a rounding error — it
is half an element going missing.

Warns once per session (`maxlog = 1`): the interesting information is *that* it
happens and how badly, not one line per node per time step.
"""
function check_element_balance(before, after, system; rtol = 1.0e-6, label = "")
    element, err = element_balance_error(before, after, system)
    if err > rtol
        @warn "equilibrate did not conserve the elements$(isempty(label) ? "" : " ($label)")" *
            " — this state is not an equilibrium of the problem posed" element relative_error = err rtol maxlog = 1
    end
    return element, err
end
