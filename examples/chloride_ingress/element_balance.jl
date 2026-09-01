# The dialogue with ChemistryLab: what goes into an equilibration must come out of it.
#
# This file guards the *interface*, not the chemistry. Whether portlandite has the right
# solubility is ChemistryLab's business and is validated there; whether this package hands
# it a well-formed system and reads the answer back without losing anything is ours, and
# nothing else was checking it.
#
# Two separate questions, kept separate — conflating them is what made an earlier diagnosis
# wrong, and the confusion is worth describing because it is easy to fall into:
#
#   1. **Is the system well formed?** `speciation(substances, seeds; aggregate_state =
#      [AS_AQUEOUS])` returns the *seeded* species whether or not they match the filter, so
#      seeding it with "Portlandite" and then appending the solids lists Portlandite twice.
#      The conservation matrix gets duplicate columns, the equilibrium splits the moles
#      between them, and `moles(state, "Portlandite")` reads back one of the two. That is a
#      construction error, and it must be reported as one.
#
#   2. **Did `equilibrate` conserve what it was given?** Answering this means summing over
#      the columns the system actually holds — *without* deduplicating. Deduplicating here
#      skips the moles parked in a duplicate column and reports them as destroyed matter,
#      which turns a malformed system into a phantom violation of mass conservation and
#      sends the reader looking upstream for a bug that is not there.
#
# The balance is deliberately physical rather than solver-internal. Reading the optimizer's
# return code would be the obvious alternative and it is not portable: on OptimaSolver
# 0.2.7, the version this package pins, `OptimaOptimizer._cache[]` is still `nothing` after
# `equilibrate`. A balance computed from the states themselves needs no cooperation and
# survives a change of solver.

"""
    unique_species(species) -> Vector

`species` with duplicates removed, keeping the first occurrence of each symbol.

Call it on everything handed to `ChemicalSystem`. It is cheap, it is idempotent, and the
failure it prevents does not announce itself: a duplicated column is a perfectly valid
optimisation problem whose answer is meaningless.
"""
function unique_species(species)
    seen = Set{String}()
    kept = similar(collect(species), 0)
    for sp in species
        name = string(symbol(sp))
        name in seen && continue
        push!(seen, name)
        push!(kept, sp)
    end
    return kept
end

"""
    duplicate_species(species) -> Vector{String}

The symbols appearing more than once, in order of first appearance. Empty is what a
well-formed species set looks like.
"""
function duplicate_species(species)
    counts = Dict{String, Int}()
    order = String[]
    for sp in species
        name = string(symbol(sp))
        haskey(counts, name) || push!(order, name)
        counts[name] = get(counts, name, 0) + 1
    end
    return [n for n in order if counts[n] > 1]
end

"""
    element_amounts(state, system) -> Dict{Symbol, Float64}

Moles of each chemical element in `state`, summed over the species of `system` **as it
holds them**.

No deduplication, deliberately: this measures what the equilibration did with the problem
it was posed, and a duplicated column still carries real moles. Use
[`duplicate_species`](@ref) to ask the other question.
"""
function element_amounts(state, system)
    totals = Dict{Symbol, Float64}()
    for sp in system.species
        n = ustrip(moles(state, string(symbol(sp))))
        (isfinite(n) && n != 0) || continue
        for (element, ν) in composition(sp.formula)
            totals[element] = get(totals, element, 0.0) + n * ν
        end
    end
    return totals
end

"""
    element_balance_error(before, after, system; floor = 1e-12) -> (element, relative_error)

Worst relative change of any element across an equilibration, and which element it is.

Elements absent from `before` are skipped rather than reported as an infinite error: a
species set can legitimately gain an element that was seeded at exactly zero.

So are elements present only in trace. Keeping a species in the system without giving it any
matter means seeding it at something like 1e-16 mol — chloride, before any chloride has
arrived — and a *relative* balance on that quantity reports rounding as a catastrophe: a
discrepancy of 3e-16 mol comes out as 289 %. `floor` is a fraction of the largest element
present, below which an element carries no balance worth checking.
"""
function element_balance_error(before, after, system; floor = 1.0e-12)
    a, b = element_amounts(before, system), element_amounts(after, system)
    isempty(a) && return :none, 0.0
    negligible = floor * maximum(abs, values(a))
    worst_element, worst = :none, 0.0
    for (element, n_in) in a
        abs(n_in) > negligible || continue
        err = abs(get(b, element, 0.0) - n_in) / abs(n_in)
        err > worst && ((worst_element, worst) = (element, err))
    end
    return worst_element, worst
end

"""
    check_element_balance(before, after, system; rtol = 1e-2, label = "")

Warn when an equilibration failed to conserve the elements it was given.

`rtol` is loose on purpose, and looser than it looks. The failure this guards against is an
element *disappearing* — half of it, or all of it — which is what a malformed system or a
mangled read-back produces. On a mature-OPC assemblage the round trip currently closes to
about 1e-3 on sulphur and better than that on everything else; that residue is unexplained
and belongs to the equilibrium solve rather than to this interface, so the threshold sits
above it rather than pretending to measure it.

Warns once per session (`maxlog = 1`): the interesting information is *that* it happens and
how badly, not one line per node per time step.
"""
function check_element_balance(before, after, system; rtol = 1.0e-2, label = "")
    element, err = element_balance_error(before, after, system)
    if err > rtol
        @warn "equilibrate did not conserve the elements$(isempty(label) ? "" : " ($label)")" *
            " — this state is not an equilibrium of the problem posed" element relative_error = err rtol maxlog = 1
    end
    return element, err
end
