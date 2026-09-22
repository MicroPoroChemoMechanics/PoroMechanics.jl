module PoroMechanicsChemistryLabExt

using PoroMechanics
using ChemistryLab
using DynamicQuantities
using ForwardDiff
import PoroMechanics: equilibrated_transport, equilibrium_state, speciate

function equilibrated_transport(
        components::ComponentSet, system::ChemicalSystem; initial_state, phi = 0.121, tortuosity = nothing, dirichlet = (),
    )
    names = collect(components.names)
    fixed_names = string.(collect(keys(components.fixed)))
    prim = string.(symbol.(system.SM.primaries))
    all_names = vcat(names, fixed_names)
    length(unique(all_names)) == length(all_names) ||
        throw(ArgumentError("component names must be unique and disjoint from fixed totals"))
    isempty(setdiff(all_names, prim)) || throw(ArgumentError("unknown primary in component basis"))
    missing = setdiff(prim, vcat(all_names, ["Zz"]))
    isempty(missing) || throw(ArgumentError("missing component totals: $(join(missing, ", "))"))
    length(names) == length(components.z) == length(components.D) ||
        throw(DimensionMismatch("names, z and D must have equal lengths"))
    all(d -> isfinite(d) && d >= 0, components.D) ||
        throw(ArgumentError("diffusivities must be finite and nonnegative"))
    all(isfinite, values(components.fixed)) || throw(ArgumentError("fixed totals must be finite"))
    isfinite(phi) && 0 < phi <= 1 || throw(ArgumentError("porosity must lie in (0, 1]"))
    isempty(dirichlet) || length(dirichlet) == length(names) ||
        throw(DimensionMismatch("provide one boundary tuple per transported component"))
    initial_state.system === system || throw(ArgumentError("initial_state uses a different system"))
    all(n -> isfinite(ustrip(us"mol", n)) && ustrip(us"mol", n) >= 0, initial_state.n) ||
        throw(ArgumentError("initial species amounts must be finite and nonnegative"))
    ustrip(us"mol", moles(initial_state, "H2O@")) > 0 ||
        throw(ArgumentError("initial_state must contain liquid water"))
    rows = (
        transported = [findfirst(==(nm), prim) for nm in names],
        fixed = [findfirst(==(nm), prim) for nm in fixed_names],
    )
    return EquilibratedTransport(components, system, rows, phi, tortuosity, dirichlet, copy(initial_state))
end


function equilibrium_state(m::EquilibratedTransport, totals)
    b = component_totals(m, totals)
    all(isfinite, b) || throw(DomainError(b, "component totals must be finite"))
    for r in eachindex(b)
        all(>=(0), @view(m.system.SM.A[r, :])) && ForwardDiff.value(b[r]) < 0 &&
            throw(DomainError(b[r], "negative total for a nonnegative conservation row"))
    end
    reference = m.initial_state
    seed = [ustrip(us"mol", n) + zero(eltype(b)) for n in reference.n]
    state = ChemicalState(m.system, seed .* us"mol"; T = reference.T[], P = reference.P[])
    return ChemistryLab.equilibrate_certified(state; b)
end

function speciate(m::EquilibratedTransport, totals)
    eq, cert = equilibrium_state(m, totals)
    volume = ustrip(us"m^3", eq.V_phases[].liquid)
    ok = cert !== nothing && cert.optimal && isfinite(volume) && ForwardDiff.value(volume) > 0
    ok || return ntuple(_ -> zero(eltype(totals)), ncomp(m.components)), false
    c = ntuple(ncomp(m.components)) do k
        r = m.rows.transported[k]
        sum(m.system.SM.A[r, j] * ustrip(us"mol", eq.n[j]) for j in m.system.idx_aqueous) / volume
    end
    return c, all(isfinite, c)
end

end
