# # Equilibrium-coupled component transport
#
# Unknowns are component inventories in mol/m³ of medium. The complete signed
# conservation vector is passed as `b`, separately from a physical starting state.
# ChemistryLab owns equilibrium, certification and implicit sensitivities.
# This prototype uses one Fickian diffusivity per component; migration is omitted.

using PoroMechanics
using ChemistryLab
using DynamicQuantities
using OptimaSolver
using ForwardDiff

"""
    ComponentSet(; names, z, D, fixed)

Transported primaries, charges and diffusivities. `fixed` is a named tuple of all
remaining primary totals (mol/m³ of medium). The charge row `Zz` defaults to zero.
"""
Base.@kwdef struct ComponentSet{NN, ZZ, DD, FF}
    names::NN
    z::ZZ
    D::DD
    fixed::FF = (;)
end
ncomp(c::ComponentSet) = length(c.names)

struct EquilibratedTransport{C, S, R, P, TT, B, I} <: PoroMechanics.AbstractPoroModel
    components::C
    system::S
    rows::R
    phi::P
    tortuosity::TT
    dirichlet::B
    initial_state::I
end

"""
    equilibrated_transport(components, system; initial_state, phi=0.121,
                           tortuosity=nothing, dirichlet=())

Build a model with a complete basis and a nonnegative starting composition for one
m³ of medium. `initial_state` supplies T, P and a starting guess, never the imposed
inventories. Specify water and aluminum through `fixed` when held constant.
"""
function equilibrated_transport(
        components, system; initial_state, phi = 0.121, tortuosity = nothing, dirichlet = (),
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

PoroMechanics.nspecies(m::EquilibratedTransport) = ncomp(m.components)
PoroMechanics.species_names(m::EquilibratedTransport) = Symbol.("T_", m.components.names)

"""Complete signed totals, including fixed components and the charge row."""
function component_totals(m::EquilibratedTransport, totals)
    length(totals) == ncomp(m.components) || throw(DimensionMismatch("incorrect number of totals"))
    T = promote_type(eltype(totals), map(typeof, values(m.components.fixed))...)
    b = zeros(T, size(m.system.SM.A, 1))
    b[m.rows.transported] .= totals
    b[m.rows.fixed] .= collect(values(m.components.fixed))
    return b
end

"""
    equilibrium_state(m, totals) -> (state, certificate)

Solve with explicit signed totals. Dual-valued seed amounts select ChemistryLab's
implicit differentiation route; their values remain a physical starting composition.
ChemistryLab evaluates the certificate on the primal solution.
"""
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

"""
    speciate(m, totals) -> (aqueous_component_concentrations, ok)

Concentrations are in mol/m³ of solution, summed over all aqueous species.
Uncertified answers return `ok=false`; `flux!` rejects them. Solver exceptions
propagate to VoronoiFVM, whose transient controller can retry a smaller time step.
"""
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

struct LocalEquilibriumError <: Exception end
Base.showerror(io::IO, ::LocalEquilibriumError) = print(io, "local chemical equilibrium is not certified")

function PoroMechanics.storage!(f, u, node, m::EquilibratedTransport, ::Any)
    for i in 1:ncomp(m.components)
        f[i] = u[i]
    end
    return nothing
end

function PoroMechanics.flux!(f, u, edge, m::EquilibratedTransport, ::Any)
    c1, ok1 = speciate(m, @view u[:, 1])
    ok1 || throw(LocalEquilibriumError())
    c2, ok2 = speciate(m, @view u[:, 2])
    ok2 || throw(LocalEquilibriumError())
    τ = m.tortuosity === nothing ? one(m.phi) : tortuosity(m.tortuosity, m.phi, 1)
    for i in 1:ncomp(m.components)
        f[i] = m.components.D[i] * τ * (c1[i] - c2[i])
    end
    return nothing
end

_bc_comp!(f, u, bnode, ::Tuple{}, i) = nothing
function _bc_comp!(f, u, bnode, d::Tuple, i)
    PoroMechanics.apply_dirichlet!(f, u, bnode, first(d); species = i)
    return _bc_comp!(f, u, bnode, Base.tail(d), i + 1)
end
PoroMechanics.bcondition!(f, u, bnode, m::EquilibratedTransport, ::Any) =
    _bc_comp!(f, u, bnode, m.dirichlet, 1)
