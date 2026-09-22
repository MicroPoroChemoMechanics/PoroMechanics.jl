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
"""Number of transported components, excluding fixed totals."""
ncomp(c::ComponentSet) = length(c.names)

"""
    EquilibratedTransport

Transport of conserved component inventories [mol/m³ of medium]. Construct it with
[`equilibrated_transport`](@ref), which checks the complete chemical basis and seed.
Storage is the inventory itself. Fluxes use aqueous component concentrations from
certified local equilibrium, with one Fickian diffusivity per component.

This reduced model omits electromigration; `components.z` records charge numbers
but does not close a current balance. Use [`NernstPlanck`](@ref) for explicit ionic
migration. No mineral or surface term is added outside the conserved inventories.
"""
struct EquilibratedTransport{C, S, R, P, TT, B, I} <: AbstractPoroModel
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

Requires `using ChemistryLab, DynamicQuantities` to activate the optional extension.
Also load `OptimaSolver`, which provides ChemistryLab's certified solver.
Use the patched ChemistryLab environment prepared by `scripts/prepare_chemistrylab.jl`.
ChemistryLab owns the equilibrium solve, its certificate and implicit sensitivities.
`D` denotes a bulk effective diffusivity, or a free-water diffusivity multiplied by
`tortuosity = D_eff/D⁰`; no extra porosity factor is applied to the flux.
"""
function equilibrated_transport end

nspecies(m::EquilibratedTransport) = ncomp(m.components)
species_names(m::EquilibratedTransport) = Symbol.("T_", m.components.names)

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
function equilibrium_state end

"""
    speciate(m, totals) -> (aqueous_component_concentrations, ok)

Concentrations are in mol/m³ of solution, summed over all aqueous species.
Uncertified answers return `ok=false`; `flux!` rejects them. Solver exceptions
propagate to VoronoiFVM, whose transient controller can retry a smaller time step.
"""
function speciate end

"""Raised when a transport flux would use an uncertified local equilibrium."""
struct LocalEquilibriumError <: Exception end
Base.showerror(io::IO, ::LocalEquilibriumError) = print(io, "local chemical equilibrium is not certified")

function storage!(f, u, node, m::EquilibratedTransport, ::Any)
    for i in 1:ncomp(m.components)
        f[i] = u[i]
    end
    return nothing
end

function flux!(f, u, edge, m::EquilibratedTransport, ::Any)
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

bcondition!(f, u, bnode, m::EquilibratedTransport, ::Any) =
    _transport_dirichlet!(f, u, bnode, m.dirichlet, 1)
