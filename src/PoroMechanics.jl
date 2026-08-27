"""
    PoroMechanics.jl

Reactive transport and poromechanics of porous media, on finite volume and finite
element backends: unsaturated flow, non-isothermal drying, Biot poroelasticity and
reactive transport in cementitious materials.

# Architecture

Two layers, so that a physics model stays a self-contained description of its own
equations and knows nothing about time stepping or assembly:

## Layer 1 — Core abstractions (this module)
- [`AbstractPoroModel`](@ref) — supertype of every physics model. Material parameters
  live in the concrete struct; multiple dispatch selects the constitutive behavior.
- [`AbstractPoroSolver`](@ref) — supertype for time-stepping strategies (Monolithic, SNIA…).

## Layer 2 — Physics models
Each model lives in its own file under `src/Models/` and defines:
- A concrete `<Name>Model <: AbstractPoroModel` struct holding material parameters.
- The required interface methods (`storage!`, `flux!`, `bcondition!` for FVM models;
  `assemble_element!` for FEM models).

Jacobians are never written by hand: `VoronoiFVM.jl` differentiates the FVM callbacks
with `ForwardDiff.jl`.

# Backend convention
- **Transport / diffusion / flow** → `VoronoiFVM.jl`
- **Coupled mechanics** (Biot, BBM…) → `Ferrite.jl`

# License
MIT — see `LICENSE`.
"""
module PoroMechanics

using VoronoiFVM: VoronoiFVM

# ── Public re-exports ──────────────────────────────────────────────────────────
export AbstractPoroModel, AbstractPoroSolver
export storage!, flux!, bcondition!, reaction!, assemble_element!
export element_matrices!, facet_load!

# Constitutive layer
export AbstractRetention, VanGenuchten, ExponentialCutoff
export saturation, dsaturation_dpc
export AbstractRelativePermeability, Mualem, PowerLawKrl
export relative_permeability, gas_relative_permeability
export AbstractTortuosity, OhJang, tortuosity

# Backends
export fvm_system

# ── Core abstractions ──────────────────────────────────────────────────────────

# ── Constitutive layer ─────────────────────────────────────────────────────────
# Material laws, owned by the package: retention curves, relative permeabilities,
# tortuosity models. They know nothing about grids, assembly or time stepping, and
# their coefficients are type-parameterised so that a result can be differentiated
# with respect to the parameters themselves.

include("Constitutive/Retention.jl")
include("Constitutive/RelativePermeability.jl")
include("Constitutive/Tortuosity.jl")

"""
    AbstractPoroModel

Supertype for all physics models in PoroMechanics.jl.

Concrete subtypes must implement the methods appropriate for their backend:

### FVM backend (VoronoiFVM.jl) — transport / diffusion
- `storage!(f, u, node, model, data)` — accumulation terms ∂M/∂t
- `flux!(f, u, edge, model, data)`    — inter-node fluxes (Darcy, Fick, Fourier…)
- `bcondition!(f, u, node, model, data)` — boundary conditions

### FEM backend (Ferrite.jl) — coupled mechanics
- `assemble_element!(Ke, re, el, u_el, model, cv, Δt)` — element stiffness & residual

A model must also expose:
- `nspecies(model)::Int` — number of primary unknowns
- `species_names(model)::Vector{Symbol}` — human-readable names for unknowns
"""
abstract type AbstractPoroModel end

"""
    AbstractPoroSolver

Supertype for time-stepping strategies.
"""
abstract type AbstractPoroSolver end

# ── Interface stubs (fall-through implementations raise informative errors) ────

"""
    storage!(f, u, node, model::AbstractPoroModel, data)

Fill `f` with the accumulation (storage) terms for `model` at `node`.
Must be implemented by each concrete FVM model.
"""
function storage!(f, u, node, model::AbstractPoroModel, data)
    error("storage! not implemented for $(typeof(model))")
end

"""
    flux!(f, u, edge, model::AbstractPoroModel, data)

Fill `f` with the inter-node flux terms for `model` at `edge`.
Must be implemented by each concrete FVM model.
"""
function flux!(f, u, edge, model::AbstractPoroModel, data)
    error("flux! not implemented for $(typeof(model))")
end

"""
    bcondition!(f, u, node, model::AbstractPoroModel, data)

Apply boundary conditions for `model` at boundary `node`.
Default: no-flux (Neumann zero). Override for Dirichlet or non-zero Neumann.
"""
function bcondition!(f, u, node, model::AbstractPoroModel, data)
    # Default: zero-flux (do nothing)
    return nothing
end

"""
    reaction!(f, u, node, model::AbstractPoroModel, data)

Fill `f` with volumetric reaction (source/sink) terms and algebraic constraints
for `model` at `node`.

Use cases:
- Chemical reactions between species (e.g. precipitation, equilibrium)
- Algebraic constraints (e.g. electroneutrality: `f[iψ] = Σ zᵢ·cᵢ`)

Default: no reaction. Override for models with coupled chemistry or constraints.
"""
function reaction!(::Any, ::Any, ::Any, ::AbstractPoroModel, ::Any)
    # Default: no reaction
    return nothing
end

"""
    assemble_element!(Ke, re, el, u_el, model::AbstractPoroModel, cv, Δt)

Assemble the element stiffness matrix `Ke` and residual vector `re` for FEM models.
Must be implemented by each concrete Ferrite-based model.
"""
function assemble_element!(Ke, re, el, u_el, model::AbstractPoroModel, cv, Δt)
    error("assemble_element! not implemented for $(typeof(model))")
end

"""
    element_matrices!(ke1, ke2, cell, model::AbstractPoroModel, cv_u, cv_p)

Fill the stationary element matrix `ke1` and the storage element matrix `ke2`
for linear coupled FEM models (Biot poroelasticity and similar).

The global system is solved at each time step as:
    A · xⁿ⁺¹ = f_ext + (1/Δt) · K2 · xⁿ,   with A = K1 + K2/Δt

where `K1 = Σ ke1` (elastic rigidity + Darcy conductivity) and
      `K2 = Σ ke2` (Biot coupling + compressibility storage).

Must be implemented by each concrete Ferrite-based model that uses this split.
"""
function element_matrices!(_ke1, _ke2, _cell, model::AbstractPoroModel, _cv_u, _cv_p)
    error("element_matrices! not implemented for $(typeof(model))")
end

"""
    facet_load!(fe, facet, model::AbstractPoroModel, fv)

Accumulate the surface load vector `fe` for boundary facet `facet`.
Default: no traction (zero Neumann). Override for pressure or traction loads.
"""
function facet_load!(fe, facet, model::AbstractPoroModel, fv)
    return nothing
end

# ── Backends ───────────────────────────────────────────────────────────────────
# Glue to the solver packages. The physics lives above; these only wire it up.

include("Backends/FVM.jl")

# ── Helper: number of species ──────────────────────────────────────────────────

"""
    nspecies(model::AbstractPoroModel) -> Int

Return the number of primary unknowns (species) solved by `model`.
"""
function nspecies(::AbstractPoroModel)
    error("nspecies not implemented for this model")
end

"""
    species_names(model::AbstractPoroModel) -> Vector{Symbol}

Return the symbolic names of the primary unknowns, e.g. `[:p_l, :T]`.
"""
function species_names(::AbstractPoroModel)
    error("species_names not implemented for this model")
end

end # module PoroMechanics
