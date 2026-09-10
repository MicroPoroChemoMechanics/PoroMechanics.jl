# sorbing_transport.jl — multi-ionic transport whose storage is the inventory
#
# `nernst_planck.jl` transports ions. This adds the surface, and it adds it in the one
# place that makes the scheme conservative: the **storage**.
#
#     T_i = φ c_i + S_i(c, β)
#
# `run_4.jl` writes `(φ + K_d) c` instead, with `K_d = dS/dc` frozen between two chemistry
# passes. That is a consistent linearisation of the same balance — see
# `CLAUDE/gia_formulation.md` §3, constats 1 and 2 — but the quantity it conserves is
# `(φ + K_d) c`, which is not the inventory. `test/reactive/conservation.jl` measures the
# gap: on a Langmuir tracer, 7e-14 for the inventory form against 0.79 for the frozen one.
#
# ## What changes, mechanically
#
# `S_i` here is a **function of the unknowns**, not a field refreshed between segments. So
# `VoronoiFVM` forms `(T(uⁿ⁺¹) − T(uⁿ))/Δt` on the real inventory, and differentiates it:
# the whole matrix `∂S_i/∂c_j` lands in the Jacobian, exactly, including the cross terms
# `run_4.jl` throws away (constat 3) and without the lag that constat 1 is about.
#
# The surface potential β is still found by a root-find inside `solve_dlm`, once per node
# per Newton iteration. That is the *reduced Newton* of step E applied to the surface
# alone, at the scale where it is cheap: a scalar bisection, not an interior-point solve.
# Making β a nodal unknown instead — `dlm_residual` is there for it — is the next rung, and
# the one that removes the root-find altogether.
#
# ## What does not change yet
#
# The mineral terms `Σ ν_im n_m` are **not** here. Frozen over a segment they would be a
# constant in the storage, which cancels exactly in `(T(uⁿ⁺¹) − T(uⁿ))` and contributes
# nothing — writing them down would look conservative and do nothing. The mineral exchange
# stays a per-segment jump until it becomes an unknown with its complementarity, §9.

using PoroMechanics

"""
    IonIndex(; Cl, Na, K, Ca, OH, Mg = 0)

Where each ion the double layer needs sits in the transported vector. `Mg = 0` means the
system has none, and the model passes a zero concentration — magnesium at zero contributes
nothing to the site sum, the charge sum or the ionic strength, which is why `dlm.jl` needs
no separate binary implementation.
"""
Base.@kwdef struct IonIndex
    Cl::Int
    Na::Int
    K::Int
    Ca::Int
    OH::Int
    Mg::Int = 0
end

"""
    SorbingTransport(; transport, dlm, ions, n_csh, x_cas)

Composition, not extension: `transport` owns the fluxes, the boundary conditions and the
electroneutrality constraint, and this wrapper adds the surface to the storage. Every
callback other than `storage!` is delegated unchanged.

| field | meaning |
|---|---|
| `transport` | a [`NernstPlanck`](@ref) |
| `dlm` | a `DLM`, from `dlm.jl` |
| `ions` | an [`IonIndex`](@ref) |
| `n_csh` | C-S-H content [mol·m⁻³ of medium] |
| `x_cas` | Ca/Si ratio of the C-S-H |
"""
Base.@kwdef struct SorbingTransport{TR, DL, T} <: PoroMechanics.AbstractPoroModel
    transport::TR
    dlm::DL
    ions::IonIndex
    n_csh::T = 635.0
    x_cas::T = 1.5
end

## Delegation, not reimplementation: the wrapper is not a different physics, it is the same
## transport with one more term in one callback.
PoroMechanics.nspecies(m::SorbingTransport) = PoroMechanics.nspecies(m.transport)
PoroMechanics.species_names(m::SorbingTransport) = PoroMechanics.species_names(m.transport)

PoroMechanics.flux!(f, u, edge, m::SorbingTransport, data) =
    PoroMechanics.flux!(f, u, edge, m.transport, data)
PoroMechanics.reaction!(f, u, node, m::SorbingTransport, data) =
    PoroMechanics.reaction!(f, u, node, m.transport, data)
PoroMechanics.bcondition!(f, u, bnode, m::SorbingTransport, data) =
    PoroMechanics.bcondition!(f, u, bnode, m.transport, data)

"""
    adsorbed(m::SorbingTransport, u) -> (S_Cl, S_Na, S_K, S_Ca, S_Mg)

The double layer loadings at the local composition [mol·m⁻³ of medium]. `u` is a node's
unknown vector, so this is what `storage!` adds and what the AD differentiates.
"""
function adsorbed(m::SorbingTransport, u)
    ix = m.ions
    z = zero(eltype(u))
    c_Mg = ix.Mg == 0 ? z : max(u[ix.Mg], z)
    _, S_Cl, S_Na, S_K, S_Ca, S_Mg = solve_dlm(
        max(u[ix.Cl], z), max(u[ix.Na], z), max(u[ix.K], z), max(u[ix.Ca], z), c_Mg,
        max(u[ix.OH], eps(one(eltype(u)))), m.n_csh, m.x_cas;
        dlm = m.dlm,
    )
    return S_Cl, S_Na, S_K, S_Ca, S_Mg
end

"""
    storage!(f, u, node, m::SorbingTransport, data)

`T_i = φ c_i + S_i(c)`. The ions the double layer does not bind keep the transport term
alone, and the potential keeps its zero storage.

The `max(·, 0)` guards inside [`adsorbed`](@ref) matter here: Newton proposes negative
concentrations on the way to a solution, and the double layer expressions are not defined
there. Clamping the *argument* rather than the result keeps the derivative continuous at
zero, which a clamp on the output would not.
"""
function PoroMechanics.storage!(f, u, node, m::SorbingTransport, data)
    PoroMechanics.storage!(f, u, node, m.transport, data)
    S_Cl, S_Na, S_K, S_Ca, S_Mg = adsorbed(m, u)
    ix = m.ions
    f[ix.Cl] += S_Cl
    f[ix.Na] += S_Na
    f[ix.K] += S_K
    f[ix.Ca] += S_Ca
    ix.Mg == 0 || (f[ix.Mg] += S_Mg)
    return nothing
end

"""
    inventory_node!(m::SorbingTransport)

The physical inventory as a node function, for `conservation_defect`. Identical to
`storage!` here — which is the point: the scheme stores what the physics conserves, so the
harness reports round-off. Hand it to a frozen-`K_d` scheme instead and it reports the gap.
"""
inventory_node!(m::SorbingTransport) =
    (f, u, node, data = nothing) -> PoroMechanics.storage!(f, u, node, m, data)

# ── β as a nodal unknown ──────────────────────────────────────────────────────
#
# [`SorbingTransport`](@ref) finds the surface potential by a root-find inside the storage
# term. That works, and it has a floor: `solve_dlm` brackets β to 1e-9 and
# `dS_Cl/dβ ≈ 0.2`, so the inventory is only defined to about 1e-9 relative and the
# conservation defect settles at 1e-8 instead of round-off.
#
# The alternative carries β as an unknown and states the Gouy-Chapman balance as its
# equation. Nothing is solved inside a callback any more: the loadings are explicit in
# `(β, c)` and the residual is one more algebraic row, exactly like the electroneutrality
# constraint next to it. That is the globally implicit treatment, on the sub-system where
# it costs one unknown per node.
#
# The two models are the same physics written twice, which is the point: they must agree.

"""
    SurfaceResolvedTransport(; transport, dlm, ions, n_csh, x_cas)

Same as [`SorbingTransport`](@ref), with the surface potential promoted to an unknown.

The unknown vector is the ions, then `Ψ`, then `β`, so `nspecies` is one more than the
transport's. `β` needs no boundary condition: its equation is node-local.
"""
Base.@kwdef struct SurfaceResolvedTransport{TR, DL, T} <: PoroMechanics.AbstractPoroModel
    transport::TR
    dlm::DL
    ions::IonIndex
    n_csh::T = 635.0
    x_cas::T = 1.5
end

PoroMechanics.nspecies(m::SurfaceResolvedTransport) =
    PoroMechanics.nspecies(m.transport) + 1
ibeta(m::SurfaceResolvedTransport) = PoroMechanics.nspecies(m)

PoroMechanics.species_names(m::SurfaceResolvedTransport) =
    vcat(PoroMechanics.species_names(m.transport), [:β])

PoroMechanics.bcondition!(f, u, bnode, m::SurfaceResolvedTransport, data) =
    PoroMechanics.bcondition!(f, u, bnode, m.transport, data)

"""
    loadings(m::SurfaceResolvedTransport, u) -> (S_Cl, S_Na, S_K, S_Ca, S_Mg)

The double layer loadings, **explicit** in the unknowns: `β` is read from `u`, not solved
for. No root-find, so no tolerance to inherit.
"""
function loadings(m::SurfaceResolvedTransport, u)
    ix = m.ions
    z = zero(eltype(u))
    c_Mg = ix.Mg == 0 ? z : max(u[ix.Mg], z)
    return dlm_loadings(
        u[ibeta(m)],
        max(u[ix.Cl], z), max(u[ix.Na], z), max(u[ix.K], z), max(u[ix.Ca], z), c_Mg,
        max(u[ix.OH], eps(one(eltype(u)))), m.n_csh, m.x_cas; dlm = m.dlm,
    )
end

function PoroMechanics.storage!(f, u, node, m::SurfaceResolvedTransport, data)
    PoroMechanics.storage!(f, u, node, m.transport, data)
    S_Cl, S_Na, S_K, S_Ca, S_Mg = loadings(m, u)
    ix = m.ions
    f[ix.Cl] += S_Cl
    f[ix.Na] += S_Na
    f[ix.K] += S_K
    f[ix.Ca] += S_Ca
    ix.Mg == 0 || (f[ix.Mg] += S_Mg)
    f[ibeta(m)] = zero(eltype(f))          # algebraic
    return nothing
end

function PoroMechanics.flux!(f, u, edge, m::SurfaceResolvedTransport, data)
    PoroMechanics.flux!(f, u, edge, m.transport, data)
    f[ibeta(m)] = zero(eltype(f))          # the surface does not move
    return nothing
end

"""
    reaction!(f, u, node, m::SurfaceResolvedTransport, data)

The transport's constraints, plus the Gouy-Chapman charge balance as `β`'s own equation.

The residual is **scaled by `F Γ_max`**, which is what makes it dimensionless and `O(1)`.
Unscaled it is `O(0.1)` next to concentration rows of `O(500)`, and VoronoiFVM measures
Newton convergence on the norm of the whole vector: a row four orders below the others is
declared converged long before it is.
"""
function PoroMechanics.reaction!(f, u, node, m::SurfaceResolvedTransport, data)
    PoroMechanics.reaction!(f, u, node, m.transport, data)
    ix = m.ions
    z = zero(eltype(u))
    c_Mg = ix.Mg == 0 ? z : max(u[ix.Mg], z)
    scale = _F_FARADAY * gamma_max(m.dlm, m.x_cas)
    f[ibeta(m)] = dlm_residual(
        u[ibeta(m)],
        max(u[ix.Cl], z), max(u[ix.Na], z), max(u[ix.K], z), max(u[ix.Ca], z), c_Mg,
        max(u[ix.OH], eps(one(eltype(u)))), m.x_cas; dlm = m.dlm,
    ) / scale
    return nothing
end
