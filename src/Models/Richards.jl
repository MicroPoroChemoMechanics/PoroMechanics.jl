"""
Richards' equation: single-phase flow in an unsaturated porous medium.

The liquid moves under its own pressure gradient and under gravity while the gas phase is
assumed to stay at a constant pressure — the classical approximation that reduces two-phase
flow to one equation. With ``p_l`` the unknown and ``p_c = p_g - p_l`` the capillary
pressure,

```math
\\frac{\\partial}{\\partial t}\\big(\\rho_l\\,\\phi\\,S_l(p_c)\\big)
  + \\nabla\\cdot\\mathbf{w}_l = 0, \\qquad
\\mathbf{w}_l = -\\frac{\\rho_l k_{int} k_{rl}(p_c)}{\\mu_l}
              \\left(\\nabla p_l - \\rho_l \\mathbf{g}\\right)
```

Everything nonlinear sits in ``S_l`` and ``k_{rl}``, which come from the constitutive layer
rather than being written out here: [`VanGenuchten`](@ref) or [`ExponentialCutoff`](@ref)
for retention, [`Mualem`](@ref), [`PowerLawKrl`](@ref) or [`GardnerKrl`](@ref) for relative
permeability. Any combination works, and each carries its coefficients as type parameters,
so a result can be differentiated with respect to them.
"""

"""
    RichardsModel(; phi, rho_l, k_int, mu_l, p_g, gravity, gravity_axis, retention, rel_perm, dirichlet)

Unsaturated single-phase flow. The unknown is the liquid pressure ``p_l`` [Pa].

| field | meaning |
|---|---|
| `phi` | porosity [-] |
| `rho_l` | liquid density [kg/m³] |
| `k_int` | intrinsic permeability [m²] — a scalar, or one value per cell region |
| `mu_l` | dynamic viscosity [Pa·s] |
| `p_g` | gas pressure, held constant [Pa] |
| `gravity` | signed component of gravity along `gravity_axis` [m/s²] |
| `gravity_axis` | coordinate axis gravity acts along: 1 for ``x``, 2 for ``y``, 3 for ``z`` |
| `retention` | ``S_l(p_c)`` |
| `rel_perm` | ``k_{rl}(p_c)`` |
| `dirichlet` | imposed pressures, as `((region, value), …)` |

`gravity_axis` defaults to 1, which is the only orientation a 1D column along ``x`` can
have. It has to be said out loud in two or more dimensions: a vertical column meshed in the
``(x, y)`` plane is driven by gravity along ``y``, and projecting it on ``x`` silently
removes the drainage rather than failing.

`dirichlet` carries the boundary data because that is what distinguishes one *case* from
another, not one *model* from another: a Richards model with the pressure imposed on the
right is the same physics as one with it imposed on the left. Hard-wiring a region number
into the model would make the model unusable for the next problem, which is exactly what
happens when a physics model is written inside a script.

Boundaries not named are no-flow, which is `VoronoiFVM`'s default and the usual meaning of
an unlisted boundary in a flow problem.

A layered or composite medium is likewise a *case*, not a second model: pass a collection
for `k_int`, indexed by the cell region, and the same struct covers it.

```julia
RichardsModel(; k_int = [8.9e-12, 8.9e-13], gravity = -9.81, gravity_axis = 2, …)
```
"""
Base.@kwdef struct RichardsModel{T, P, R, K, B} <: AbstractPoroModel
    phi::T = 0.3
    rho_l::T = 1.0e3
    k_int::P = 1.0e-20
    mu_l::T = 1.0e-3
    p_g::T = 1.0e5
    gravity::T = 0.0
    gravity_axis::Int = 1
    retention::R = VanGenuchten(1.5e6, 0.06)
    rel_perm::K = Mualem(3.0e6, 0.5)
    dirichlet::B = ()
end

## Promote rather than require a single type: differentiating with respect to one parameter
## makes that field a `Dual` while the rest stay `Float64`. Two fields stay out of the
## promotion — `gravity_axis`, which indexes a coordinate rather than carrying physics, and
## `k_int`, which may be a whole collection of permeabilities.
function RichardsModel(
        phi, rho_l, k_int, mu_l, p_g, gravity, gravity_axis, retention, rel_perm, dirichlet
    )
    phi′, rho_l′, mu_l′, p_g′, gravity′ = promote(phi, rho_l, mu_l, p_g, gravity)
    return RichardsModel(
        phi′, rho_l′, k_int, mu_l′, p_g′, gravity′,
        gravity_axis, retention, rel_perm, dirichlet
    )
end

"""
    intrinsic_permeability(m::RichardsModel, region) -> k

Permeability of cell region `region`. Dispatch rather than a runtime branch, so a
single-region model pays nothing for the generality and a `Vector{<:Dual}` still
differentiates.
"""
intrinsic_permeability(m::RichardsModel{<:Any, <:Number}, region) = m.k_int
intrinsic_permeability(m::RichardsModel, region) = m.k_int[region]

nspecies(::RichardsModel) = 1
species_names(::RichardsModel) = [:p_l]

"""
    liquid_saturation(m::RichardsModel, pc)

``S_l(p_c)`` from the model's retention curve.
"""
liquid_saturation(m::RichardsModel, pc) = saturation(m.retention, pc)

"""
    liquid_conductivity(m::RichardsModel, pc, region = 1) -> ρ_l k_int k_rl(p_c) / μ_l

The coefficient in front of ``\\nabla p_l`` in the mass flux, in kg/(m·s·Pa). `region`
selects the permeability when the model carries one per cell region.
"""
liquid_conductivity(m::RichardsModel, pc, region = 1) =
    m.rho_l * intrinsic_permeability(m, region) / m.mu_l *
    relative_permeability(m.rel_perm, pc)

"""Storage term ``\\rho_l \\phi S_l(p_c)``."""
function storage!(f, u, node, m::RichardsModel, data)
    f[1] = m.rho_l * m.phi * liquid_saturation(m, m.p_g - u[1])
    return nothing
end

"""
Two-point flux, with the conductivity evaluated at the mean capillary pressure of the edge.

`VoronoiFVM` divides `f` by the edge length, which is why the gravity term carries an
explicit edge projection while the pressure term does not. That projection is taken along
`gravity_axis`, so the same model drives a horizontal column and a vertical one.

Note `edge.coord` is the coordinate matrix of the **whole grid**, not of the edge's two
nodes: the endpoints are `edge.coord[:, edge.node[1]]` and `edge.coord[:, edge.node[2]]`.
Writing `edge.coord[a, 2] - edge.coord[a, 1]` instead reads global nodes 1 and 2 whatever
edge is being assembled. On a uniform 1D grid those happen to be adjacent and equally
spaced, so the mistake returns the right number and hides; in two dimensions it silently
removes gravity and the column stops draining.
"""
function flux!(f, u, edge, m::RichardsModel, data)
    pl1, pl2 = u[1, 1], u[1, 2]
    K = liquid_conductivity(m, m.p_g - (pl1 + pl2) / 2, edge.region)
    a = m.gravity_axis
    dx = edge.coord[a, edge.node[2]] - edge.coord[a, edge.node[1]]
    f[1] = K * (pl1 - pl2) + K * m.rho_l * m.gravity * dx
    return nothing
end

"""Imposed pressures, from the model's `dirichlet` field."""
function bcondition!(f, u, bnode, m::RichardsModel, data)
    for (region, value) in m.dirichlet
        VoronoiFVM.boundary_dirichlet!(f, u, bnode; species = 1, region = region, value = value)
    end
    return nothing
end
