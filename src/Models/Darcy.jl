"""
Transient single-phase Darcy flow in a saturated porous medium.

The pore pressure ``p`` [Pa] is the only unknown. A compressible fluid in a compressible
skeleton stores mass in proportion to the pressure, at a rate set by the storativity ``S``
[Pa⁻¹], and moves it by Darcy's law:

```math
S\\,\\frac{\\partial p}{\\partial t} - \\nabla\\cdot\\left(\\frac{k}{\\mu}\\,\\nabla p\\right) = 0
```

This is the saturated limit of [`RichardsModel`](@ref) — the same balance with ``S_l = 1``
and ``k_{rl} = 1``, so that nothing nonlinear remains. It is kept as a model of its own
because the linear problem is the one with a closed-form solution, and because a saturated
aquifer or column has no use for a retention curve.

``S`` is left as a single coefficient rather than assembled from the poroelastic constants,
so that a measured storativity can be used directly. [`storage_coefficient`](@ref) builds
the same number from a [`BiotPoroelastic`](@ref) medium when the poroelastic description is
the one at hand.
"""

"""
    DarcyModel(; k_int, mu_l, storativity, dirichlet)

Linear single-phase flow in a saturated porous medium. The unknown is the pore pressure
``p`` [Pa].

| field | meaning |
|---|---|
| `k_int` | intrinsic permeability [m²] — a scalar, or one value per cell region |
| `mu_l` | dynamic viscosity [Pa·s] |
| `storativity` | storage coefficient ``S`` [Pa⁻¹] — a scalar, or one value per cell region |
| `dirichlet` | imposed pressures, as `((region, value), …)` |

Boundaries not named are impermeable. A value in `dirichlet` may be a number, or a function
of time — which is how a pressure is ramped rather than stepped:

```julia
DarcyModel(; dirichlet = ((1, 0.0), (2, t -> p_top * min(1, t / t_c))))
```

The ramp is a property of the *case*, not of Darcy's law, so it belongs in the data and not
in a method. Its practical use is to remove the discontinuity between a zero initial
condition and an imposed pressure, which otherwise forces the step-size controller down to
`Δt_min` at startup: a Dirichlet condition is applied unconditionally, so the jump it
creates does not shrink when `Δt` does.

A layered column is a case as well: pass a collection for `k_int` or `storativity`, indexed
by the cell region.
"""
Base.@kwdef struct DarcyModel{P, T, S, B} <: AbstractPoroModel
    k_int::P = 1.0e-12
    mu_l::T = 1.0e-3
    storativity::S = 1.0e-8
    dirichlet::B = ()
end

## Promote rather than require a single type: a `Dual` in one parameter leaves the rest
## `Float64`, which is what differentiating with respect to that parameter does. Only the
## scalar case promotes; a per-region collection is left alone. The promoted values go to
## the parametric constructor, not back through this method, which would recurse.
function DarcyModel(k_int::Number, mu_l::Number, storativity::Number, dirichlet)
    k, mu, S = promote(k_int, mu_l, storativity)
    return DarcyModel{typeof(k), typeof(mu), typeof(S), typeof(dirichlet)}(k, mu, S, dirichlet)
end

intrinsic_permeability(m::DarcyModel{<:Number}, region) = m.k_int
intrinsic_permeability(m::DarcyModel, region) = m.k_int[region]

"""
    storativity(m::DarcyModel, region) -> S

Storage coefficient of cell region `region`, in Pa⁻¹.
"""
storativity(m::DarcyModel{<:Any, <:Any, <:Number}, region) = m.storativity
storativity(m::DarcyModel, region) = m.storativity[region]

"""
    mobility(m::DarcyModel, region) -> k/μ

The coefficient in front of ``\\nabla p`` in Darcy's law, in m²/(Pa·s).
"""
mobility(m::DarcyModel, region = 1) = intrinsic_permeability(m, region) / m.mu_l

nspecies(::DarcyModel) = 1
species_names(::DarcyModel) = [:p]

"""Storage term ``S p``."""
function storage!(f, u, node, m::DarcyModel, data)
    f[1] = storativity(m, node.region) * u[1]
    return nothing
end

"""Two-point Darcy flux ``(k/\\mu)(p_1 - p_2)``."""
function flux!(f, u, edge, m::DarcyModel, data)
    f[1] = mobility(m, edge.region) * (u[1, 1] - u[1, 2])
    return nothing
end

"""Imposed pressures, from the model's `dirichlet` field, resolved at the current time."""
function bcondition!(f, u, bnode, m::DarcyModel, data)
    apply_dirichlet!(f, u, bnode, m.dirichlet)
    return nothing
end
