"""
Fickian diffusion of a solute through a saturated porous medium.

With ``c`` the concentration in the pore solution [mol/m³], ``\\varphi`` the porosity and
``D`` the effective diffusion coefficient,

```math
\\varphi \\frac{\\partial c}{\\partial t} = \\nabla\\cdot\\big(D\\,\\varphi\\,\\nabla c\\big)
```

The porosity cancels on a homogeneous medium, leaving pure diffusion; it does not cancel
across a material interface, which is why it is carried explicitly on both terms rather
than simplified away.
"""

"""
    FickModel(; phi, D, dirichlet)

Diffusion of one solute in a saturated porous medium. The unknown is the concentration
``c`` [mol/m³].

| field | meaning |
|---|---|
| `phi` | porosity [-] — a scalar, or one value per cell region |
| `D` | effective diffusion coefficient [m²/s] — a scalar, or one value per cell region |
| `dirichlet` | imposed concentrations, as `((region, value), …)` |

Boundaries not named are sealed: zero flux is `VoronoiFVM`'s default and the usual meaning
of an unlisted boundary in a transport problem.

`dirichlet` carries the boundary data rather than a method, because that is what
distinguishes one *case* from another and not one *model* from another — see
[`RichardsModel`](@ref) for the same argument at greater length. A value may be a number or
a function of time, resolved by [`PoroMechanics.dirichlet_value`](@ref).

A layered barrier is likewise a case, not a second model: pass a collection for `phi` and
`D`, indexed by the cell region.

```julia
FickModel(; phi = [0.30, 0.12], D = [1.0e-10, 2.0e-12], dirichlet = ((1, 1.0),))
```

The coefficients are type parameters, so a profile can be differentiated with respect to
`D` and not only with respect to the unknowns — which is what an inverse identification of
a diffusion coefficient from a measured profile needs.
"""
Base.@kwdef struct FickModel{T, P, B} <: AbstractPoroModel
    phi::T = 0.3
    D::P = 1.0e-10
    dirichlet::B = ()
end

## Promote rather than require a single type: differentiating with respect to one parameter
## makes that field a `Dual` while the rest stay `Float64`. Only the scalar case promotes —
## a per-region collection is left alone, as in `RichardsModel`. The promoted values are
## handed to the parametric constructor rather than back to this method, which would
## otherwise call itself: `promote` on two `Float64` returns two `Float64`.
function FickModel(phi::Number, D::Number, dirichlet)
    p, d = promote(phi, D)
    return FickModel{typeof(p), typeof(d), typeof(dirichlet)}(p, d, dirichlet)
end

"""
    porosity(m::FickModel, region) -> φ

Porosity of cell region `region`. Dispatch rather than a run-time branch, so a
single-region model pays nothing for the generality.
"""
porosity(m::FickModel{<:Number}, region) = m.phi
porosity(m::FickModel, region) = m.phi[region]

"""
    diffusivity(m::FickModel, region) -> D

Effective diffusion coefficient of cell region `region`, in m²/s.
"""
diffusivity(m::FickModel{<:Any, <:Number}, region) = m.D
diffusivity(m::FickModel, region) = m.D[region]

nspecies(::FickModel) = 1
species_names(::FickModel) = [:c]

"""Storage term ``\\varphi c``."""
function storage!(f, u, node, m::FickModel, data)
    f[1] = porosity(m, node.region) * u[1]
    return nothing
end

"""
Two-point flux ``D \\varphi (c_1 - c_2)``.

`VoronoiFVM` divides `f` by the edge length, so the flux is written as a difference of node
values rather than as a gradient.
"""
function flux!(f, u, edge, m::FickModel, data)
    f[1] = diffusivity(m, edge.region) * porosity(m, edge.region) * (u[1, 1] - u[1, 2])
    return nothing
end

"""Imposed concentrations, from the model's `dirichlet` field."""
function bcondition!(f, u, bnode, m::FickModel, data)
    apply_dirichlet!(f, u, bnode, m.dirichlet)
    return nothing
end
