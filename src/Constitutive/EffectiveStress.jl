"""
Effective stress in an unsaturated medium.

When both a liquid and a gas occupy the pores, "the" pore pressure is no longer defined:
the two phases push on the skeleton with different pressures, weighted by how much of the
pore space each occupies. Bishop's proposal was to replace them by a single equivalent
pressure,

```math
\\pi = p_g - \\chi\\,p_c , \\qquad p_c = p_g - p_l
```

so that the total stress keeps the saturated form ``\\sigma = \\sigma' - b\\,\\pi\\,I``. The
whole modelling question is what ``\\chi`` should be.

Taking ``\\chi = S_l`` makes ``\\pi`` the saturation-weighted average
``S_l p_l + (1-S_l) p_g``, which is Coussy's equivalent pore pressure with the interfacial
energy neglected. It is the usual choice and the default here. Fitted alternatives exist
because ``\\chi = S_l`` is known to be imperfect for fine-grained soils, so the coefficient
is a model of its own rather than a hard-wired formula.
"""

"""
    AbstractBishop

Supertype of the Bishop coefficient models. A concrete model implements
[`bishop_coefficient`](@ref)`(model, pc)`.
"""
abstract type AbstractBishop end

"""
    bishop_coefficient(model, pc) -> χ ∈ [0, 1]

Weight given to the liquid pressure in the equivalent pore pressure at capillary pressure
`pc` [Pa].
"""
function bishop_coefficient end

# ── χ = S_l ───────────────────────────────────────────────────────────────────

"""
    SaturationBishop(retention)

``\\chi = S_l(p_c)``, so the equivalent pore pressure is the saturation-weighted average of
the two phase pressures. Coussy's form with the interfacial energy dropped, and the usual
default.
"""
struct SaturationBishop{R <: AbstractRetention} <: AbstractBishop
    retention::R
end

bishop_coefficient(b::SaturationBishop, pc) = saturation(b.retention, pc)

# ── χ = S_l^n ─────────────────────────────────────────────────────────────────

"""
    PowerBishop(retention, n)

``\\chi = S_l(p_c)^n``, the fitted generalisation used when ``\\chi = S_l`` overestimates
the contribution of the liquid — as it does in fine-grained soils, where part of the water
is held in menisci that transmit little stress. `n = 1` recovers
[`SaturationBishop`](@ref).
"""
struct PowerBishop{R <: AbstractRetention, T} <: AbstractBishop
    retention::R
    n::T
end

bishop_coefficient(b::PowerBishop, pc) = saturation(b.retention, pc)^b.n

# ── Equivalent pore pressure and effective stress ─────────────────────────────

"""
    equivalent_pore_pressure(model, p_l, p_g = zero(p_l)) -> π

The single pressure that plays the role of ``p`` in the saturated theory,

```math
\\pi = p_g - \\chi(p_c)\\,p_c , \\qquad p_c = p_g - p_l
```

At full saturation (``p_c \\le 0``, so ``\\chi = 1``) this returns ``p_l`` exactly, which is
the check that the unsaturated theory contains the saturated one.
"""
function equivalent_pore_pressure(model::AbstractBishop, p_l, p_g = zero(p_l))
    pc = p_g - p_l
    return p_g - bishop_coefficient(model, pc) * pc
end

"""
    unsaturated_total_stress(b, χmodel, σ_eff, p_l, p_g = 0) -> σ

Total stress of an unsaturated poroelastic medium,
``\\sigma = \\sigma' - b\\,\\pi\\,I``, with ``\\pi`` the equivalent pore pressure.

Tension positive, as everywhere else here.
"""
function unsaturated_total_stress(
        b, χmodel::AbstractBishop, σ_eff::Tensors.SymmetricTensor{2}, p_l, p_g = zero(p_l)
    )
    return total_stress(b, σ_eff, equivalent_pore_pressure(χmodel, p_l, p_g))
end

"""
    suction_stress(χmodel, p_l, p_g = 0) -> χ p_c

The isotropic tension that suction exerts on the skeleton, ``\\chi p_c \\ge 0``: what pulls
the grains together and gives an unsaturated soil its apparent cohesion. It is the part of
the effective stress that vanishes on wetting, and therefore the mechanism behind collapse
on saturation.
"""
function suction_stress(χmodel::AbstractBishop, p_l, p_g = zero(p_l))
    pc = p_g - p_l
    return pc <= 0 ? zero(pc) : bishop_coefficient(χmodel, pc) * pc
end
