"""
Linear poroelasticity — Biot's material and the constants derived from it.

Like the retention and permeability curves, the material is parameterized by the *type* of
its coefficients rather than fixed to `Float64`. Every derived quantity below is therefore
differentiable with respect to the parameters it is built from: a sensitivity of a
consolidation time to the Biot coefficient, or a calibration of the storage modulus against
a measured settlement curve, needs exactly that.
"""

"""
    AbstractPoroelastic

Supertype of the poroelastic materials.
"""
abstract type AbstractPoroelastic <: AbstractPoroModel end

"""
    BiotPoroelastic(; E, nu, k, mu_l, b, N)

Linear isotropic Biot poroelasticity.

| Field | Meaning |
|---|---|
| `E`, `nu` | drained Young's modulus [Pa] and Poisson's ratio [-] |
| `k` | intrinsic permeability [m²] |
| `mu_l` | dynamic viscosity of the fluid [Pa·s] |
| `b` | Biot coefficient [-] |
| `N` | storage modulus at constant strain [Pa⁻¹], the inverse of the Biot modulus `M` |

Permeability sits in the material rather than beside it because Biot's theory couples
Darcy flow to the skeleton: [`consolidation_coefficient`](@ref) is a property of the pair,
not of either half.
"""
Base.@kwdef struct BiotPoroelastic{T} <: AbstractPoroelastic
    E::T = 1.0e8
    nu::T = 0.2
    k::T = 1.0e-13
    mu_l::T = 1.0e-3
    b::T = 1.0
    N::T = 7.2e-9
end

function BiotPoroelastic(E, nu, k, mu_l, b, N)
    return BiotPoroelastic(promote(E, nu, k, mu_l, b, N)...)
end

Base.eltype(::BiotPoroelastic{T}) where {T} = T

PoroMechanics.nspecies(::BiotPoroelastic) = 3
PoroMechanics.species_names(::BiotPoroelastic) = [:u1, :u2, :p]

# ── Elastic constants ─────────────────────────────────────────────────────────

"""
    lame(m) -> (λ, μ)

Lamé coefficients [Pa], drained.
"""
lame(m::BiotPoroelastic) = (
    m.E * m.nu / ((1 + m.nu) * (1 - 2m.nu)),
    m.E / (2 * (1 + m.nu)),
)

"""
    shear_modulus(m) -> G [Pa]
"""
shear_modulus(m::BiotPoroelastic) = m.E / (2 * (1 + m.nu))

"""
    bulk_modulus(m) -> K = λ + 2μ/3 [Pa]

Drained bulk modulus.
"""
function bulk_modulus(m::BiotPoroelastic)
    λ, μ = lame(m)
    return λ + 2μ / 3
end

"""
    oedometric_modulus(m) -> M_o = λ + 2μ = K + 4μ/3 [Pa]

The modulus that governs uniaxial strain, and the one that appears throughout the
consolidation constants.
"""
function oedometric_modulus(m::BiotPoroelastic)
    λ, μ = lame(m)
    return λ + 2μ
end

"""
    biot_modulus(m) -> M = 1/N [Pa]
"""
biot_modulus(m::BiotPoroelastic) = one(eltype(m)) / m.N

# ── Coupling constants ────────────────────────────────────────────────────────

"""
    compaction_coefficient(m) -> c_m [Pa⁻¹]

Geertsma's uniaxial compaction coefficient,

```math
c_m = \\frac{b\\,(1-2\\nu)}{2G(1-\\nu)} = \\frac{b}{M_o}
```

the constant of proportionality in ``\\varepsilon = c_m p + f(t)`` for problems whose
displacement field is irrotational.
"""
compaction_coefficient(m::BiotPoroelastic) = m.b / oedometric_modulus(m)

"""
    storage_coefficient(m) -> S [Pa⁻¹]

Storage at constant stress, ``S = N + b^2/M_o = 1/M + b\\,c_m``.
"""
storage_coefficient(m::BiotPoroelastic) = m.N + m.b * compaction_coefficient(m)

"""
    consolidation_coefficient(m) -> c [m²/s]

Uniaxial diffusivity ``c = (k/\\mu_l)/S``, obtained by eliminating the strain between the
constitutive law and the storage equation under uniaxial conditions.
"""
consolidation_coefficient(m::BiotPoroelastic) =
    (m.k / m.mu_l) / storage_coefficient(m)

"""
    hydraulic_conductivity(m) -> κ = k/μ_l [m²/(Pa·s)]
"""
hydraulic_conductivity(m::BiotPoroelastic) = m.k / m.mu_l

"""
    skempton(m) -> B [-]

Skempton's coefficient ``B = b/(NK + b^2) = bM/(K + b^2 M)``: the fraction of an
isotropic stress increment carried by the pore fluid under undrained conditions.
"""
skempton(m::BiotPoroelastic) = m.b / (m.N * bulk_modulus(m) + m.b^2)

"""
    undrained_poisson(m) -> ν_u [-]

```math
\\nu_u = \\frac{3\\nu + bB(1-2\\nu)}{3 - bB(1-2\\nu)}
```

which tends to 1/2 as the fluid and grains become incompressible (``B \\to 1``).
"""
function undrained_poisson(m::BiotPoroelastic)
    q = m.b * skempton(m) * (1 - 2m.nu)
    return (3m.nu + q) / (3 - q)
end

"""
    undrained_bulk_modulus(m) -> K_u = K + b²M [Pa]
"""
undrained_bulk_modulus(m::BiotPoroelastic) =
    bulk_modulus(m) + m.b^2 * biot_modulus(m)
