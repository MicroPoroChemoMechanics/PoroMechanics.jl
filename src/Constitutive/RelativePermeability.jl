"""
Relative permeability curves.

Same rule as the retention curves: coefficients are type-parameterized so that a
`ForwardDiff.Dual` can enter them, making a result differentiable with respect to the
material parameters.
"""

"""
    AbstractRelativePermeability

Supertype of the relative permeability curves. A concrete curve implements
[`relative_permeability`](@ref).
"""
abstract type AbstractRelativePermeability end

"""
    relative_permeability(curve, pc)

Relative permeability to the liquid, ``k_{rl} \\in [0, 1]``, at capillary pressure `pc`
[Pa]. Returns 1 for `pc ≤ 0`.
"""
function relative_permeability end

# ── Mualem ────────────────────────────────────────────────────────────────────

"""
    Mualem(a, m)

Mualem relative permeability built on a Van Genuchten effective saturation
[mualem1976](@cite), [vangenuchten1980](@cite):

```math
k_{rl}(p_c) = \\sqrt{S_e}\\left(1 - \\left(1 - S_e^{1/m}\\right)^{m}\\right)^2,
\\qquad S_e = \\left(1 + (p_c/a)^n\\right)^{-m}, \\quad n = 1/(1-m)
```

`a` and `m` are the parameters of the ``k_{rl}`` curve, which need not equal those of the
retention curve.

The derivative ``\\mathrm{d}\\sqrt{S_e}/\\mathrm{d}S_e = 1/(2\\sqrt{S_e})`` diverges as
``S_e \\to 0``. In the dry zone ``k_{rl}`` is physically zero, so below `S_e = 1e-14` the
curve returns exactly zero, which keeps the gradient finite for ForwardDiff.
"""
struct Mualem{T} <: AbstractRelativePermeability
    a::T   # air-entry pressure of the k_rl curve [Pa]
    m::T   # exponent [-]
end

Mualem(a, m) = Mualem(promote(a, m)...)
Base.eltype(::Mualem{T}) where {T} = T

function relative_permeability(k::Mualem, pc)
    T = promote_type(typeof(pc), eltype(k))
    pc <= 0 && return one(T)
    n = one(T) / (one(T) - k.m)
    Se = clamp((one(T) + (pc / k.a)^n)^(-k.m), zero(T), one(T))
    Se < 1.0e-14 && return zero(T)          # dry zone: k_rl = 0, gradient = 0
    arg = clamp(one(T) - Se^(one(T) / k.m), zero(T), one(T))
    return sqrt(Se) * (one(T) - arg^k.m)^2
end

# ── Power law ─────────────────────────────────────────────────────────────────

"""
    PowerLawKrl(a, n, m)

Relative permeability of the same functional form as a Van Genuchten curve, but with
independent exponents:

```math
k_{rl}(p_c) = \\left(1 + (p_c / a)^n\\right)^{-m}
```

Used by the non-isothermal drying model, where the two materials are fitted with `n = 2`
and different `m`, so the Van Genuchten constraint ``n = 1/(1-m)`` does not hold.
"""
struct PowerLawKrl{T} <: AbstractRelativePermeability
    a::T
    n::T
    m::T
end

PowerLawKrl(a, n, m) = PowerLawKrl(promote(a, n, m)...)
Base.eltype(::PowerLawKrl{T}) where {T} = T

function relative_permeability(k::PowerLawKrl, pc)
    T = promote_type(typeof(pc), eltype(k))
    pc <= 0 && return one(T)
    return (one(T) + (pc / k.a)^k.n)^(-k.m)
end

# ── Gas relative permeability ─────────────────────────────────────────────────

"""
    gas_relative_permeability(sl)

Relative permeability to the gas phase as a function of liquid saturation:

```math
k_{rg}(S_l) = (1 - S_l)^2 \\left(1 - S_l^{5/3}\\right)
```

Zero at full saturation. `sl` is clamped just below 1 so the expression stays
differentiable at the endpoint.
"""
function gas_relative_permeability(sl)
    T = typeof(float(sl))
    sl >= 1 && return zero(T)
    s = clamp(sl, zero(T), one(T) - T(1.0e-12))
    return (one(T) - s)^2 * (one(T) - s^(T(5) / T(3)))
end

# ── Gardner ───────────────────────────────────────────────────────────────────

"""
    GardnerKrl(α)

Exponential relative permeability [gardner1958](@cite):

```math
k_{rl}(p_c) = \\exp(-\\alpha\\, p_c), \\qquad p_c > 0
```

`α` [Pa⁻¹] need not equal the `α` of the matching [`Gardner`](@ref) retention curve.

Substituting this law into the steady Richards equation turns it into a *linear* ordinary
differential equation for ``\\exp(\\alpha p_l)``, so the steady profile above a water table
has a closed form — see the Gardner infiltration benchmark.
"""
struct GardnerKrl{T} <: AbstractRelativePermeability
    α::T
end

Base.eltype(::GardnerKrl{T}) where {T} = T

function relative_permeability(k::GardnerKrl, pc)
    T = promote_type(typeof(pc), eltype(k))
    pc <= 0 && return one(T)
    return exp(-k.α * pc)
end
