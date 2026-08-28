"""
Tortuosity models — the factor by which a porous microstructure reduces the free-solution
diffusivity of a dissolved species, ``D_\\text{eff} = \\tau \\, D_\\text{free}``.
"""

"""
    AbstractTortuosity

Supertype of the tortuosity models. A concrete model implements
[`tortuosity`](@ref)`(model, φ, S_l)`.
"""
abstract type AbstractTortuosity end

"""
    tortuosity(model, φ, S_l = 1)

Tortuosity factor ``\\tau`` at porosity `φ` and liquid saturation `S_l`. For a saturated
medium the saturation argument may be omitted.
"""
function tortuosity end

# ── Oh & Jang ─────────────────────────────────────────────────────────────────

"""
    OhJang(; phi_c, n, ds, tau_agg, sat_exponent = 4.5, cap_fraction = 0.5)

Tortuosity of a cementitious material after Oh & Jang [ohjang2004](@cite), extended by a
saturation factor:

```math
\\tau = \\tau_\\text{paste}(\\varphi) \\; \\tau_\\text{agg} \\; S_l^{\\,q}
```

with the paste contribution built from the capillary porosity
``\\varphi_\\text{cap} = f \\varphi``:

```math
\\tau_\\text{paste} = \\left(m_p + \\sqrt{m_p^2 +
  \\frac{d_s^{1/n}\\,\\varphi_c}{1 - \\varphi_c}}\\right)^{n},
\\qquad
m_p = \\frac{(\\varphi_\\text{cap} - \\varphi_c)
  + d_s^{1/n}\\,(1 - \\varphi_c - \\varphi_\\text{cap})}{2\\,(1 - \\varphi_c)}
```

| Field | Meaning |
|---|---|
| `phi_c` | percolation threshold ``\\varphi_c`` [-] |
| `n` | Oh-Jang exponent [-] |
| `ds` | solid-to-pore diffusivity ratio ``d_s`` [-] |
| `tau_agg` | aggregate tortuosity factor [-], calibrated on the measured ``D_\\text{app}`` |
| `sat_exponent` | exponent ``q`` of the saturation factor, 4.5 by default |
| `cap_fraction` | fraction ``f`` of the total porosity that is capillary, 0.5 by default |

For a saturated medium ``S_l = 1`` and the saturation factor is one, which is why the
saturated and unsaturated chloride-ingress models share this single implementation.
"""
Base.@kwdef struct OhJang{T} <: AbstractTortuosity
    phi_c::T
    n::T
    ds::T
    tau_agg::T
    sat_exponent::T = 4.5
    cap_fraction::T = 0.5
end

function OhJang(phi_c, n, ds, tau_agg, sat_exponent = 4.5, cap_fraction = 0.5)
    return OhJang(promote(phi_c, n, ds, tau_agg, sat_exponent, cap_fraction)...)
end

Base.eltype(::OhJang{T}) where {T} = T

function tortuosity(t::OhJang, phi, sl = 1)
    T = promote_type(typeof(phi), typeof(sl), eltype(t))
    phi_cap = phi > 0 ? T(t.cap_fraction) * phi : zero(T)
    phi_c = T(t.phi_c)
    n = T(t.n)
    dsn = T(t.ds)^(one(T) / n)
    ## The 1/2 below is the factor in the Oh-Jang expression for m_p — unrelated to
    ## `cap_fraction`, which happens to share the value 0.5 by default.
    m_p = T(0.5) * ((phi_cap - phi_c) + dsn * (one(T) - phi_c - phi_cap)) /
        (one(T) - phi_c)
    tau_paste = (m_p + sqrt(m_p^2 + dsn * phi_c / (one(T) - phi_c)))^n
    return tau_paste * T(t.tau_agg) * sl^T(t.sat_exponent)
end
