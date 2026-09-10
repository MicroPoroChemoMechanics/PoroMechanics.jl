# afm_exchange.jl — the monosulphate ⇌ Friedel's salt exchange, as a complementarity
#
#     Ca₄Al₂(SO₄)(OH)₁₂·6H₂O  +  2 Cl⁻   ⇌   Ca₄Al₂Cl₂(OH)₁₂·4H₂O  +  SO₄²⁻
#
# Both phases carry the same two aluminium, so the aluminium balance ties them:
# `n_MS + n_FS = n_AFm` is a constant, and one unknown describes the pair. Writing it
# normalised, `x̂ = n_FS / n_AFm ∈ [0, 1]`, is not cosmetic — see the scaling note below.
#
# ## Why a complementarity and not an equation
#
# The exchange runs only while both phases can exist. Outside that range one of them is
# exhausted and the affinity no longer has to vanish:
#
#     x̂ = 0        ⟹  𝒜 ≤ 0        no Friedel's salt, and none wants to form
#     0 < x̂ < 1    ⟹  𝒜 = 0        the two coexist, mass action holds
#     x̂ = 1        ⟹  𝒜 ≥ 0        the monosulphate is gone
#
# Imposing `𝒜 = 0` everywhere would drive `x̂` outside `[0, 1]`; imposing `x̂ ≥ 0` alone
# lets `n_MS = n_AFm − n_FS` go negative. It is a complementarity problem **on a box**, and
# both bounds have to be in it.
#
# ## The reformulation, and the one that looks right and is not
#
# With `F = −𝒜` this is the standard box MCP, whose "mid" reformulation is
#
#     Φ(x̂) = x̂ − clamp(x̂ − λ F, 0, 1)
#
# smoothed by replacing `max` and `min` with `(a+b±√((a−b)²+ε²))/2`.
#
# The nesting of two Fischer-Burmeister functions — `φ(x̂, −φ(1−x̂, F))`, which is what one
# writes first — is **exactly wrong here**: measured on the four corner cases it returns
# zero on the two infeasible ones and non-zero on the two feasible ones. The box needs the
# mid function, not a nested FB.
#
# ## Two scaling points that decide whether Newton converges
#
# `x̂` is dimensionless and in `[0, 1]`; `𝒜` is dimensionless and can reach ten. `λ` brings
# the two to the same size — any `λ > 0` leaves the solution set unchanged and only the
# conditioning moves.
#
# And in the coexistence regime `Φ = λF`, which does not depend on `x̂` at all: the row's
# own diagonal is zero, and `x̂` is determined by the chloride and sulfate balances instead.
# The `ε` smoothing is what puts a small diagonal back, so driving `ε` to zero too early
# makes the linear system harder, not easier. Continue on it, and stop where the residual
# is met rather than at a fixed schedule.

using PoroMechanics

## Smooth max and min. One square root each, and `ForwardDiff`-safe everywhere.
_smax(a, b, ε) = (a + b + sqrt((a - b)^2 + ε^2)) / 2
_smin(a, b, ε) = (a + b - sqrt((a - b)^2 + ε^2)) / 2

"""
    box_mcp(x, F, ε; λ = 1)

Smoothed residual of the box complementarity problem on `[0, 1]`: zero exactly when
`x ∈ [0,1]` and `F` obeys the three regimes above.
"""
box_mcp(x, F, ε; λ = 1) = x - _smin(_smax(x - λ * F, zero(x), ε), one(x), ε)

"""
    AFmExchange(; inner, ions, n_afm, logK, c_ref, eps_mcp, lambda)

The AFm exchange on top of another model, which owns everything else. The unknown vector
gains one entry, `x̂`, at the end.

| field | meaning |
|---|---|
| `inner` | the model being extended, typically a `SurfaceResolvedTransport` |
| `ions` | an `IonIndex`; `Cl` and `SO4` must both be set |
| `n_afm` | total AFm aluminium, as `n_MS + n_FS` [mol·m⁻³ of medium] |
| `logK` | base-10 equilibrium constant of the exchange, activities referred to `c_ref` |
| `c_ref` | reference concentration [mol·m⁻³], 1000 for mol/L |
| `eps_mcp` | smoothing of the complementarity |
| `lambda` | scaling between `x̂` and the affinity |

!!! note "Calcium is not here on purpose"
    Both phases carry four calcium, so `T_Ca` gains `4 n_afm` whatever `x̂` is — a constant,
    which cancels exactly in `(T(uⁿ⁺¹) − T(uⁿ))` and would contribute nothing while looking
    as though it did. The same argument is why the rest of the mineral assemblage is absent
    until it too becomes an unknown.
"""
Base.@kwdef struct AFmExchange{IN, T} <: PoroMechanics.AbstractPoroModel
    inner::IN
    ions::IonIndex
    n_afm::T = 100.0
    logK::T = 1.0
    c_ref::T = 1000.0
    eps_mcp::T = 1.0e-4
    lambda::T = 0.1
end

PoroMechanics.nspecies(m::AFmExchange) = PoroMechanics.nspecies(m.inner) + 1
ifs(m::AFmExchange) = PoroMechanics.nspecies(m)

PoroMechanics.species_names(m::AFmExchange) =
    vcat(PoroMechanics.species_names(m.inner), [:x_FS])

PoroMechanics.bcondition!(f, u, bnode, m::AFmExchange, data) =
    PoroMechanics.bcondition!(f, u, bnode, m.inner, data)

"""
    affinity(m::AFmExchange, u)

`𝒜 = ln K − ln Q` for `MS + 2Cl⁻ ⇌ FS + SO₄²⁻`, with the solids at unit activity and the
aqueous activities referred to `c_ref`. Dimensionless, and positive when Friedel's salt is
favoured.
"""
function affinity(m::AFmExchange, u)
    ix = m.ions
    tiny = eps(one(eltype(u)))
    c_Cl = max(u[ix.Cl], tiny) / m.c_ref
    c_SO4 = max(u[ix.SO4], tiny) / m.c_ref
    return m.logK * log(oftype(c_Cl, 10)) - (log(c_SO4) - 2 * log(c_Cl))
end

function PoroMechanics.storage!(f, u, node, m::AFmExchange, data)
    PoroMechanics.storage!(f, u, node, m.inner, data)
    ix = m.ions
    x = u[ifs(m)]
    f[ix.Cl] += 2 * m.n_afm * x                  # Friedel's salt carries two chloride
    f[ix.SO4] += m.n_afm * (one(x) - x)          # monosulphate carries one sulfate
    f[ifs(m)] = zero(eltype(f))                  # algebraic
    return nothing
end

function PoroMechanics.flux!(f, u, edge, m::AFmExchange, data)
    PoroMechanics.flux!(f, u, edge, m.inner, data)
    f[ifs(m)] = zero(eltype(f))                  # a mineral does not move
    return nothing
end

function PoroMechanics.reaction!(f, u, node, m::AFmExchange, data)
    PoroMechanics.reaction!(f, u, node, m.inner, data)
    f[ifs(m)] = box_mcp(
        u[ifs(m)], -affinity(m, u), m.eps_mcp; λ = m.lambda
    )
    return nothing
end
