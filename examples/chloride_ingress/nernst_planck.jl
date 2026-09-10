# nernst_planck.jl — multi-ionic transport with a zero-current closure
#
# The transport core of the rewrite of `run_4.jl` onto the balances of
# `CLAUDE/gia_formulation.md`. It is a *component*, like `dlm.jl`, not a case: it carries
# no boundary values, no chemistry and no material data of its own.
#
# ## Why this exists
#
# `run_4.jl` transports four ions with four different free-water diffusivities
# (2.03, 1.33, 1.96, 0.79 ×10⁻⁹ m²/s) through four **independent** Fick laws, with no
# migration term. Ions of different mobility cannot separate freely: the faster ones build
# a charge imbalance that pulls the slower ones along, which is what the diffusion
# potential is. Measured on `run_4`, the net current it carries reaches
#
#     |Σ zᵢ Jᵢ| / Σ |zᵢ Jᵢ| = 0.46
#
# at the front. The model then restores neutrality inside the chemistry pass, by letting
# OH⁻ absorb whatever charge the transport created — which amounts to postulating an
# instantaneous, unmodelled charge carrier.
#
# ChemistryLab makes the point precisely: charge is a **pseudo-element** `:Zz`, one row of
# the conservation matrix, and on the OPC system that row is independent — `A` is 9×34 of
# rank 9, and rank 8 without it. Charge is therefore a conserved component like any other,
# and it needs a flux law, not a repair.
#
# ## What it solves
#
#     ∂ₜ(φ cᵢ) + ∇·Jᵢ = 0,     Jᵢ = − Dᵢ τ(φ) (∇cᵢ + zᵢ cᵢ ∇Ψ)
#     Σᵢ zᵢ cᵢ = 0
#
# with `Ψ = F φ_e / RT` the dimensionless potential. The closure is the **algebraic**
# electroneutrality constraint, node-local, with no storage and no flux of its own — the
# use `reaction!` names in its own docstring.
#
# It is worth saying why the obvious alternative is wrong, because it looks right and
# produces a smooth answer. Imposing `∇·(Σᵢ zᵢ Jᵢ) = 0` as the potential's equation adds
# nothing: weight the ion balances by `zᵢ` and sum, and that is exactly what you get. The
# system is then singular in `Ψ`, which comes back at round-off while every other check —
# neutrality, conservation — still passes, because they were already implied. Zero current
# is a **consequence** of the constraint, not a substitute for it; `edge_current` below
# measures it as a diagnostic rather than imposing it.
#
# ## Scharfetter-Gummel, not centred differences
#
# The migration term is advective in disguise, so a centred flux oscillates once the
# potential varies over a cell. `fbernoulli_pm` gives the exponentially fitted flux that
# VoronoiFVM ships for drift-diffusion, and it degenerates to the plain difference when the
# potential is flat — which is the check `test/reactive/nernst_planck.jl` starts with.

using PoroMechanics
using VoronoiFVM: fbernoulli_pm, boundary_dirichlet!

"""
    NernstPlanck(; phi, D, z, tortuosity, dirichlet)

Multi-ionic transport with a zero-current closure.

| field | meaning |
|---|---|
| `phi` | porosity, scalar |
| `D` | free-water diffusivities, one per ion, in a tuple |
| `z` | charge numbers, one per ion, in a tuple |
| `tortuosity` | an `AbstractTortuosity`; `τ` here is `D_eff/D⁰`, **not** a geometric factor |
| `dirichlet` | one boundary tuple per unknown, ions first and the potential last |

The unknowns are the `n` ion concentrations followed by the potential `Ψ`, so
`nspecies == length(D) + 1`.

Every coefficient is a type parameter, so a `ForwardDiff.Dual` can enter a *parameter*.

!!! note "The tortuosity convention"
    `OhJang` returns `D_eff/D⁰`, porosity included — see
    `CLAUDE/gia_formulation.md` §4.2. The flux therefore carries **no** extra `φ`, while
    the storage does. Mixing that up with `FickModel`, whose `D` is a pore diffusivity,
    costs a factor `1/φ`.
"""
Base.@kwdef struct NernstPlanck{T, DD, ZZ, TT, B} <: PoroMechanics.AbstractPoroModel
    phi::T = 0.121
    D::DD = (2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9)
    z::ZZ = (-1, 1, 1, 2)
    tortuosity::TT = nothing
    dirichlet::B = ()
end

nions(m::NernstPlanck) = length(m.D)
PoroMechanics.nspecies(m::NernstPlanck) = nions(m) + 1
ipot(m::NernstPlanck) = nions(m) + 1

PoroMechanics.species_names(m::NernstPlanck) =
    vcat([Symbol("c_", i) for i in 1:nions(m)], [:Ψ])

"""
    effective_diffusivity(m, i) -> Float64

`Dᵢ τ(φ)`, the coefficient the flux actually carries.
"""
effective_diffusivity(m::NernstPlanck, i) =
    m.tortuosity === nothing ? m.D[i] : m.D[i] * tortuosity(m.tortuosity, m.phi, 1)

# ── The interface ─────────────────────────────────────────────────────────────

"""
    storage!(f, u, node, m::NernstPlanck, data)

`φ cᵢ` for each ion, and **zero** for the potential, which makes its row algebraic: the
constraint in `reaction!` holds at every instant rather than accumulating.
"""
function PoroMechanics.storage!(f, u, node, m::NernstPlanck, ::Any)
    for i in 1:nions(m)
        f[i] = m.phi * u[i]
    end
    f[ipot(m)] = zero(eltype(f))
    return nothing
end

"""
    reaction!(f, u, node, m::NernstPlanck, data)

The electroneutrality constraint `Σᵢ zᵢ cᵢ = 0`, which is what determines `Ψ`. Zero for the
ions: they have no volumetric source here.
"""
function PoroMechanics.reaction!(f, u, node, m::NernstPlanck, ::Any)
    for i in 1:nions(m)
        f[i] = zero(eltype(f))
    end
    f[ipot(m)] = sum(m.z[i] * u[i] for i in 1:nions(m))
    return nothing
end

"""
    flux!(f, u, edge, m::NernstPlanck, data)

Scharfetter-Gummel for each ion. The potential carries no flux of its own.

`fbernoulli_pm(x)` returns `(B(x), B(-x))` with `B(x) = x/(eˣ−1)`, and the fitted flux of
`−D(∇c + z c ∇Ψ)` is `D (B(−δ) c₁ − B(δ) c₂)` with `δ = z(Ψ₁ − Ψ₂)`. At `δ = 0` both
Bernoulli factors are one and this is the plain difference.
"""
function PoroMechanics.flux!(f, u, edge, m::NernstPlanck, ::Any)
    iψ = ipot(m)
    δψ = u[iψ, 1] - u[iψ, 2]
    for i in 1:nions(m)
        bp, bm = fbernoulli_pm(m.z[i] * δψ)          # (B(δ), B(−δ))
        f[i] = effective_diffusivity(m, i) * (bm * u[i, 1] - bp * u[i, 2])
    end
    ## No flux of its own: the potential is fixed by the constraint in `reaction!`.
    f[iψ] = zero(eltype(f))
    return nothing
end

## Head-and-tail recursion over the per-unknown boundary tuples, for the reason
## `apply_dirichlet!` gives: a heterogeneous tuple iterated in a plain loop boxes, and the
## allocations land inside the assembly loop, per facet and per Newton iteration.
_bc_each!(f, u, bnode, ::Tuple{}, i) = nothing
function _bc_each!(f, u, bnode, d::Tuple, i)
    PoroMechanics.apply_dirichlet!(f, u, bnode, first(d); species = i)
    return _bc_each!(f, u, bnode, Base.tail(d), i + 1)
end

"""
    bcondition!(f, u, bnode, m::NernstPlanck, data)

One boundary tuple per unknown. The potential needs at least one Dirichlet value: only its
gradient enters the fluxes, so without a reference the system is singular by a constant.
"""
function PoroMechanics.bcondition!(f, u, bnode, m::NernstPlanck, ::Any)
    return _bc_each!(f, u, bnode, m.dirichlet, 1)
end

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""
    net_charge(m, u) -> Vector

`Σᵢ zᵢ cᵢ` at every node. Zero to round-off when the initial and boundary data are
electroneutral — the property the zero-current closure is there to preserve, and the one
`run_4.jl` restores by hand at every chemistry pass instead.
"""
net_charge(m::NernstPlanck, u) =
    [sum(m.z[i] * u[i, k] for i in 1:nions(m)) for k in axes(u, 2)]

"""
    edge_current(m, u, dx) -> Vector

`Σᵢ zᵢ Jᵢ` on every edge, and the sum of `|zᵢ Jᵢ|` to scale it by. The ratio of the two is
the number that reads 0.46 on `run_4.jl` and must read zero here.
"""
function edge_current(m::NernstPlanck, u, dx)
    n = size(u, 2)
    cur = zeros(n - 1)
    scale = zeros(n - 1)
    for k in 1:(n - 1)
        δψ = u[ipot(m), k] - u[ipot(m), k + 1]
        for i in 1:nions(m)
            bp, bm = fbernoulli_pm(m.z[i] * δψ)
            J = effective_diffusivity(m, i) * (bm * u[i, k] - bp * u[i, k + 1]) / dx
            cur[k] += m.z[i] * J
            scale[k] += abs(m.z[i] * J)
        end
    end
    return cur, scale
end
