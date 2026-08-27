"""
The Barcelona Basic Model [alonso1990](@cite) — elastoplasticity for unsaturated soils.

What distinguishes it from a saturated critical-state model is that **suction is a second
loading variable**. Drying a soil enlarges its yield surface; wetting shrinks it. A sample
sitting safely inside the elastic domain can therefore be brought to yield without touching
the stress, simply by wetting it — and it then compresses. That is *collapse on wetting*,
the phenomenon the model was built to explain, and the test suite checks for it explicitly
because a model that misses it has missed the point.

## Sign conventions

The interface of this package is tension positive; soil mechanics is compression positive.
Rather than pick one and leave traps everywhere, the boundary is explicit:
[`material_response`](@ref) takes and returns tension-positive tensors, and converts once,
on entry and exit. Everything named `p`, `εv`, `p0`, `pc_star` inside is **compression
positive**.

## Flow rule

Associated flow is used, ``g = f``. Alonso *et al.* use a non-associated potential in the
deviatoric direction, scaled so that Jaky's ``K_0`` is recovered on oedometric paths; that
refinement changes the ratio of deviatoric to volumetric plastic strain but neither the
yield surface nor the hardening. It is a deliberate, documented simplification, not an
oversight, and swapping it means providing one more derivative.
"""

# ── Material ──────────────────────────────────────────────────────────────────

"""
    BBM(; κ, λ0, r, β, M, k_s, nu, e0, p_ref, p_min)

| Field | Meaning |
|---|---|
| `κ` | elastic swelling index [-] |
| `λ0` | virgin compression index at zero suction [-] |
| `r` | ratio ``\\lambda(\\infty)/\\lambda(0)`` [-] |
| `β` | rate at which the LC curve saturates [Pa⁻¹] |
| `M` | slope of the critical state line [-] |
| `k_s` | cohesion–suction coefficient [-] |
| `nu` | Poisson's ratio [-] |
| `e0` | initial void ratio [-] |
| `p_ref` | reference pressure of the LC law [Pa] |
| `p_min` | floor on the mean compressive stress [Pa] |

Defaults are the parameter set of the reference test case.
"""
Base.@kwdef struct BBM{T} <: AbstractMaterial
    κ::T = 0.011
    λ0::T = 0.065
    r::T = 0.75
    β::T = 2.0e-5
    M::T = 1.2
    k_s::T = 0.8
    nu::T = 0.15
    e0::T = 0.9
    p_ref::T = 1.0e4
    p_min::T = 1.0e2
end

## Promote rather than require a single type: differentiating with respect to one
## parameter makes that field a `Dual` while the others stay `Float64`.
function BBM(κ, λ0, r, β, M, k_s, nu, e0, p_ref, p_min)
    return BBM(promote(κ, λ0, r, β, M, k_s, nu, e0, p_ref, p_min)...)
end

Base.eltype(::BBM{T}) where {T} = T

"""
    BBMState(σ, ε, pc_star, εv_p, suction)

Internal variables: the stress and strain carried from the previous step (the elastic law
is pressure dependent), the preconsolidation pressure at zero suction `pc_star` — the
hardening variable — the accumulated volumetric plastic strain, and the suction the point
currently sits at.
"""
struct BBMState{S, E, T} <: AbstractMaterialState
    σ::S
    ε::E
    pc_star::T
    εv_p::T
    suction::T
end

## The three scalars are promoted rather than required to match: differentiating the
## response with respect to the strain turns `pc_star` and `εv_p` into `Dual`s while the
## suction, an input, stays a `Float64`.
function BBMState(σ, ε, pc_star, εv_p, suction)
    pc, evp, suc = promote(pc_star, εv_p, suction)
    return BBMState{typeof(σ), typeof(ε), typeof(pc)}(σ, ε, pc, evp, suc)
end

"""
    initial_state(m::BBM, σ0, pc_star; suction = 0)

Start from an in-situ stress and a preconsolidation pressure. Both are required: the elastic
stiffness is undefined without a stress, and the yield surface without `pc_star`.
"""
function initial_state(
        m::BBM, σ0::Tensors.SymmetricTensor{2}, pc_star; suction = zero(pc_star)
    )
    return BBMState(σ0, zero(σ0), pc_star, zero(pc_star), suction)
end

initial_state(::BBM) = error(
    "BBM has no default state: it needs an in-situ stress and a preconsolidation " *
        "pressure. Use `initial_state(m, σ0, pc_star; suction)`."
)

# ── Invariants, compression positive ──────────────────────────────────────────

"""Mean compressive net stress ``p = -\\mathrm{tr}(\\sigma)/3`` [Pa]."""
mean_pressure(σ::Tensors.SymmetricTensor{2}) = -Tensors.tr(σ) / 3

"""Deviatoric stress tensor."""
deviator(σ::Tensors.SymmetricTensor{2}) = Tensors.dev(σ)

"""Von Mises equivalent stress ``q = \\sqrt{\\tfrac{3}{2}\\,s:s}`` [Pa]."""
function equivalent_stress(σ::Tensors.SymmetricTensor{2})
    s = Tensors.dev(σ)
    return sqrt(3 * (s ⊡ s) / 2)
end

# ── The LC curve ──────────────────────────────────────────────────────────────

"""
    compression_index(m, s) -> λ(s)

``\\lambda(s) = \\lambda(0)\\left[(1-r)e^{-\\beta s} + r\\right]``: the soil compresses less
readily as it dries, tending to ``r\\lambda(0)`` at high suction.
"""
compression_index(m::BBM, s) = m.λ0 * ((1 - m.r) * exp(-m.β * s) + m.r)

"""
    preconsolidation(m, s, pc_star) -> p₀(s)

The **Loading–Collapse** curve, which is the model's central idea:

```math
\\frac{p_0(s)}{p_\\text{ref}} =
  \\left(\\frac{p_c^*}{p_\\text{ref}}\\right)^{\\frac{\\lambda(0)-\\kappa}{\\lambda(s)-\\kappa}}
```

``p_0`` grows with suction, so drying expands the elastic domain and wetting contracts it.
A stress state left untouched while the soil is wetted can therefore end up outside the
yield surface — which is collapse.
"""
function preconsolidation(m::BBM, s, pc_star)
    λs = compression_index(m, s)
    return m.p_ref * (pc_star / m.p_ref)^((m.λ0 - m.κ) / (λs - m.κ))
end

"""
    yield_function(m, p, q, s, pc_star) -> f

```math
f = q^2 - M^2\\,(p + k_s s)\\,(p_0(s) - p)
```

An ellipse in the ``(p,q)`` plane running from a tensile intercept ``-k_s s`` — the
apparent cohesion suction confers — to the preconsolidation pressure ``p_0(s)``.
Negative inside, zero on the surface.
"""
function yield_function(m::BBM, p, q, s, pc_star)
    p0 = preconsolidation(m, s, pc_star)
    return q^2 - m.M^2 * (p + m.k_s * s) * (p0 - p)
end

"""``\\partial f/\\partial p = M^2(2p + k_s s - p_0)``."""
function dyield_dp(m::BBM, p, s, pc_star)
    p0 = preconsolidation(m, s, pc_star)
    return m.M^2 * (2p + m.k_s * s - p0)
end

"""``\\partial f/\\partial q = 2q``."""
dyield_dq(::BBM, q) = 2q

# ── Elastic moduli ────────────────────────────────────────────────────────────

"""
    bbm_moduli(m, p) -> (K, G)

``K = p(1+e_0)/\\kappa`` and ``G = 3K(1-2\\nu)/(2(1+\\nu))``, with `p` floored at `m.p_min`
because a vanishing modulus would break the solve and a soil in real tension is outside
this law.
"""
function bbm_moduli(m::BBM, p)
    K = max(p, m.p_min) * (1 + m.e0) / m.κ
    G = 3K * (1 - 2m.nu) / (2 * (1 + m.nu))
    return K, G
end

"""
    hardening_modulus(m) -> (1+e₀)/(λ(0) − κ)

The coefficient of the hardening law ``\\dot p_c^* = H\\,p_c^*\\,\\dot\\varepsilon_v^p``.
"""
hardening_modulus(m::BBM) = (1 + m.e0) / (m.λ0 - m.κ)

# ── Return mapping ────────────────────────────────────────────────────────────

"""
    return_residual(m, x, p_tr, q_tr, s, pc_star_n, K, G) -> (r1, r2)

Residual of the return map, in the two unknowns `x = (p, Δγ)`.

Given ``\\Delta\\gamma`` and ``p``, the deviatoric return and the hardening follow in closed
form, so the system reduces to two equations: the volumetric return, and consistency.

The hardening is integrated exponentially,
``p_c^* = p_c^{*n}\\exp(H\\,\\Delta\\varepsilon_v^p)``, which keeps it positive whatever the
step size — a plain forward increment can drive it negative and destroy the yield surface.
"""
function return_residual(m::BBM, x, p_tr, q_tr, s, pc_star_n, K, G)
    p, Δγ = x[1], x[2]
    q = q_tr / (1 + 6G * Δγ)

    ## Volumetric plastic strain, compression positive, from associated flow.
    ## ∂f/∂p is evaluated at the *updated* preconsolidation, so the two are solved together.
    H = hardening_modulus(m)
    Δεv_p = Δγ * dyield_dp(m, p, s, pc_star_n)      # predictor for the hardening
    pc_star = pc_star_n * exp(H * Δεv_p)
    Δεv_p = Δγ * dyield_dp(m, p, s, pc_star)        # corrected with the updated surface

    r1 = p - p_tr + K * Δεv_p
    r2 = yield_function(m, p, q, s, pc_star) / (m.M^2 * m.p_ref^2)   # scaled to O(1)
    return (r1 / m.p_ref, r2)
end

"""
    solve_return_map(m, p_tr, q_tr, s, pc_star_n, K, G; tol, maxiter)

Newton on the two-equation residual, with the Jacobian from `ForwardDiff`. Returns
`(p, Δγ, q, pc_star, converged)`.
"""
function solve_return_map(
        m::BBM, p_tr, q_tr, s, pc_star_n, K, G; tol = 1.0e-10, maxiter = 50
    )
    x = [p_tr, zero(p_tr)]
    R(y) = collect(return_residual(m, y, p_tr, q_tr, s, pc_star_n, K, G))
    converged = false
    for _ in 1:maxiter
        r = R(x)
        if maximum(abs, r) < tol
            converged = true
            break
        end
        J = ForwardDiff.jacobian(R, x)
        x -= J \ r
        x[2] = max(x[2], zero(x[2]))     # the multiplier cannot go negative
    end
    p, Δγ = x[1], x[2]
    q = q_tr / (1 + 6G * Δγ)
    H = hardening_modulus(m)
    Δεv_p = Δγ * dyield_dp(m, p, s, pc_star_n)
    pc_star = pc_star_n * exp(H * Δεv_p)
    Δεv_p = Δγ * dyield_dp(m, p, s, pc_star)
    pc_star = pc_star_n * exp(H * Δεv_p)
    return p, Δγ, q, pc_star, Δεv_p, converged
end

# ── The response ──────────────────────────────────────────────────────────────

"""
    material_response(m::BBM, ε, state, Δt) -> (σ, ∂σ∂ε, state_new)

Tension-positive strain in, tension-positive stress out. The suction is read from the
state, so a wetting or drying step is applied by handing in a state whose `suction` has
changed — which is exactly how a collapse test is driven.

The tangent returned on a plastic step is the **continuum** elastoplastic tangent — see
[`elastoplastic_tangent`](@ref) for what that does and does not buy.
"""
function material_response(
        m::BBM, ε::Tensors.SymmetricTensor{2, dim}, state::BBMState, Δt
    ) where {dim}
    s = state.suction
    Δε = ε - state.ε

    ## Elastic predictor, with the moduli of the incoming state.
    p_n = mean_pressure(state.σ)
    K, G = bbm_moduli(m, p_n)
    λ_lame = K - 2G / 3
    C_e = elastic_stiffness(LinearElastic(λ_lame, G), Val(dim))
    σ_tr = state.σ + C_e ⊡ Δε

    p_tr = mean_pressure(σ_tr)
    q_tr = equivalent_stress(σ_tr)

    if yield_function(m, p_tr, q_tr, s, state.pc_star) <= 0
        new = BBMState(σ_tr, ε, state.pc_star, state.εv_p, s)
        return σ_tr, C_e, new
    end

    p, Δγ, q, pc_star, Δεv_p, ok = solve_return_map(m, p_tr, q_tr, s, state.pc_star, K, G)
    ok || @warn "BBM return map did not converge" p_tr q_tr s pc_star_n = state.pc_star

    ## Rebuild the stress: the deviator keeps its direction and is scaled by q/q_tr.
    dev_tr = deviator(σ_tr)
    scale = q_tr > 0 ? q / q_tr : zero(q)
    σ = scale * dev_tr - p * one(σ_tr)

    C = elastoplastic_tangent(m, C_e, p, q, s, pc_star, Δγ, K, G, dev_tr, q_tr, Val(dim))
    new = BBMState(σ, ε, pc_star, state.εv_p + Δεv_p, s)
    return σ, C, new
end

"""
    elastoplastic_tangent(m, C_e, p, q, s, pc_star, Δγ, K, G, dev_tr, q_tr, ::Val{dim})

The **continuum** elastoplastic tangent, from the consistency condition differentiated at
the converged point:

```math
C = C^e - \\frac{(C^e : n)\\otimes(n : C^e)}{n : C^e : n + H_p},
\\qquad H_p = -\\frac{\\partial f}{\\partial p_c^*}\\,H\\,p_c^*\\,\\frac{\\partial f}{\\partial p}
```

with ``n = \\partial f/\\partial\\sigma``. It is obtained from ``\\dot f = 0``, not by
differentiating through the Newton loop of the return map — which is the right instinct, and
also the only practical one, since nested automatic differentiation through that loop
returns `NaN`.

**What this is not.** The *algorithmic* (consistent) tangent, which is the exact Jacobian of
the discrete return map, differs from this one by terms of order ``\\Delta\\gamma``. Measured
on a plastic isotropic step, the gap against a finite-difference Jacobian falls from 65 % at
``\\Delta\\varepsilon_v = 4\\times10^{-3}`` to 15 % at ``2.5\\times10^{-4}`` — correct in the
limit, and enough to drive a global Newton, but at a linear rather than quadratic rate. The
algorithmic tangent is not implemented yet; it is the next piece of work on this model, and
pretending otherwise would show up as a slow global solve rather than a wrong answer.
"""
function elastoplastic_tangent(
        m::BBM, C_e, p, q, s, pc_star, Δγ, K, G, dev_tr, q_tr, ::Val{dim}
    ) where {dim}
    ## Flow direction in tensor form. p is compression positive, so ∂p/∂σ = −I/3.
    ∂f∂p = dyield_dp(m, p, s, pc_star)
    ∂f∂q = dyield_dq(m, q)

    Iden = one(Tensors.SymmetricTensor{2, dim})
    n_dev = q_tr > 0 ? (3 * dev_tr / (2 * q_tr)) : zero(dev_tr)
    n = -∂f∂p * Iden / 3 + ∂f∂q * n_dev

    ## Hardening: ṗc* = H pc* ε̇v^p and ∂f/∂pc* through the LC curve.
    H = hardening_modulus(m)
    ∂f∂pc = ForwardDiff.derivative(z -> yield_function(m, p, q, s, z), pc_star)
    Hp = -∂f∂pc * H * pc_star * ∂f∂p

    Cn = C_e ⊡ n
    denom = (n ⊡ Cn) + Hp
    return abs(denom) < eps(typeof(denom)) ? C_e : C_e - (Cn ⊗ Cn) / denom
end
