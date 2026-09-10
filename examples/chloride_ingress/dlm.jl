# dlm.jl — surface complexation on C-S-H, one implementation
#
# NOTE — this belongs in ChemistryLab.jl. Surface complexation is chemistry, not
# transport. It lives here only because ChemistryLab.jl does not expose an electrical
# double layer yet — its `AbstractSurfaceModel` is a *kinetic* surface-area model, not an
# EDL. The signatures below are chosen so that the move upstream is a delete plus an
# import.
#
# This file replaces three near-identical copies that used to live in `run_4.jl`
# (`solve_dlm`), `tran2018.jl` (`solve_dlm_marks`) and `chloride_ternary.jl`
# (`solve_dlm_ternary`). They differed in exactly four ways:
#
#   1. magnesium present or absent,
#   2. the chloride binding mechanism — outer-sphere ≡SiOHCl⁻ against the neutral
#      ternary ≡SiOCaCl,
#   3. the site density Γ_max, a scalar against an interpolation in x_CaS,
#   4. the `n_csh ≤ 0` guard, present in two of the three.
#
# Points 1 and 3 are special cases, not variants: magnesium at zero concentration
# contributes exactly nothing, and an interpolation between two equal endpoints is the
# scalar. Only point 2 is a genuine branch, and it is the one carried by dispatch.
#
# ## The model
#
#   Surface reactions on C-S-H (≡SiOH = neutral silanol site):
#     ≡SiOH  ⇌  ≡SiO⁻ + H⁺                Ka1
#     ≡SiO⁻ + Ca²⁺  ⇌  ≡SiOCa⁺            K_Ca    (inner sphere, +1)
#     ≡SiO⁻ + Mg²⁺  ⇌  ≡SiOMg⁺            K_Mg    (inner sphere, +1, seawater)
#     ≡SiO⁻ + Na⁺   ⇌  ≡SiONa             K_Na    (neutral, no Boltzmann factor)
#     ≡SiO⁻ + K⁺    ⇌  ≡SiOK              K_K = K_Na
#   and, depending on the mechanism, one of
#     ≡SiOH + Cl⁻          ⇌  ≡SiOHCl⁻    K_Cl    (outer sphere, −1)
#     ≡SiOH + Ca²⁺ + Cl⁻   ⇌  ≡SiOCaCl + H⁺       (ternary, neutral)
#
#   Diffuse layer, Gouy-Chapman:
#     σ₀ = √(8 ε₀ ε_r R T I) · sinh(β/2),   β = F ψ / (R T)
#
# ## Why it is split into a residual and a solver
#
# [`dlm_residual`](@ref) is the charge balance with **no root finding inside**. A globally
# implicit scheme makes β a nodal unknown and needs exactly that residual; the sequential
# scheme needs the root, and [`solve_dlm`](@ref) provides it on top. Writing the solver
# first and the residual never — which is what the three copies did — is what made β
# impossible to hand to a global Newton.

using ForwardDiff

# ── Chloride binding mechanism ────────────────────────────────────────────────

"""
Supertype of the chloride binding mechanisms on C-S-H. The mechanism is a *type*, not a
flag, so the site sum and the charge sum select their chloride term by dispatch and no
run-time branch survives into the differentiated code.
"""
abstract type ChlorideBinding end

"""
    OuterSphere()

Outer-sphere chloride, `≡SiOH + Cl⁻ ⇌ ≡SiOHCl⁻` (Tran & Soive 2018). The complex carries
−1, so it appears in the site sum **and** lowers the surface charge.
"""
struct OuterSphere <: ChlorideBinding end

"""
    TernaryNeutral()

Ternary chloride, `≡SiOH + Ca²⁺ + Cl⁻ ⇌ ≡SiOCaCl + H⁺` (Thermoddem 2023). The complex is
neutral: it occupies a site but contributes nothing to the surface charge, and it takes no
Boltzmann factor. That absence from the charge sum is the whole difference, and it is why
the mechanism cannot be reduced to a different value of `K_Cl`.
"""
struct TernaryNeutral <: ChlorideBinding end

# ── Parameters ────────────────────────────────────────────────────────────────

"""
    DLM(; Ka1, K_Ca, K_Mg, K_Cl, K_Na_Tob, K_Na_Jen,
          Gamma_max_Tob, Gamma_max_Jen, a_s, eps_r, Kw, x_cas, n_csh0, mechanism)

Surface complexation constants for C-S-H.

**Every coefficient is a type parameter**, which is what lets a `ForwardDiff.Dual` enter a
*parameter* — so a chloride profile can be differentiated with respect to `K_Cl` or
`Gamma_max`, and not only with respect to the concentrations. The three implementations
this replaces declared every field `::Float64`, which made that impossible.

| field | meaning | unit |
|---|---|---|
| `Ka1` | deprotonation ≡SiOH → ≡SiO⁻ + H⁺ | mol·m⁻³ |
| `K_Ca`, `K_Mg` | inner-sphere cation constants | m³·mol⁻¹ |
| `K_Cl` | chloride constant; its meaning follows `mechanism` | m³·mol⁻¹ or m⁶·mol⁻² |
| `K_Na_Tob`, `K_Na_Jen` | alkali constant at x_CaS = 0.83 and 1.67 | m³·mol⁻¹ |
| `Gamma_max_Tob`, `Gamma_max_Jen` | site density at the same two end members | mol·m⁻² |
| `a_s` | BET specific surface area | m²·mol⁻¹ |
| `eps_r` | relative permittivity of water | – |
| `Kw` | ionic product of water | mol²·m⁻⁶ |
| `x_cas` | default Ca/Si ratio, when the caller has no per-node value | – |
| `n_csh0` | default C-S-H content | mol·m⁻³ of medium |

`K_Na` and `Γ_max` are interpolated linearly in `x_cas` between the tobermorite and
jennite end members. Setting the two endpoints equal recovers a constant, which is how the
binary parameter sets are expressed — no separate code path.

See [`DLM_TRAN2018`](@ref) and [`DLM_TERNARY`](@ref) for the two published sets.
"""
struct DLM{T, M <: ChlorideBinding}
    Ka1::T
    K_Ca::T
    K_Mg::T
    K_Cl::T
    K_Na_Tob::T
    K_Na_Jen::T
    Gamma_max_Tob::T
    Gamma_max_Jen::T
    a_s::T
    eps_r::T
    Kw::T
    x_cas::T
    n_csh0::T
    mechanism::M
end

## The keyword constructor promotes, then calls the parametric constructor **explicitly**.
## Calling `DLM(...)` from here would be more specific than the automatic one whenever the
## arguments are already of a common type, and would recurse until the stack overflows.
function DLM(;
        Ka1 = 2.0e-10,
        K_Ca = 2.0,
        K_Mg = 0.10,
        K_Cl = 4.47e-4,
        K_Na_Tob = 1.106e-3,
        K_Na_Jen = 9.0e-5,
        Gamma_max_Tob = 1.3e-6,
        Gamma_max_Jen = 1.3e-6,
        a_s = 85_000.0,
        eps_r = 78.5,
        Kw = 6.76e-9,
        x_cas = 1.5,
        n_csh0 = 2000.0,
        mechanism::ChlorideBinding = OuterSphere(),
    )
    p = promote(
        Ka1, K_Ca, K_Mg, K_Cl, K_Na_Tob, K_Na_Jen,
        Gamma_max_Tob, Gamma_max_Jen, a_s, eps_r, Kw, x_cas, n_csh0
    )
    return DLM{eltype(p), typeof(mechanism)}(p..., mechanism)
end

Base.eltype(::DLM{T}) where {T} = T

"""
    DLM_TRAN2018(; kwargs...)

Binary outer-sphere set of Tran, Soive, Bonnet & Khelidj (2018), *Cem. Concr. Res.* **110**,
70–85, extended to Mg²⁺ for seawater. Site density 1.3×10⁻⁶ mol·m⁻² at both end members,
i.e. independent of x_CaS.

  Ka1 = 2.0e-13 mol/L → 2.0e-10 mol/m³ ;  K_Ca = 2000 L/mol → 2.0 m³/mol
  K_Cl = 0.447 L/mol → 4.47e-4 m³/mol  (≡SiOH + Cl⁻ ⇌ ≡SiOHCl⁻)
  K_Na = 1.106e-3 (tobermorite, x = 0.83) … 9.0e-5 (jennite, x = 1.67) m³/mol
  a_s  = 85 000 m²/mol  (S_BET = 500 m²/g × M_CSH ≈ 170 g/mol, Soive 2017)
"""
DLM_TRAN2018(; kwargs...) = DLM(; mechanism = OuterSphere(), kwargs...)

"""
    DLM_TERNARY(; kwargs...)

Ternary neutral set: the complex ≡SiOCaCl of Thermoddem (2023), with the site densities of
Yoshida et al. (2021, Fig. 5) — 4.3 nm⁻² = 7.14×10⁻⁶ mol·m⁻² for tobermorite and
7.0 nm⁻² = 1.162×10⁻⁵ mol·m⁻² for jennite, interpolated in x_CaS.

`K_Cl` here is the ternary constant, 1.585×10⁻¹³, and is **not** commensurable with the
outer-sphere one: the two mechanisms consume different reactants.
"""
DLM_TERNARY(; kwargs...) = DLM(;
    K_Cl = 1.585e-13,
    Gamma_max_Tob = 7.14e-6,
    Gamma_max_Jen = 1.162e-5,
    mechanism = TernaryNeutral(),
    kwargs...
)

# ── Interpolations in the Ca/Si ratio ─────────────────────────────────────────

const _X_TOB = 0.83
const _X_JEN = 1.67

"""
    k_na(dlm, x_cas)

Alkali constant, linear in `x_cas` between the tobermorite and jennite end members:
`K_Na(x) = K_Na_Jen + (K_Na_Tob − K_Na_Jen)(x_J − x)/(x_J − x_T)`, clamped outside.
"""
function k_na(dlm::DLM, x_cas)
    x_c = clamp(x_cas, _X_TOB, _X_JEN)
    return dlm.K_Na_Jen + (dlm.K_Na_Tob - dlm.K_Na_Jen) * (_X_JEN - x_c) / (_X_JEN - _X_TOB)
end

"""
    gamma_max(dlm, x_cas)

Site density, linear in `x_cas` between the same end members. Equal endpoints give a
constant, which is how the binary sets are written.
"""
function gamma_max(dlm::DLM, x_cas)
    t = clamp((x_cas - _X_TOB) / (_X_JEN - _X_TOB), 0, 1)
    return dlm.Gamma_max_Tob + t * (dlm.Gamma_max_Jen - dlm.Gamma_max_Tob)
end

# ── Site sum and charge sum ───────────────────────────────────────────────────
#
# Both sums are normalised by [≡SiOH]. `A` is the site balance — the θᵢ below are its
# terms — so site conservation is structural and needs no separate equation. `B` collects
# the *charged* complexes only, which is where the two mechanisms part company.

## The chloride term of the site sum. Present in both mechanisms, with different
## reactants: outer-sphere consumes Cl⁻ alone and takes the Boltzmann factor of a −1
## complex; the ternary consumes Ca²⁺ and Cl⁻ and releases H⁺, and being neutral takes no
## factor at all.
_cl_site(::OuterSphere, K_Cl, β, cCl, cCa, cH) = K_Cl * cCl * exp(β)
_cl_site(::TernaryNeutral, K_Cl, β, cCl, cCa, cH) = K_Cl * cCa * cCl / cH

## The chloride term of the charge sum: −1 for the outer-sphere complex, exactly nothing
## for the neutral one. `zero` rather than `0.0`, so a dual parameter is not stripped.
_cl_charge(::OuterSphere, K_Cl, β, cCl, cH) = K_Cl * cCl * exp(β)
_cl_charge(m::TernaryNeutral, K_Cl, β, cCl, cH) = zero(promote_type(typeof(K_Cl), typeof(β), typeof(cCl), typeof(cH)))

## The same chloride term as `_cl_site`, but multiplied by `[≡SiOH]` in the factor order
## the three replaced implementations used. Floating-point multiplication is commutative
## and **not** associative, so `K X c exp(β)` and `(K c exp(β)) X` differ in the last
## bits; keeping the original order is what makes the replacement exact rather than
## merely equivalent.
_cl_theta(::OuterSphere, K_Cl, X, β, cCl, cCa, cH) = K_Cl * X * cCl * exp(β)
_cl_theta(::TernaryNeutral, K_Cl, X, β, cCl, cCa, cH) = K_Cl * X * cCa * cCl / cH

"""
    dlm_site_sum(β, cCl, cNa, cK, cCa, cMg, cH, dlm, x_cas) -> A

Sum of the surface species normalised by `[≡SiOH]`, so that `[≡SiOH] = Γ_max / A`.
"""
function dlm_site_sum(β, cCl, cNa, cK, cCa, cMg, cH, dlm::DLM, x_cas)
    Ka1 = dlm.Ka1
    KNa = k_na(dlm, x_cas)
    T = promote_type(typeof(β), typeof(cCl), typeof(cH), eltype(dlm))
    return one(T) +
        Ka1 * exp(β) / cH +                        # ≡SiO⁻
        dlm.K_Ca * Ka1 * cCa * exp(-β) / cH +      # ≡SiOCa⁺
        dlm.K_Mg * Ka1 * cMg * exp(-β) / cH +      # ≡SiOMg⁺
        _cl_site(dlm.mechanism, dlm.K_Cl, β, cCl, cCa, cH) +
        (KNa * cNa + KNa * cK) * Ka1 / cH          # ≡SiONa + ≡SiOK, K_K = K_Na
end

"""
    dlm_charge_sum(β, cCl, cCa, cMg, cH, dlm) -> B

Signed sum of the *charged* surface species, normalised the same way, so that the surface
charge is `σ₀ = F Γ_max B / A`.
"""
function dlm_charge_sum(β, cCl, cCa, cMg, cH, dlm::DLM)
    Ka1 = dlm.Ka1
    return dlm.K_Ca * Ka1 * cCa * exp(-β) / cH +   # ≡SiOCa⁺   +1
        dlm.K_Mg * Ka1 * cMg * exp(-β) / cH -      # ≡SiOMg⁺   +1
        Ka1 * exp(β) / cH -                        # ≡SiO⁻     −1
        _cl_charge(dlm.mechanism, dlm.K_Cl, β, cCl, cH)
end

# ── The residual ──────────────────────────────────────────────────────────────

const _F_FARADAY = 96485.0        # [C/mol]
const _R_GAS = 8.314              # [J/mol/K]
const _EPS_0 = 8.854e-12          # [F/m]

"""
    ionic_strength(cCl, cNa, cK, cCa, cMg, cOH, cH)

Ionic strength [mol·m⁻³], floored at 1 so the diffuse-layer capacitance stays finite in a
nearly pure solution.
"""
function ionic_strength(cCl, cNa, cK, cCa, cMg, cOH, cH)
    T = promote_type(typeof(cCl), typeof(cNa), typeof(cK), typeof(cCa),
        typeof(cMg), typeof(cOH), typeof(cH))
    return max(0.5 * (cCl + cNa + cK + 4 * cCa + 4 * cMg + cOH + cH), one(T))
end

"""
    dlm_residual(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, x_cas; dlm, T_K) -> f(β)

Gouy-Chapman charge balance, `σ₀(β) − σ_DL(β)`, **without any root finding**:

```math
f(β) = F Γ_\\max \\frac{B(β)}{A(β)} - \\sqrt{8 ε_0 ε_r R T I}\\, \\sinh(β/2)
```

Zero at the surface potential. This is the function a globally implicit scheme puts in
`reaction!` with β as a nodal unknown; [`solve_dlm`](@ref) is the sequential scheme's way
of using the same expression.
"""
function dlm_residual(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, x_cas; dlm::DLM, T_K = 293.15)
    c_H = dlm.Kw / max(c_OH, 1.0e-20)
    I = ionic_strength(c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, c_H)
    σ_cap = sqrt(8 * _EPS_0 * dlm.eps_r * _R_GAS * T_K * I)
    A = dlm_site_sum(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_H, dlm, x_cas)
    B = dlm_charge_sum(β, c_Cl, c_Ca, c_Mg, c_H, dlm)
    return _F_FARADAY * gamma_max(dlm, x_cas) * B / A - σ_cap * sinh(β / 2)
end

"""
    dlm_loadings(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm)
        -> (S_Cl, S_Na, S_K, S_Ca, S_Mg)

Adsorbed amounts [mol·m⁻³ of medium] at a **given** surface potential — explicit, no
solve. `S_i = a_s · n_csh · θ_i(β, c)`.
"""
function dlm_loadings(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm::DLM)
    Ka1 = dlm.Ka1
    KNa = k_na(dlm, x_cas)
    c_H = dlm.Kw / max(c_OH, 1.0e-20)
    X = gamma_max(dlm, x_cas) /
        dlm_site_sum(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_H, dlm, x_cas)   # [≡SiOH], mol/m²

    θ_Cl = _cl_theta(dlm.mechanism, dlm.K_Cl, X, β, c_Cl, c_Ca, c_H)
    θ_Ca = dlm.K_Ca * Ka1 * X * c_Ca * exp(-β) / c_H
    θ_Mg = dlm.K_Mg * Ka1 * X * c_Mg * exp(-β) / c_H
    θ_Na = KNa * Ka1 * X * c_Na / c_H
    θ_K = KNa * Ka1 * X * c_K / c_H

    fac = dlm.a_s * n_csh
    return θ_Cl * fac, θ_Na * fac, θ_K * fac, θ_Ca * fac, θ_Mg * fac
end

# ── The solver ────────────────────────────────────────────────────────────────

"""
    solve_dlm(c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm, T_K = 293.15)
        -> (β, S_Cl, S_Na, S_K, S_Ca, S_Mg)

Surface potential and adsorbed amounts. Concentrations in mol·m⁻³ of pore solution,
`n_csh` and the returned loadings in mol·m⁻³ of medium.

Returns zeros when `n_csh ≤ 0` — no C-S-H, no surface — with `zero(T)` rather than the
literal, so the dual type survives the branch.

## Differentiability

Bisection is a sequence of comparisons on floating-point values whose bracket endpoints are
constants, so differentiating through it returns `dβ/dc = 0`. The bracket is therefore
closed on the **stripped** values, and one Newton step at the converged root restores the
derivative: `f(β★)` is zero to the bisection tolerance, so β does not move, while its dual
part becomes exactly `−(∂f/∂c)/(∂f/∂β)` — the implicit-function derivative of the root.
Any root finder added here needs the same treatment.
"""
function solve_dlm(
        c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas;
        dlm::DLM, T_K = 293.15,
    )
    T_out = promote_type(
        typeof(c_Cl), typeof(c_Na), typeof(c_K), typeof(c_Ca),
        typeof(c_Mg), typeof(c_OH), typeof(n_csh), eltype(dlm)
    )
    n_csh ≤ 0 && return ntuple(_ -> zero(T_out), 6)

    f(β) = dlm_residual(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, x_cas; dlm = dlm, T_K = T_K)

    v(x) = ForwardDiff.value(x)
    fv(β) = dlm_residual(
        β, v(c_Cl), v(c_Na), v(c_K), v(c_Ca), v(c_Mg), v(c_OH), v(x_cas);
        dlm = _stripped(dlm), T_K = v(T_K)
    )

    β_lo, β_hi = -10.0, 10.0
    f_lo, f_hi = fv(β_lo), fv(β_hi)
    bracketed = f_lo * f_hi < 0.0

    β_star = 0.0
    if bracketed
        for _ in 1:64
            β_mid = 0.5 * (β_lo + β_hi)
            f_mid = fv(β_mid)
            if f_mid * f_lo < 0.0
                β_hi = β_mid
            else
                β_lo = β_mid
                f_lo = f_mid
            end
            abs(β_hi - β_lo) < 1.0e-9 && break
        end
        β_star = 0.5 * (β_lo + β_hi)
    else
        ## No sign change on [−10, 10]: there is no root to differentiate, so β is a
        ## constant and carries no derivative. That is the honest answer, not a failure.
        β_star = abs(f_lo) < abs(f_hi) ? β_lo : β_hi
    end

    β = bracketed ? β_star - f(β_star) / ForwardDiff.derivative(fv, β_star) : β_star

    S_Cl, S_Na, S_K, S_Ca, S_Mg =
        dlm_loadings(β, c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm = dlm)
    return β, S_Cl, S_Na, S_K, S_Ca, S_Mg
end

## The bracket must be closed on plain values. When a *parameter* carries the dual — which
## is the point of `DLM{T}` — stripping the concentrations is not enough, the constants
## have to be stripped too.
_stripped(dlm::DLM{<:AbstractFloat}) = dlm
function _stripped(dlm::DLM)
    v = ForwardDiff.value
    return DLM{typeof(v(dlm.Ka1)), typeof(dlm.mechanism)}(
        v(dlm.Ka1), v(dlm.K_Ca), v(dlm.K_Mg), v(dlm.K_Cl),
        v(dlm.K_Na_Tob), v(dlm.K_Na_Jen),
        v(dlm.Gamma_max_Tob), v(dlm.Gamma_max_Jen),
        v(dlm.a_s), v(dlm.eps_r), v(dlm.Kw), v(dlm.x_cas), v(dlm.n_csh0),
        dlm.mechanism,
    )
end
