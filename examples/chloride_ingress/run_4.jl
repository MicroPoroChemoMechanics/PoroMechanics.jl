# chloride_ingress example — Phase 4: TOUGHREACT / SNIA + DLM surface complexation on C-S-H
#
# Reference: Tran, V.Q., Soive, A., Bonnet, S., Khelidj, A. (2018).
#   A numerical model including thermodynamic equilibrium, kinetic control and
#   surface complexation in order to explain chloride binding capacity of concrete.
#   Cement and Concrete Research, 110, 70–85.
#
# Compared with Phase 3 (Langmuir + linear), the phenomenological isotherms are
# replaced by DLM (Double Layer Model) surface complexation:
#
#   Surface reactions on C-S-H (≡SiOH = neutral silanol site):
#     ≡SiOH  ⇌  ≡SiO⁻ + H⁺              Ka1  = 2.0×10⁻¹³ mol/L  (deprotonation)
#     ≡SiO⁻ + Ca²⁺  ⇌  ≡SiOCa⁺          K_Ca = 2000 L/mol        (inner-sphere)
#     ≡SiOH + Cl⁻   ⇌  ≡SiOHCl⁻         K_Cl = 0.447 L/mol       (outer-sphere)
#     ≡SiO⁻ + Na⁺   ⇌  ≡SiONa            K_Na(x_cas) interpolated (inner-sphere)
#     ≡SiO⁻ + K⁺    ⇌  ≡SiOK             K_K  = K_Na              (inner-sphere)
#
#   Diffuse double layer equation (Gouy-Chapman):
#     σ₀ = √(8·ε₀·ε_r·R·T·I) · sinh(β/2)   with β = F·ψ/(R·T)
#
# SNIA strategy:
#   1. Transport: pure Fick (4 primary ions), linear retardation ≈ local DLM.
#   2. Chemistry: equilibrate() (Gibbs) + solve_dlm() (DLM), node by node.
#   3. Update of the effective K_d after each chemical step.
#
# Usage :
#   julia --project examples/chloride_ingress/run_4.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver
using ForwardDiff

## `unique_species` and the element balance — the guards on the dialogue with
## ChemistryLab. See `element_balance.jl` for why the two questions are kept apart.
include("element_balance.jl")

# ── Transport species indices ─────────────────────────────────────────────────
const ICL4 = 1   # c_Cl   z = -1
const INA4 = 2   # c_Na   z = +1
const IK4 = 3   # c_K    z = +1
const ICA4 = 4   # c_Ca   z = +2

const V_REV_4 = 1.0e-3   # [m³] = 1 dm³

# ── DLM parameters ────────────────────────────────────────────────────────────
#
# NOTE — this belongs in ChemistryLab.jl.
# Surface complexation is chemistry, not transport. It lives here only because
# ChemistryLab.jl does not expose it yet; when it does, delete this and call it.
# Three near-identical copies of the model currently exist in this directory
# (run_4.jl, tran2018.jl, chloride_ternary.jl), which is the argument for moving it.


"""
Parameters of the DLM (Double Layer Model) for C-S-H.

All equilibrium constants are in SI units (mol/m³ or m³/mol).
Ka1 and K_OHCl: converted from the mol/L and L/mol of Tran 2018 (× 10⁻³).

Tran 2018 parameters (Table 1):
  Ka1 = 2.0e-13 mol/L → 2.0e-10 mol/m³
  K_Ca = 2000 L/mol → 2.0 m³/mol
  K_OHCl = 0.447 L/mol → 4.47e-4 m³/mol
  K_Na (Tobermorite, x=0.83) = 1.106 L/mol → 1.106e-3 m³/mol
  K_Na (Jennite, x=1.67) = 0.090 L/mol → 9.0e-5 m³/mol
  Γ_max = 1.3e-6 mol/m²_CSH
  a_s = 85 000 m²/mol_CSH  (S_BET = 500 m²/g × M_CSH ≈ 170 g/mol)
"""
Base.@kwdef struct DLMParams
    Ka1::Float64 = 2.0e-10   # [mol/m³]  deprotonation ≡SiOH → ≡SiO⁻ + H⁺
    K_Ca::Float64 = 2.0       # [m³/mol]  ≡SiO⁻ + Ca²⁺ → ≡SiOCa⁺
    K_OHCl::Float64 = 4.47e-4   # [m³/mol]  ≡SiOH + Cl⁻ → ≡SiOHCl⁻
    K_Na_Tob::Float64 = 1.106e-3  # [m³/mol]  ≡SiO⁻ + Na⁺ → ≡SiONa (x=0.83)
    K_Na_Jen::Float64 = 9.0e-5    # [m³/mol]  ≡SiO⁻ + Na⁺ → ≡SiONa (x=1.67)
    Gamma_max::Float64 = 1.3e-6    # [mol/m²_CSH] site density
    a_s::Float64 = 85000.0   # [m²_CSH/mol_CSH] BET specific surface area
    x_cas::Float64 = 1.5       # [-] Ca/Si ratio of the C-S-H (fixed)
    n_csh0::Float64 = 2000.0    # [mol_CSH/m³_concrete] initial C-S-H content
    eps_r::Float64 = 78.5      # [-] relative permittivity of water
    Kw_SI::Float64 = 6.76e-9   # [mol²/m⁶] ionic product of water at 20 °C
end

"""
    k_na_dlm(dlm, x_cas) -> Float64

Linear interpolation of K_Na (and K_K) between Tobermorite and Jennite:
  K_Na(x) = K_Na_Jen + (K_Na_Tob − K_Na_Jen) × (1.67 − x) / (1.67 − 0.83)
"""
function k_na_dlm(dlm::DLMParams, x_cas::Float64)
    xT, xJ = 0.83, 1.67
    x_c = clamp(x_cas, xT, xJ)
    return dlm.K_Na_Jen + (dlm.K_Na_Tob - dlm.K_Na_Jen) * (xJ - x_c) / (xJ - xT)
end

"""
    solve_dlm(c_Cl, c_Na, c_K, c_Ca, c_OH, n_csh; dlm, T_K)

Solves the DLM equilibrium for the C-S-H and returns the surface potential
β = F·ψ/(R·T) together with the adsorbed concentrations (mol/m³_concrete).

**Method**: bisection on β over [-10, 10] of the equation:
    F·Γ_max · B(β)/A(β) = √(8·ε₀·ε_r·R·T·I) · sinh(β/2)

where:
  A(β) = denominator of the site balance
  B(β) = (≡SiOCa⁺ − ≡SiO⁻ − ≡SiOHCl⁻) / [≡SiOH]  (normalised net charge)

**Returns** `(β, S_Cl, S_Na, S_K, S_Ca)` in mol/m³_concrete.
The S_i values can come out negative (which has no physical meaning); they are
guarded by `max(., 0)` upstream when the model is updated.
"""
function solve_dlm(
    c_Cl::Real, c_Na::Real, c_K::Real, c_Ca::Real, c_OH::Real,
    n_csh::Real;
    dlm::DLMParams,
    T_K::Real=293.15,
)
    c_H = dlm.Kw_SI / max(c_OH, 1.0e-20)
    # Ionic strength [mol/m³]
    I = max(0.5 * (c_Cl + c_Na + c_K + 4.0 * c_Ca + c_OH + c_H), 1.0)

    Ka1 = dlm.Ka1
    KCa = dlm.K_Ca
    KCl = dlm.K_OHCl
    KNa = k_na_dlm(dlm, dlm.x_cas)
    KK = KNa    # K⁺: same constant as Na⁺ (Tran 2018)

    # DL capacitance coefficient [C/m²] = √(8·ε₀·ε_r·R·T·I)
    F_val = 96485.0
    R_val = 8.314
    σ_cap = sqrt(8.0 * 8.854e-12 * dlm.eps_r * R_val * T_K * I)

    # Surface speciation, normalised by [≡SiOH]. Written as functions of the
    # concentrations rather than closing over them, so that the same expressions serve
    # both the live (possibly dual) arguments and the stripped values the bracket needs.
    A(β, cCl, cNa, cK, cCa, cH) = (1.0 +
                                   Ka1 * exp(β) / cH +                # ≡SiO⁻
                                   KCa * Ka1 * cCa * exp(-β) / cH +   # ≡SiOCa⁺
                                   KCl * cCl * exp(β) +               # ≡SiOHCl⁻
                                   (KNa * cNa + KK * cK) * Ka1 / cH)  # ≡SiONa + ≡SiOK (no β)

    B(β, cCa, cCl, cH) = (KCa * Ka1 * cCa * exp(-β) / cH -   # +1
                          Ka1 * exp(β) / cH -                 # −1
                          KCl * cCl * exp(β))                 # −1

    # f(β) = σ₀(β) − σ_DL(β)  (residual of the DLM balance equation)
    residual(β, cCl, cNa, cK, cCa, cH, σ) =
        F_val * dlm.Gamma_max * B(β, cCa, cCl, cH) / A(β, cCl, cNa, cK, cCa, cH) -
        σ * sinh(β / 2.0)

    f(β) = residual(β, c_Cl, c_Na, c_K, c_Ca, c_H, σ_cap)

    # Bisection is a sequence of comparisons on floating-point values: differentiating
    # through it would return dβ/dc = 0, because the bracket endpoints are constants
    # rather than functions of the concentrations. So the bracket is closed on the
    # stripped values, and the derivative is restored afterwards by a single Newton step
    # at the converged root. `f(β★)` is zero to the bisection tolerance, so the step does
    # not move β; its dual part is exactly −(∂f/∂c)/(∂f/∂β), the implicit-function
    # derivative of the root.
    v(x) = ForwardDiff.value(x)
    fv(β) = residual(β, v(c_Cl), v(c_Na), v(c_K), v(c_Ca), v(c_H), v(σ_cap))

    β_lo, β_hi = -10.0, 10.0
    f_lo = fv(β_lo)
    f_hi = fv(β_hi)
    bracketed = f_lo * f_hi < 0.0

    β_star = 0.0   # fallback: no surface potential
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
        # No sign change: f stays positive or negative across [-10, 10].
        # Pick the β where |f| is smallest (closest to balance). There is no root here,
        # so β is a constant and carries no derivative — which is the honest answer.
        β_star = abs(f_lo) < abs(f_hi) ? β_lo : β_hi
    end

    β = bracketed ? β_star - f(β_star) / ForwardDiff.derivative(fv, β_star) : β_star

    X = dlm.Gamma_max / A(β, c_Cl, c_Na, c_K, c_Ca, c_H)   # [≡SiOH] in mol/m²_CSH

    # Surface species [mol/m²_CSH]
    theta_OCa = KCa * Ka1 * X * c_Ca * exp(-β) / c_H
    theta_OHCl = KCl * X * c_Cl * exp(β)
    theta_ONa = KNa * Ka1 * X * c_Na / c_H
    theta_OK = KK * Ka1 * X * c_K / c_H

    # Bulk adsorbed [mol/m³_concrete]
    fac = dlm.a_s * n_csh
    S_Cl = theta_OHCl * fac
    S_Na = theta_ONa * fac
    S_K = theta_OK * fac
    S_Ca = theta_OCa * fac

    return β, S_Cl, S_Na, S_K, S_Ca
end

# ── Model ─────────────────────────────────────────────────────────────────────

"""
Parameters of the chloride_ingress model — Phase 4 (TOUGHREACT + C-S-H DLM).

Transport : 4 primary species (Cl⁻, Na⁺, K⁺, Ca²⁺), Fick's law.
Chemistry : Gibbs through equilibrate() + DLM surface complexation.
Adsorption: DLM replaces the phenomenological isotherms of Phase 3.
"""
mutable struct ChlorideModel4 <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64

    # ── Spatial fields ────────────────────────────────────────────────────────
    phi::Vector{Float64}
    n_ch::Vector{Float64}
    n_ett::Vector{Float64}
    n_ms::Vector{Float64}
    n_fs::Vector{Float64}
    c_so4_local::Vector{Float64}
    c_oh_frozen::Vector{Float64}

    # ── DLM: adsorbed concentrations [mol/m³_concrete] ────────────────────────
    S_Cl::Vector{Float64}
    S_Na::Vector{Float64}
    S_K::Vector{Float64}
    S_Ca::Vector{Float64}

    # ── DLM: effective K_d (secant approx., updated after each chemistry step) ─
    Kd_Cl::Vector{Float64}
    Kd_Na::Vector{Float64}
    Kd_K::Vector{Float64}
    Kd_Ca::Vector{Float64}

    # ── DLM parameters ────────────────────────────────────────────────────────
    dlm::DLMParams

    # ── Diffusion ionique en eau libre [m²/s] ─────────────────────────────────
    D_Cl::Float64
    D_Na::Float64
    D_K::Float64
    D_Ca::Float64

    # ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────
    phi_c::Float64
    n_OJ::Float64
    ds_OJ::Float64
    tau_agg::Float64

    # ── Physical constants ────────────────────────────────────────────────────
    Faraday::Float64
    R_gas::Float64
    T_K::Float64

    # ── Boundary conditions (x = 0: external NaCl) ────────────────────────────
    c_cl_BC::Float64
    c_na_BC::Float64
    c_k_BC::Float64
    c_ca_BC::Float64

    # ── Conditions initiales ──────────────────────────────────────────────────
    c_cl_init::Float64
    c_na_init::Float64
    c_k_init::Float64
    c_ca_init::Float64
end

# Reuses compute_opc_ic from run_3.jl — copied here so this file stands alone.
# (Same implementation; only the name V_REV_4 differs, its value is still 1e-3 m³.)
function _compute_opc_ic4(
    cs,
    has_friedels::Bool;
    phi0=0.121,
    T_K=293.15,
    n_ch0=1640.0,
    n_ms0=100.0,
    n_ett0=0.0,
    m_clinker=350.0,
    na2o_frac=0.0016,
    k2o_frac=0.0049,
)
    T_q = T_K * us"K"
    V_liq = phi0 * V_REV_4
    m_ck = m_clinker * V_REV_4
    M_Na2O, M_K2O = 61.98e-3, 94.20e-3
    n_Na2O = na2o_frac * m_ck / M_Na2O
    n_K2O = k2o_frac * m_ck / M_K2O
    n_na = 2.0 * n_Na2O
    n_k = 2.0 * n_K2O
    n_oh_alk = n_na + n_k
    n_h2o_ox = n_Na2O + n_K2O
    n_water = V_liq * 55_500.0 - n_h2o_ox
    n_ch_mol = n_ch0 * V_REV_4
    n_ms_mol = n_ms0 * V_REV_4
    n_ett_mol = n_ett0 * V_REV_4

    state = ChemicalState(cs; T=T_q)
    set_quantity!(state, "H2O@", n_water * us"mol")
    set_quantity!(state, "Na+", n_na * us"mol")
    set_quantity!(state, "K+", n_k * us"mol")
    set_quantity!(state, "OH-", n_oh_alk * us"mol")
    set_quantity!(state, "Cl-", 1.0e-16 * us"mol")
    set_quantity!(state, "Portlandite", n_ch_mol * us"mol")
    set_quantity!(state, "ettringite", n_ett_mol * us"mol")
    set_quantity!(state, "monosulphate12", n_ms_mol * us"mol")
    set_quantity!(state, "SO4-2", 1.0e-16 * us"mol")

    @info "compute_opc_ic4: initial OPC equilibrium computation…"
    local state_eq
    try
        state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))
    catch e
        error("OPC IC equilibrate failed: $e")
    end

    V_liq_eq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
    V_liq_eq < 1e-15 && error("Zero liquid volume after the OPC IC equilibrate")

    c_ca = ustrip(moles(state_eq, "Ca+2")) / V_liq_eq
    c_cl = ustrip(moles(state_eq, "Cl-")) / V_liq_eq
    c_na = ustrip(moles(state_eq, "Na+")) / V_liq_eq
    c_k = ustrip(moles(state_eq, "K+")) / V_liq_eq
    c_oh = ustrip(moles(state_eq, "OH-")) / V_liq_eq
    c_so4 = ustrip(moles(state_eq, "SO4-2")) / V_liq_eq

    n_ch_eq = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_4, 0.0)
    n_ett_eq = max(ustrip(moles(state_eq, "ettringite")) / V_REV_4, 0.0)
    n_ms_eq = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_4, 0.0)
    n_fs_eq = has_friedels ? max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_4, 0.0) : 0.0

    @info "Initial OPC equilibrium:" φ = round(phi0; digits=4) c_OH = round(c_oh; sigdigits=4) c_Ca = round(c_ca; sigdigits=4) n_CH = round(n_ch_eq; digits=0)

    return (phi=phi0, n_ch=n_ch_eq, n_ett=n_ett_eq, n_ms=n_ms_eq, n_fs=n_fs_eq,
        c_ca=c_ca, c_cl=c_cl, c_na=c_na, c_k=c_k, c_oh=c_oh, c_so4=c_so4)
end

"""
    ChlorideModel4(N_nodes, cs, has_friedels; dlm, kwargs...)

Constructor with thermodynamic OPC initialisation + initial DLM.
`dlm` : `DLMParams` (optional, defaults to Tran 2018).
`kwargs` : forwarded to `_compute_opc_ic4` (phi0, T_K, n_ch0, …).
"""
function ChlorideModel4(
    N_nodes::Int, cs, has_friedels::Bool;
    dlm::DLMParams=DLMParams(),
    kwargs...,
)
    ic = _compute_opc_ic4(cs, has_friedels; kwargs...)
    N = N_nodes

    # ── DLM at the initial OPC state (Cl ≈ 0, in equilibrium with the pore solution) ─
    _, S_Cl0, S_Na0, S_K0, S_Ca0 = solve_dlm(
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca, ic.c_oh, dlm.n_csh0;
        dlm=dlm, T_K=get(kwargs, :T_K, 293.15),
    )
    ε = 1.0e-15
    Kd_Cl0 = max(S_Cl0, 0.0) / max(ic.c_cl, ε)
    Kd_Na0 = max(S_Na0, 0.0) / max(ic.c_na, ε)
    Kd_K0 = max(S_K0, 0.0) / max(ic.c_k, ε)
    Kd_Ca0 = max(S_Ca0, 0.0) / max(ic.c_ca, ε)

    @info "DLM initial OPC :" S_Cl = round(S_Cl0; sigdigits=3) S_Na = round(S_Na0; sigdigits=3) S_K = round(S_K0; sigdigits=3) S_Ca = round(S_Ca0; sigdigits=3)

    return ChlorideModel4(
        0.05,                             # L
        fill(ic.phi, N),
        fill(ic.n_ch, N),
        fill(ic.n_ett, N),
        fill(ic.n_ms, N),
        fill(ic.n_fs, N),
        fill(ic.c_so4, N),
        fill(ic.c_oh, N),
        # DLM surface loadings (initial)
        fill(max(S_Cl0, 0.0), N),
        fill(max(S_Na0, 0.0), N),
        fill(max(S_K0, 0.0), N),
        fill(max(S_Ca0, 0.0), N),
        # Effective K_d (initial)
        fill(Kd_Cl0, N),
        fill(Kd_Na0, N),
        fill(Kd_K0, N),
        fill(Kd_Ca0, N),
        dlm,
        # Diffusion in free water [m²/s]
        2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9,
        # Oh-Jang tortuosity params
        0.18, 2.7, 2.0e-4, 0.27,
        # Physical constants
        96485.0, 8.314, 293.15,
        # BCs (x=0 : 0.523 mol/L NaCl)
        523.0, 523.0, 1.0, 0.0,
        # ICs
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca,
    )
end

PoroMechanics.nspecies(::ChlorideModel4) = 4
PoroMechanics.species_names(::ChlorideModel4) = [:c_Cl, :c_Na, :c_K, :c_Ca]

# ── Fonctions utilitaires ─────────────────────────────────────────────────────

"""
    _tortuosity4(phi, m::ChlorideModel4)

Oh-Jang tortuosity of the cement paste, from the package constitutive layer.
"""
function _tortuosity4(phi, m::ChlorideModel4)
    ## Saturated medium: S_l = 1, so the saturation factor is one.
    oj = OhJang(; phi_c = m.phi_c, n = m.n_OJ, ds = m.ds_OJ, tau_agg = m.tau_agg)
    return tortuosity(oj, phi, 1)
end

@inline function _node_idx4(x::Float64, m::ChlorideModel4)
    N = length(m.phi) - 1
    return clamp(round(Int, x / (m.L / N)) + 1, 1, N + 1)
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

"""
    PoroMechanics.storage!(f, u, node, m::ChlorideModel4, ::Any)

Storage term: φ·c + S_DLM ≈ (φ + K_d_eff)·c  (local linear approximation).
K_d_eff[i] = S_DLM[i] / c[i] is updated after every chemical step.
"""
function PoroMechanics.storage!(f, u, node, m::ChlorideModel4, ::Any)
    i = _node_idx4(node.coord[1], m)
    phi = m.phi[i]
    f[ICL4] = (phi + m.Kd_Cl[i]) * u[ICL4]
    f[INA4] = (phi + m.Kd_Na[i]) * u[INA4]
    f[IK4] = (phi + m.Kd_K[i]) * u[IK4]
    f[ICA4] = (phi + m.Kd_Ca[i]) * u[ICA4]
end

function PoroMechanics.flux!(f, u, edge, m::ChlorideModel4, ::Any)
    x_mid = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    i = _node_idx4(x_mid, m)
    phi_i = (i < length(m.phi)) ? (m.phi[i] + m.phi[i+1]) / 2 : m.phi[i]
    tau = _tortuosity4(phi_i, m)
    f[ICL4] = m.D_Cl * tau * (u[ICL4, 1] - u[ICL4, 2])
    f[INA4] = m.D_Na * tau * (u[INA4, 1] - u[INA4, 2])
    f[IK4] = m.D_K * tau * (u[IK4, 1] - u[IK4, 2])
    f[ICA4] = m.D_Ca * tau * (u[ICA4, 1] - u[ICA4, 2])
end

function PoroMechanics.bcondition!(f, u, bnode, m::ChlorideModel4, ::Any)
    boundary_dirichlet!(f, u, bnode; species=ICL4, region=1, value=m.c_cl_BC)
    boundary_dirichlet!(f, u, bnode; species=INA4, region=1, value=m.c_na_BC)
    boundary_dirichlet!(f, u, bnode; species=IK4, region=1, value=m.c_k_BC)
    boundary_dirichlet!(f, u, bnode; species=ICA4, region=1, value=m.c_ca_BC)
end

# ── ChemistryLab initialisation (same as Phase 3) ─────────────────────────────

function _init_chemistry4()
    data_path = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    isfile(data_path) || error("cemdata18 introuvable : $data_path")
    @info "Loading cemdata18…"
    substances = build_species(data_path)
    dict_sp = Dict(symbol(s) => s for s in substances)
    seeds = split("Portlandite ettringite monosulphate12 H2O@ Ca+2 OH- Cl- Na+ K+ H+ Al+3 SO4-2")
    aq_species = speciation(substances, seeds; aggregate_state=[AS_AQUEOUS])
    solid_names = ["Portlandite", "ettringite", "monosulphate12", "C4AClH10"]
    solid_species = [dict_sp[n] for n in solid_names if haskey(dict_sp, n)]
    missing_s = filter(n -> !haskey(dict_sp, n), solid_names)
    isempty(missing_s) || @warn "Phases not found: $(join(missing_s, ", "))"
    has_friedels = haskey(dict_sp, "C4AClH10")
    has_friedels || @warn "C4AClH10 missing — thermodynamic Cl⁻ binding disabled"
    species = vcat(collect(aq_species), solid_species)
    @info "Selected species" n_aq = length(aq_species) n_solid = length(solid_species)
    cs = ChemicalSystem(unique_species(species), CEMDATA_PRIMARIES)
    return cs, has_friedels
end

# ── Chemical step: Gibbs + DLM ────────────────────────────────────────────────

"""
    chemistry_step4!(m, u, cs, has_friedels)

SNIA chemical step — Phase 4 (TOUGHREACT + DLM).

For every interior node (i ≥ 2):
  1. Recover OH⁻ from the primary charge balance (EN).
  2. equilibrate() (Gibbs) : redistribution thermodynamique.
  3. solve_dlm(): DLM surface complexation on C-S-H.
  4. Update: free concentrations, solids, porosity, K_d_eff.

The K_d_eff[i] are updated to feed storage!() on the next transport segment:
K_d_Cl = dS_DLM/dc + 2·dN_fs/dc (tangents), K_d_Na/K/Ca = secant.
"""
function chemistry_step4!(m::ChlorideModel4, u::Matrix, cs, has_friedels::Bool)
    N = size(u, 2)
    T_q = m.T_K * us"K"
    Vm_CH = 33.06e-6
    Vm_ett = 707.0e-6
    Vm_ms = 309.0e-6
    Vm_fs = 271.0e-6
    ε = 1.0e-15

    ## Species positions, resolved once: the tangent below seeds the composition vector
    ## by index rather than by name, because `set_quantity!` writes into a Float64 array
    ## and a dual seed has to be laid out at construction time.
    sp_names = String.(symbol.(cs.species))
    i_cl = findfirst(==("Cl-"), sp_names)
    i_oh = findfirst(==("OH-"), sp_names)
    i_fs = has_friedels ? findfirst(==("C4AClH10"), sp_names) : nothing

    for i in 2:N
        phi_i = m.phi[i]
        C_Cl = max(u[ICL4, i], 0.0)
        C_Na = max(u[INA4, i], 0.0)
        C_K = max(u[IK4, i], 0.0)
        C_Ca = max(u[ICA4, i], 0.0)

        n_cl_aq = C_Cl * phi_i * V_REV_4
        n_na = C_Na * phi_i * V_REV_4
        n_k = C_K * phi_i * V_REV_4
        n_ca_aq = C_Ca * phi_i * V_REV_4
        n_oh_en = max(n_na + n_k + 2.0 * n_ca_aq - n_cl_aq, 1.0e-20)
        n_water = phi_i * V_REV_4 * 55_500.0

        state = ChemicalState(cs; T=T_q)
        set_quantity!(state, "H2O@", n_water * us"mol")
        set_quantity!(state, "OH-", n_oh_en * us"mol")
        set_quantity!(state, "Cl-", n_cl_aq * us"mol")
        set_quantity!(state, "Na+", n_na * us"mol")
        set_quantity!(state, "K+", n_k * us"mol")
        set_quantity!(state, "Ca+2", n_ca_aq * us"mol")
        set_quantity!(state, "Portlandite", m.n_ch[i] * V_REV_4 * us"mol")
        set_quantity!(state, "ettringite", m.n_ett[i] * V_REV_4 * us"mol")
        set_quantity!(state, "monosulphate12", m.n_ms[i] * V_REV_4 * us"mol")
        set_quantity!(state, "SO4-2", m.c_so4_local[i] * phi_i * V_REV_4 * us"mol")
        has_friedels && set_quantity!(state, "C4AClH10", m.n_fs[i] * V_REV_4 * us"mol")

        local state_eq
        try
            state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))
        catch e
            @warn "equilibrate failed at node $i — left unchanged" exception = e
            continue
        end

        V_liq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
        (isnan(V_liq) || V_liq < 1e-15) && (@warn "zero/NaN V_liq at node $i"; continue)

        c_ca_new = ustrip(moles(state_eq, "Ca+2")) / V_liq
        c_cl_new = ustrip(moles(state_eq, "Cl-")) / V_liq
        c_na_new = ustrip(moles(state_eq, "Na+")) / V_liq
        c_k_new = ustrip(moles(state_eq, "K+")) / V_liq
        c_oh_new = ustrip(moles(state_eq, "OH-")) / V_liq
        c_so4_new = ustrip(moles(state_eq, "SO4-2")) / V_liq

        n_ch_new = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_4, 0.0)
        n_ett_new = max(ustrip(moles(state_eq, "ettringite")) / V_REV_4, 0.0)
        n_ms_new = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_4, 0.0)
        n_fs_new = has_friedels ? max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_4, 0.0) : 0.0

        # ── Porosity update ───────────────────────────────────────────────────
        Δn_ch = n_ch_new - m.n_ch[i]
        Δn_ett = n_ett_new - m.n_ett[i]
        Δn_ms = n_ms_new - m.n_ms[i]
        Δn_fs = n_fs_new - m.n_fs[i]
        phi_new = phi_i - Δn_ch * Vm_CH - Δn_ett * Vm_ett - Δn_ms * Vm_ms - Δn_fs * Vm_fs
        m.phi[i] = clamp(phi_new, 1e-4, 0.999)
        m.n_ch[i] = n_ch_new
        m.n_ett[i] = n_ett_new
        m.n_ms[i] = n_ms_new
        m.n_fs[i] = n_fs_new
        m.c_so4_local[i] = max(c_so4_new, 0.0)
        m.c_oh_frozen[i] = max(c_oh_new, 1e-12)

        # ── Tangent dN_fs/dc_Cl, differentiated through the Gibbs solve ──────
        # The secant N_fs/c_Cl is 0 at the monosulphate→Friedel conversion front
        # (N_fs=0, c_Cl>0). The tangent dn_fs/dc is >> 0 there, because as soon as
        # c_Cl crosses the [SO4]/K_exch threshold all the monosulphate converts — which
        # is exactly where a difference quotient is least trustworthy and a step size
        # least defensible. So the tangent comes from the solve itself: a `ChemicalState`
        # carrying `ForwardDiff.Dual` amounts takes the implicit-function route through
        # the optimality conditions (one primal solve, then a saddle-point system for the
        # sensitivities), and the answer is exact with no δ to choose.
        phi_new = m.phi[i]
        n_cl_t = max(c_cl_new, 0.0) * phi_new * V_REV_4
        n_na_t = c_na_new * phi_new * V_REV_4
        n_k_t = c_k_new * phi_new * V_REV_4
        n_ca_t = c_ca_new * phi_new * V_REV_4
        n_oh_t = max(n_na_t + n_k_t + 2.0 * n_ca_t - n_cl_t, 1.0e-20)
        n_w_t = phi_new * V_REV_4 * 55_500.0

        state_t = ChemicalState(cs; T=T_q)
        set_quantity!(state_t, "H2O@", n_w_t * us"mol")
        set_quantity!(state_t, "OH-", n_oh_t * us"mol")
        set_quantity!(state_t, "Cl-", n_cl_t * us"mol")
        set_quantity!(state_t, "Na+", n_na_t * us"mol")
        set_quantity!(state_t, "K+", n_k_t * us"mol")
        set_quantity!(state_t, "Ca+2", n_ca_t * us"mol")
        set_quantity!(state_t, "Portlandite", n_ch_new * V_REV_4 * us"mol")
        set_quantity!(state_t, "ettringite", n_ett_new * V_REV_4 * us"mol")
        set_quantity!(state_t, "monosulphate12", n_ms_new * V_REV_4 * us"mol")
        set_quantity!(state_t, "SO4-2", c_so4_new * phi_new * V_REV_4 * us"mol")
        has_friedels && set_quantity!(state_t, "C4AClH10", n_fs_new * V_REV_4 * us"mol")

        # The seed direction is ∂/∂c_Cl in [mol/m³_water]. It is not chloride alone:
        # the same charge balance that sets OH⁻ above ties it to c_Cl, so OH⁻ carries the
        # opposite partial. This is what the perturbed state used to encode by rebuilding
        # `n_oh_p` from `n_cl_p`.
        dn_dc = phi_new * V_REV_4
        n_seed = [ForwardDiff.Dual{Nothing}(ustrip(us"mol", nᵢ), 0.0) for nᵢ in state_t.n]
        n_seed[i_cl] = ForwardDiff.Dual{Nothing}(ustrip(us"mol", state_t.n[i_cl]), dn_dc)
        n_seed[i_oh] = ForwardDiff.Dual{Nothing}(ustrip(us"mol", state_t.n[i_oh]), -dn_dc)

        # Secant fallback if the sensitivity solve fails: dn/dc ← n_fs/c_Cl
        dn_fs_dc = n_fs_new / max(c_cl_new, ε)
        if has_friedels
            try
                state_eq_t = equilibrate(
                    ChemicalState(cs, n_seed .* us"mol"; T=T_q),
                    OptimaOptimizer(tol=1e-10, verbose=false),
                )
                dn_fs_dc = max(
                    ForwardDiff.partials(ustrip(us"mol", state_eq_t.n[i_fs]), 1), 0.0
                ) / V_REV_4   # [mol/m³_concrete / (mol/m³_water)]
            catch e
                @warn "Friedel tangent fell back to the secant at node $i" exception = e maxlog = 1
            end
        else
            dn_fs_dc = 0.0
        end

        # ── DLM surface complexation, and its tangent, in one call ────────────
        # `solve_dlm` is differentiable in its arguments: the surface potential β is a
        # root, so bisection brackets it on the values and one Newton step restores the
        # derivative. Seeding c_Cl with a dual therefore returns S_Cl and dS_Cl/dc from a
        # single evaluation, where the difference quotient needed two and a step size.
        n_csh_i = m.dlm.n_csh0   # C-S-H held fixed for now
        c_cl_seed = ForwardDiff.Dual{Nothing}(max(c_cl_new, 0.0), 1.0)
        _, S_Cl_d, S_Na, S_K, S_Ca = solve_dlm(
            c_cl_seed, max(c_na_new, 0.0), max(c_k_new, 0.0),
            max(c_ca_new, 0.0), max(c_oh_new, ε),
            n_csh_i; dlm=m.dlm, T_K=m.T_K,
        )
        m.S_Cl[i] = max(ForwardDiff.value(S_Cl_d), 0.0)
        m.S_Na[i] = max(ForwardDiff.value(S_Na), 0.0)
        m.S_K[i] = max(ForwardDiff.value(S_K), 0.0)
        m.S_Ca[i] = max(ForwardDiff.value(S_Ca), 0.0)

        # ── Effective K_d (DLM tangent + Friedel's salt tangent) ──────────────
        dS_Cl_dc = ForwardDiff.partials(S_Cl_d, 1)
        m.Kd_Cl[i] = dS_Cl_dc + 2.0 * dn_fs_dc
        m.Kd_Na[i] = m.S_Na[i] / max(c_na_new, ε)
        m.Kd_K[i] = m.S_K[i] / max(c_k_new, ε)
        m.Kd_Ca[i] = m.S_Ca[i] / max(c_ca_new, ε)

        # ── Free concentrations after Gibbs ───────────────────────────────────
        u[ICL4, i] = max(c_cl_new, 0.0)
        u[INA4, i] = max(c_na_new, 0.0)
        u[IK4, i] = max(c_k_new, 0.0)
        u[ICA4, i] = max(c_ca_new, 0.0)
    end
end

# ── SNIA solve ────────────────────────────────────────────────────────────────

"""
    run_chloride_ingress4(; N, t_end, n_save, verbose, dlm) -> (results, model)

SNIA chloride_ingress — Phase 4 : transport Fick + Gibbs + DLM C-S-H.

Returns:
- `results` : `[(t, u, phi, n_ch, n_ett, n_ms, n_fs, c_oh, S_Cl, S_Na, S_K, S_Ca), …]`
- `model`   : the final `ChlorideModel4` state
"""
function run_chloride_ingress4(;
    N=100,
    t_end=3.1536e7,   # 1 year [s]
    n_save=12,
    verbose=false,
    dlm=DLMParams(),
    kwargs...,
)
    cs, has_friedels = _init_chemistry4()

    # Transport diagnostic (same as Phase 3)
    let phi0 = 0.121
        oj = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27)
        tau = tortuosity(oj, phi0, 1)
        D_eff = 2.032e-9 * tau
        front_1yr = 2 * sqrt(D_eff / phi0 * t_end) * 1e3
        @info "Transport (phi=0.121)" tau = round(tau; sigdigits=3) D_eff_Cl = round(D_eff; sigdigits=3) front_Fick_mm = round(front_1yr; sigdigits=3)
    end

    m = ChlorideModel4(N + 1, cs, has_friedels; dlm=dlm, kwargs...)
    grid = simplexgrid(range(0.0, m.L; length=N + 1))

    _storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data) = PoroMechanics.flux!(f, u, edge, m, data)
    _bcond!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage=_storage!,
        flux=_flux!,
        bcondition=_bcond!,
        species=[ICL4, INA4, IK4, ICA4],
    )

    inival = unknowns(sys)
    inival[ICL4, :] .= m.c_cl_init
    inival[INA4, :] .= m.c_na_init
    inival[IK4, :] .= m.c_k_init
    inival[ICA4, :] .= m.c_ca_init
    inival[ICL4, 1] = m.c_cl_BC
    inival[INA4, 1] = m.c_na_BC
    inival[IK4, 1] = m.c_k_BC
    inival[ICA4, 1] = m.c_ca_BC

    ctrl = VoronoiFVM.SolverControl(;
        Δt=1.0,
        Δt_max=t_end / (4 * n_save),
        Δu_opt=0.5 * m.c_cl_BC,
        handle_exceptions=true,
        verbose=verbose,
    )

    tsave = range(0.0, t_end; length=n_save + 1)
    results = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    u_cur = copy(inival)

    for k in 2:lastindex(tsave)
        t0, t1 = tsave[k-1], tsave[k]
        @info "Transport seg $k" t_yr = round(t1 / 3.1536e7; digits=2)

        seg = solve(sys; inival=u_cur, times=[t0, t1], control=ctrl)
        u_cur = Matrix(seg[:, :, end])

        chemistry_step4!(m, u_cur, cs, has_friedels)

        u_cur[ICL4, 1] = m.c_cl_BC
        u_cur[INA4, 1] = m.c_na_BC
        u_cur[IK4, 1] = m.c_k_BC
        u_cur[ICA4, 1] = m.c_ca_BC
        m.c_oh_frozen[1] = 1.0
        # K_d at the exposed face (external, no C-S-H)
        m.Kd_Cl[1] = 0.0
        m.Kd_Na[1] = 0.0
        m.Kd_K[1] = 0.0
        m.Kd_Ca[1] = 0.0
        m.S_Cl[1] = 0.0
        m.S_Na[1] = 0.0
        m.S_K[1] = 0.0
        m.S_Ca[1] = 0.0

        any(isnan, u_cur) && @warn "NaN after chemistry, seg $k"

        push!(results, (
            t1,
            copy(u_cur),
            copy(m.phi),
            copy(m.n_ch),
            copy(m.n_ett),
            copy(m.n_ms),
            copy(m.n_fs),
            copy(m.c_oh_frozen),
            copy(m.S_Cl),
            copy(m.S_Na),
            copy(m.S_K),
            copy(m.S_Ca),
        ))

        @info "SNIA" seg = k - 1 φ_mean = round(mean(m.phi); sigdigits=4) n_FS0 = round(m.n_fs[2]; sigdigits=4) S_Cl0 = round(m.S_Cl[2]; sigdigits=4) β_node2 = round(let (_b, _, _, _, _) = solve_dlm(max(u_cur[ICL4, 2], 0.0), max(u_cur[INA4, 2], 0.0), max(u_cur[IK4, 2], 0.0), max(u_cur[ICA4, 2], 0.0), max(m.c_oh_frozen[2], 1e-15), m.dlm.n_csh0; dlm=m.dlm, T_K=m.T_K)
            _b
        end; sigdigits=3)
    end

    return results, m
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_reference_4(results, grid)

Comparison table: Cl⁻ profile vs the C++ reference, with DLM.
Balance in mol/m³_concrete and in g Cl / 100 g cement.
"""
function compare_reference_4(results, grid; dlm::DLMParams=DLMParams())
    # Conversion constants
    M_Cl = 35.453          # g/mol
    m_clinker = 350_000.0       # g/m³_concrete (350 kg/m³)
    fac = M_Cl * 100.0 / m_clinker   # (g Cl / 100g cement) per (mol Cl / m³_concrete)

    t = results[end][1]
    u = results[end][2]
    phi = results[end][3]
    n_fs = results[end][7]
    c_oh = results[end][8]
    S_Cl = results[end][9]

    x_m = grid[Coordinates][1, :]
    x_dm = x_m .* 10.0

    C_Cl_dm3 = u[ICL4, :] ./ 1000.0   # mol/dm³ (solution poreuse)
    c_oh_dm3 = c_oh ./ 1000.0

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    println("\nProfile at t = $(round(t/3.1536e7;digits=2)) year(s) — Phase 4 (Fick + Gibbs + C-S-H DLM)")
    println("="^115)
    @printf("%-8s  %-14s  %-10s  %-12s  %-10s  %-8s  %-10s  %-10s\n",
        "x [dm]", "C_Cl [mol/dm³]", "C++ ref", "c_OH [mol/dm³]", "φ [-]", "n_FS", "S_Cl_DLM", "β_DLM")
    println("-"^115)
    for (xr, cr) in zip(x_ref, c_ref)
        idx = argmin(abs.(x_dm .- xr))
        β_node = NaN
        try
            β_node, = solve_dlm(
                max(u[ICL4, idx], 0.0), max(u[INA4, idx], 0.0), max(u[IK4, idx], 0.0),
                max(u[ICA4, idx], 0.0), max(c_oh[idx], 1e-15),
                dlm.n_csh0; dlm=dlm, T_K=293.15)
        catch
        end
        @printf("%-8.4f  %-14.4f  %-10.4f  %-12.4f  %-10.4f  %-8.2f  %-10.2f  %-10.3f\n",
            xr, C_Cl_dm3[idx], cr, c_oh_dm3[idx], phi[idx], n_fs[idx], S_Cl[idx], β_node)
    end
    println("="^115)

    idx_front = findlast(C_Cl_dm3 .> 1e-3)
    x_front = idx_front !== nothing ? x_dm[idx_front] * 100.0 : NaN
    @printf("\nCl⁻ front ≈ %.1f mm  (C++ ref ≈ 7.5 mm)\n\n", x_front)

    # Bilan en g Cl / 100g ciment
    @printf("Bilan Cl en g/100g ciment :\n")
    @printf("  %-8s  %-12s  %-12s  %-12s  %-12s\n",
        "x [dm]", "Free", "DLM-adsorbed", "Friedel-bound", "Total")
    println("  " * "-"^60)
    for xr in [0.005, 0.010, 0.020, 0.030, 0.050]
        idx = argmin(abs.(x_dm .- xr))
        cl_libre = u[ICL4, idx] * phi[idx] * fac      # pore solution → m³_concrete → g/100g
        cl_dlm = S_Cl[idx] * fac
        cl_friedel = 2.0 * n_fs[idx] * fac
        cl_total = cl_libre + cl_dlm + cl_friedel
        @printf("  %-8.3f  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n",
            xr, cl_libre, cl_dlm, cl_friedel, cl_total)
    end
end

"""
    plot_chloride_ingress4(results, grid; n_curves, save_path)

Six panneaux :
  p1 – Cl split (free / DLM-adsorbed / Friedel-bound / total) in g/100g cement at the final time.
  p2 – Evolution of total Cl in g/100g cement (several time steps).
  p3 – Porosity φ.
  p4 – AFm: Friedel's salt (FS) + monosulphate (MS).
  p5 – c_OH.
  p6 – DLM surface potential β.
"""
function plot_chloride_ingress4(results, grid; n_curves=4, save_path=nothing, dlm::DLMParams=DLMParams())
    # Conversion constants: mol Cl / m³_concrete → g Cl / 100g cement
    M_Cl = 35.453
    m_clinker = 350_000.0          # g/m³_concrete
    fac = M_Cl * 100.0 / m_clinker

    x_dm = grid[Coordinates][1, :] .* 10.0
    t_yr = 3.1536e7
    # Free Cl (mol/dm³_water) from the C++ reference solution, column c_cl_l (13)
    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]
    # Total Cl (mol/dm³_concrete) from the C++ reference solution, column n_Cl (50)
    # n_Cl = free×phi + adsorbed + 2×Friedel (checked at every node)
    x_ref_total = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
        0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075]
    n_ref_total = [0.16614, 4.16643, 4.03841, 1.55713, 0.08571, 0.08290,
        0.07986, 0.07650, 0.07265, 0.06798, 0.06189, 0.05317,
        0.03986, 0.02237, 0.00885, 0.00301]  # mol/dm³_concrete
    # 1 mol/dm³ = 1000 mol/m³ → ×1000×fac donne g/100g ciment
    c_ref_total_g100 = n_ref_total .* (1000.0 * fac)
    n_t = length(results)
    idxs = unique(clamp.(round.(Int, range(1, n_t; length=n_curves)), 1, n_t))
    pal = [:steelblue, :darkorange, :crimson, :forestgreen, :purple, :teal]

    # Final state
    t_f = results[end][1]
    u_f = results[end][2]
    phi_f = results[end][3]
    n_fs_f = results[end][7]
    c_oh_f = results[end][8]
    S_Cl_f = results[end][9]
    t_lbl = "t = $(round(t_f/t_yr; digits=2)) yr"

    # Cl forms at the final time [g/100g cement]
    C_libre = u_f[ICL4, :] .* phi_f .* fac    # solution poreuse
    C_dlm = S_Cl_f .* fac                    # adsorbed on C-S-H
    C_friedel = 2.0 .* n_fs_f .* fac             # chemically bound (Friedel's salt)
    C_total_f = C_libre .+ C_dlm .+ C_friedel

    # C++ reference: free Cl → g/100g using the simulated phi
    c_ref_g100 = [c_ref[i] * 1000.0 * phi_f[argmin(abs.(x_dm .- x_ref[i]))] * fac
                  for i in eachindex(x_ref)]

    # ── p1: Cl split at the final time ────────────────────────────────────────
    p1 = plot(; xlabel="Profondeur x [dm]",
        ylabel="Cl [g / 100 g ciment]",
        title="Phase 4 — Cl forms ($t_lbl)",
        legend=:topright)
    plot!(p1, x_dm, C_libre; lw=2, color=:steelblue, label="Cl libre (solution poreuse)")
    plot!(p1, x_dm, C_dlm; lw=2, color=:darkorange, label="adsorbed Cl (C-S-H DLM)")
    plot!(p1, x_dm, C_friedel; lw=2, color=:crimson, label="bound Cl (Friedel's salt)")
    plot!(p1, x_dm, C_total_f; lw=3, color=:black, ls=:dash, label="Cl total")
    scatter!(p1, x_ref, c_ref_g100; ms=5, color=:black, marker=:circle, label="C++ ref, free")
    #scatter!(p1, x_ref_total, c_ref_total_g100; ms=5, color=:darkgreen, marker=:diamond, label="C++ ref, total")

    # ── p2: evolution of total Cl ─────────────────────────────────────────────
    p2 = plot(; xlabel="Profondeur x [dm]",
        ylabel="Cl total [g / 100 g ciment]",
        title="Phase 4 — total Cl (evolution)",
        legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k = results[ti][1]
        u_k = results[ti][2]
        phi_k = results[ti][3]
        n_fs_k = results[ti][7]
        S_Cl_k = results[ti][9]
        C_tot_k = (u_k[ICL4, :] .* phi_k .+ S_Cl_k .+ 2.0 .* n_fs_k) .* fac
        plot!(p2, x_dm, C_tot_k; lw=2, color=pal[mod1(k, end)],
            label="t = $(round(t_k/t_yr; digits=2)) yr")
    end
    scatter!(p2, x_ref, c_ref_g100; ms=5, color=:black, marker=:circle, label="C++ ref, free (1 yr)")
    scatter!(p2, x_ref_total, c_ref_total_g100; ms=5, color=:darkgreen, marker=:diamond, label="C++ ref, total (1 yr)")

    # ── p3: porosity ──────────────────────────────────────────────────────────
    p3 = plot(; xlabel="x [dm]", ylabel="φ [-]", title="Porosity", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k = results[ti][1]
        phi_k = results[ti][3]
        plot!(p3, x_dm, phi_k; lw=2, color=pal[mod1(k, end)],
            label="t = $(round(t_k/t_yr; digits=2)) yr")
    end
    hline!(p3, [0.121]; lw=1, ls=:dot, color=:gray50, label="φ₀")

    # ── p4: AFm (Friedel's salt + monosulphate) ───────────────────────────────
    p4 = plot(; xlabel="x [dm]", ylabel="[mol/m³]", title="AFm (FS + MS)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k = results[ti][1]
        n_fs_k = results[ti][7]
        n_ms_k = results[ti][6]
        lbl = "t = $(round(t_k/t_yr; digits=2)) yr"
        col = pal[mod1(k, end)]
        plot!(p4, x_dm, n_fs_k; lw=2, color=col, label="FS $lbl")
        plot!(p4, x_dm, n_ms_k; lw=1, ls=:dash, color=col, label="MS $lbl")
    end

    # ── p5 : c_OH ─────────────────────────────────────────────────────────────
    p5 = plot(; xlabel="x [dm]", ylabel="c_OH [mol/dm³]", title="OH⁻ ($t_lbl)", legend=false)
    plot!(p5, x_dm, c_oh_f ./ 1000; lw=2, color=:darkorange)

    # ── p6 : Potentiel DLM β ──────────────────────────────────────────────────
    p6 = plot(; xlabel="x [dm]", ylabel="β = Fψ/RT [-]", title="Potentiel DLM ($t_lbl)", legend=false)
    β_profile = [
        begin
            β_v, = solve_dlm(max(u_f[ICL4, j], 0.0), max(u_f[INA4, j], 0.0), max(u_f[IK4, j], 0.0),
                max(u_f[ICA4, j], 0.0), max(c_oh_f[j], 1e-15),
                dlm.n_csh0; dlm=dlm, T_K=293.15)
            β_v
        end for j in eachindex(x_dm)
    ]
    plot!(p6, x_dm, β_profile; lw=2, color=:mediumpurple)
    hline!(p6, [0.0]; lw=1, ls=:dot, color=:gray60)

    fig = plot(p1, p2, p3, p4, p5, p6; layout=(2, 3), size=(1800, 700))
    save_path !== nothing && savefig(fig, save_path)
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "chloride_ingress — Phase 4 : Fick + Gibbs + DLM C-S-H (Tran 2018)"

    # Parameters tuned on the C++ chloride_ingress version:
    #   n_csh0 = 635 mol/m³  ← InitialContent_csh = 0.635 mol/dm³
    #   n_ms0  = 3000 mol/m³ ← unlimited Al in C++ (n_c3a = -n_friedelsalt)
    dlm = DLMParams(n_csh0=635.0)

    results, m_fin = run_chloride_ingress4(; N=100, t_end=3.1536e7, n_save=12, dlm=dlm, n_ms0=3000.0)

    grid_ref = simplexgrid(range(0.0, 0.05; length=101))
    compare_reference_4(results, grid_ref; dlm=dlm)

    try
        using Plots
        p = plot_chloride_ingress4(results, grid_ref; dlm=dlm)
        display(p)
    catch e
        @warn "Plots.jl not available" exception = e
    end
end
