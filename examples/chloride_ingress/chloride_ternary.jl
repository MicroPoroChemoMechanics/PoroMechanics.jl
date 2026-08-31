# chloride_ternary.jl — generic infrastructure: ≡SiOCaCl ternary reactive transport
# in cementitious materials exposed to an aggressive solution.
#
# Contains: ternary DLM (Thermoddem 2023 + Yoshida 2021), mineral kinetics (CSHQ /
# AFm-SS / MSH / LDH-OH), multi-ionic Fick transport (VoronoiFVM), SNIA chemical step
# (operator splitting transport ↔ chimie cemdata18), visualisation.
#
# This file has no `if abspath(PROGRAM_FILE)` block.
# It is included by a material-specific script (e.g. m100_ternary.jl) that defines the
# material structs (ClinkerComposition, CementMaterial, ExposureConditions) and the
# CementTernaryModel(N_nodes, cs_hyd; ...) constructor.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver
using Plots

include("physdata.jl")

# ── Transport species indices ─────────────────────────────────────────────────
const ICL_T = 1
const INA_T = 2
const IK_T = 3
const ICA_T = 4
const IMG_T = 5
const ISO4_T = 6
const ISI_T = 7
const IAL_T = 8

const SI_AQ_SPECIES_T = [
    ("SiO2@", 1), ("HSiO3-", 1), ("SiO3-2", 1),
    ("Ca(HSiO3)+", 1), ("Mg(HSiO3)+", 1), ("CaSiO3@", 1),
    ("AlSiO5-3", 1), ("Si4O10-4", 4),
]

# Aqueous Al species in cemdata18 — full confirmed list:
#   Al+3, AlOH+2, AlO+  (↔ Al(OH)₂⁺), AlO2H@ (↔ Al(OH)₃ aq), AlO2- (↔ Al(OH)₄⁻, dominant pH>12)
#   Al(SO4)2-, Al(SO4)+  (sulphate complexes, non-zero in seawater SO₄²⁻=27.6 mol/m³)
#   AlSiO5-3  (carries 1 Al AND 1 Si — also listed in SI_AQ_SPECIES_T, counted independently)
# Note: moles() takes the species NAME ("Al(SO4)+"), not the chemical symbol ("AlSO4+")
const AL_AQ_SPECIES_T = [
    ("Al+3", 1), ("AlOH+2", 1), ("AlO+", 1), ("AlO2H@", 1), ("AlO2-", 1),
    ("Al(SO4)2-", 1), ("Al(SO4)+", 1), ("AlSiO5-3", 1),
]

const V_REV_T = 1.0e-3

const CSHQ_EM_T = ("CSHQ-TobD", "CSHQ-TobH", "CSHQ-JenH", "CSHQ-JenD")

# C-A-S-H (CNASH_ss / CSH3T, Myers et al. 2014 / cemdata18)
const CASH_EM_T = ("CSH3T-TobH", "CSH3T-T5C", "CSH3T-T2C")

# Ca/Si ratio of each CSH3T end-member (Si only, for the Yoshida 2021 interpolation)
# Values estimated from the cemdata18 formulas — TODO: check on the first run
const XCAS_CASH_TOBH = 0.922    # Ca₀.₈₃ / Si₀.₉₀
const XCAS_CASH_T5C  = 1.667    # Ca₁.₂₅ / Si₀.₇₅
const XCAS_CASH_T2C  = 1.746    # Ca₁.₁₇ / Si₀.₆₇

# Dissolved Al stoichiometry released per mol of dissolved CSH3T (mol Al(OH)₄⁻ / mol phase)
# TODO: check against the cemdata18 formulas
const AL_STOICH_CASH_TOBH = 0.10
const AL_STOICH_CASH_T5C  = 0.25
const AL_STOICH_CASH_T2C  = 0.33

const MSH_EM_T = ("M075SH", "M15SH")

const LDH_EM_T = ("M4A-OH-LDH", "M6A-OH-LDH", "M8A-OH-LDH")

# End-members of the AFm solid solution (anion exchange SO₄²⁻ ↔ 2Cl⁻ in the interlayer)
const AFM_EM_T = ("monosulphate12", "C4AClH10")

const GYPS_NAME_T = "Gp"   # Gypse secondaire CaSO₄·2H₂O (cemdata18-thermofun)

# Nernst-Planck (diffusion potential, zero current):
# ionic charges for Cl⁻ Na⁺ K⁺ Ca²⁺ Mg²⁺ SO₄²⁻ Si (neutral)
const Z_NP_T = (-1.0, 1.0, 1.0, 2.0, 2.0, -2.0, 0.0)

const EXCLUDE_NONCSHQ_T = [
    # CSH3T removed — handled separately through ss_cash in _init_chemistry_ternary
    "ECSH1-TobCa", "ECSH1-SH", "ECSH1-NaSH", "ECSH1-KSH", "ECSH1-SrSH",
    "ECSH2-TobCa", "ECSH2-JenCa", "ECSH2-NaSH", "ECSH2-KSH",
    "ECSH2-SrSH", "ECSH2-SrSH(ACW)", "Tob-I", "Tob-II", "Jennite",
    "TobH-CNASHss", "T5C-CNASHss", "T2C-CNASHss",
]

# ════════════════════════════════════════════════════════════════════════════════
# Generic parameters — ternary DLM + kinetics
# (TransportModel is defined in the calling material script, e.g. m100_ternary.jl)
# ════════════════════════════════════════════════════════════════════════════════

# ── DLM parameters — ternary complex + Yoshida 2021 ───────────────────────────
#
# NOTE — this belongs in ChemistryLab.jl.
# Surface complexation is chemistry, not transport. It lives here only because
# ChemistryLab.jl does not expose it yet; when it does, delete this and call it.
# Three near-identical copies of the model currently exist in this directory
# (run_4.jl, tran2018.jl, chloride_ternary.jl), which is the argument for moving it.


"""
DLM parameters for the ternary complex ≡SiOCaCl (Thermoddem 2023) and the
Yoshida et al. 2021 site densities.

Equilibrium constants (concentrations in mol/m³, units m³/mol):
  Ka1     : ≡SiOH → ≡SiO⁻ + H⁺            pKa = 12.7 → Ka1 = 2.0×10⁻¹⁰ mol/m³
  K_Ca    : ≡SiO⁻ + Ca²⁺ → ≡SiOCa⁺       log K_TR = 9.4  → K_Ca = 2.0 m³/mol
  K_Mg    : ≡SiO⁻ + Mg²⁺ → ≡SiOMg⁺       (seawater extension)
  K_CaCl  : ≡SiOH + Ca²⁺ + Cl⁻ → ≡SiOCaCl + H⁺
            log K_TR = 9.8 (Thermoddem)  → K_CaCl = 10^(−12.8) ≈ 1.585×10⁻¹³ m³/mol
  K_Na_Tob/Jen : ≡SiO⁻ + Na⁺ → ≡SiONa   (linear interpolation in x_CaS)

Site densities (Yoshida et al. 2021, Figure 5):
  Gamma_max_Tob : 4.3 nm⁻² → 7.14×10⁻⁶ mol/m²   (tobermorite, x_CaS ≈ 0.83)
  Gamma_max_Jen : 7.0 nm⁻² → 1.162×10⁻⁵ mol/m²  (jennite,     x_CaS ≈ 1.67)
  Linear interpolation between the two as a function of x_CaS.

BET specific surface area: 500 m²/g (Soive 2017), unchanged from tran2018.jl.
"""
Base.@kwdef struct DLMTernaryParams
    Ka1::Float64 = 2.0e-10     # [mol/m³]  pKa = 12.7
    K_Ca::Float64 = 2.0          # [m³/mol]  log K_TR = 9.4
    K_Mg::Float64 = 0.10         # [m³/mol]  seawater extension
    K_CaCl::Float64 = 1.585e-13   # [m³/mol]  ternaire, log K_TR = 9.8
    K_Na_Tob::Float64 = 1.106e-3    # [m³/mol]  x_CaS = 0.83
    K_Na_Jen::Float64 = 9.0e-5      # [m³/mol]  x_CaS = 1.67
    Gamma_max_Tob::Float64 = 7.14e-6   # [mol/m²]  Yoshida 2021 tobermorite (4.3 nm⁻²)
    Gamma_max_Jen::Float64 = 1.162e-5  # [mol/m²]  Yoshida 2021 jennite     (7.0 nm⁻²)
    a_s::Float64 = 85000.0     # [m²/mol]  500 m²/g × ~170 g/mol (Soive 2017)
    n_csh0::Float64 = 0.0         # [mol/m³]  0 = taken dynamically from CSHQ
    eps_r::Float64 = 78.5        # [-]
    Kw_SI::Float64 = 6.76e-9     # [mol²/m⁶] produit ionique eau 20 °C
end

## `KineticParams` now lives in `physdata.jl`, included above. It is database-derived data
## like the equilibrium constants and molar volumes already there, `m100_ternary.jl` reads
## it too, and keeping the definition here made this file the de facto home of a shared
## table.


# Linear interpolation of Gamma_max between Tob (x_CaS=0.83) and Jen (x_CaS=1.67)
function _gamma_max_dlm(dlm::DLMTernaryParams, x_cas::Float64)
    xT, xJ = 0.83, 1.67
    t = clamp((x_cas - xT) / (xJ - xT), 0.0, 1.0)
    return dlm.Gamma_max_Tob + t * (dlm.Gamma_max_Jen - dlm.Gamma_max_Tob)
end

function _k_na_dlm_t(dlm::DLMTernaryParams, x_cas::Float64)
    xT, xJ = 0.83, 1.67
    x_c = clamp(x_cas, xT, xJ)
    return dlm.K_Na_Jen + (dlm.K_Na_Tob - dlm.K_Na_Jen) * (xJ - x_c) / (xJ - xT)
end

"""
    solve_dlm_ternary(c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm, T_K)

DLM equilibrium with the ternary complex ≡SiOCaCl (Thermoddem 2023) and the
sites Yoshida 2021.

The ≡SiOCaCl complex is neutral → no Boltzmann correction, no contribution to
the surface charge.

Returns `(β, S_Cl, S_Na, S_K, S_Ca, S_Mg)` in mol/m³_concrete.
"""
function solve_dlm_ternary(
    c_Cl::Float64, c_Na::Float64, c_K::Float64, c_Ca::Float64,
    c_Mg::Float64, c_OH::Float64, n_csh::Float64, x_cas::Float64;
    dlm::DLMTernaryParams,
    T_K::Float64=293.15,
)
    n_csh ≤ 0.0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    c_H = dlm.Kw_SI / max(c_OH, 1.0e-20)
    I = max(0.5 * (c_Cl + c_Na + c_K + 4.0 * c_Ca + 4.0 * c_Mg + c_OH + c_H), 1.0)

    Ka1 = dlm.Ka1
    KCa = dlm.K_Ca
    KMg = dlm.K_Mg
    KCaCl = dlm.K_CaCl
    KNa = _k_na_dlm_t(dlm, x_cas)
    KK = KNa

    Gamma_max = _gamma_max_dlm(dlm, x_cas)

    # ── Coverage fractions relative to ≡SiOH ──────────────────────────────────
    # ≡SiO⁻     (charge −1) : exp(+β)
    # ≡SiOCa⁺   (charge +1) : exp(−β)
    # ≡SiOMg⁺   (charge +1) : exp(−β)
    # ≡SiOCaCl  (charge  0) : no electrostatic correction
    # ≡SiONa/K  (neutral)   : no correction

    @inline A(β) = (1.0
                    + Ka1 * exp(β) / c_H                          # ≡SiO⁻
                    + KCa * Ka1 * c_Ca * exp(-β) / c_H           # ≡SiOCa⁺
                    + KMg * Ka1 * c_Mg * exp(-β) / c_H           # ≡SiOMg⁺
                    + KCaCl * c_Ca * c_Cl / c_H                   # ≡SiOCaCl (ternaire, neutre)
                    + (KNa * c_Na + KK * c_K) * Ka1 / c_H)       # ≡SiONa, ≡SiOK

    # Surface charge σ₀ = F·Γ_max·B/A  (≡SiOCaCl neutral → absent from B)
    @inline B(β) = (KCa * Ka1 * c_Ca * exp(-β) / c_H
                    +
                    KMg * Ka1 * c_Mg * exp(-β) / c_H
                    -
                    Ka1 * exp(β) / c_H)

    F_val = 96485.0
    R_val = 8.314
    σ_cap = sqrt(8.0 * 8.854e-12 * dlm.eps_r * R_val * T_K * I)

    f(β) = F_val * Gamma_max * B(β) / A(β) - σ_cap * sinh(β / 2.0)

    β_lo, β_hi = -10.0, 10.0
    f_lo = f(β_lo)
    f_hi = f(β_hi)
    β_sol = 0.0
    if f_lo * f_hi < 0.0
        for _ in 1:64
            β_mid = 0.5 * (β_lo + β_hi)
            f_mid = f(β_mid)
            if f_mid * f_lo < 0.0
                β_hi = β_mid
            else
                β_lo = β_mid
                f_lo = f_mid
            end
            abs(β_hi - β_lo) < 1.0e-9 && break
        end
        β_sol = 0.5 * (β_lo + β_hi)
    else
        β_sol = abs(f_lo) < abs(f_hi) ? β_lo : β_hi
    end

    β = β_sol
    X = Gamma_max / A(β)   # θ_SiOH × Γ_max [mol/m²]

    theta_OCaCl = KCaCl * X * c_Ca * c_Cl / c_H          # ≡SiOCaCl
    theta_OCa = KCa * Ka1 * X * c_Ca * exp(-β) / c_H  # ≡SiOCa⁺
    theta_OMg = KMg * Ka1 * X * c_Mg * exp(-β) / c_H  # ≡SiOMg⁺
    theta_ONa = KNa * Ka1 * X * c_Na / c_H             # ≡SiONa
    theta_OK = KK * Ka1 * X * c_K / c_H             # ≡SiOK

    fac = dlm.a_s * n_csh
    S_Cl = theta_OCaCl * fac
    S_Na = theta_ONa * fac
    S_K = theta_OK * fac
    S_Ca = theta_OCa * fac
    S_Mg = theta_OMg * fac

    return β, S_Cl, S_Na, S_K, S_Ca, S_Mg
end

# ── Model ─────────────────────────────────────────────────────────────────────

"""
Generic state model for reactive transport in cementitious material exposed to an aggressive
solution — ternary surface complex ≡SiOCaCl + Yoshida 2021 site densities.
The external constructor (in the specific material script) initialises the hydrated state
and concentration fields.
"""
mutable struct CementTernaryModel <: AbstractPoroModel
    L::Float64
    phi::Vector{Float64}
    n_ch::Vector{Float64}
    n_ett::Vector{Float64}
    n_ms::Vector{Float64}
    n_fs::Vector{Float64}
    n_brc::Vector{Float64}
    n_csh_tobh::Vector{Float64}
    n_csh_tobd::Vector{Float64}
    n_csh_jenh::Vector{Float64}
    n_csh_jend::Vector{Float64}
    n_cash_tobh::Vector{Float64}
    n_cash_t5c::Vector{Float64}
    n_cash_t2c::Vector{Float64}
    n_msh_08::Vector{Float64}
    n_msh_13::Vector{Float64}
    n_ldh_m4::Vector{Float64}
    n_ldh_m6::Vector{Float64}
    n_ldh_m8::Vector{Float64}
    n_gyp::Vector{Float64}
    c_oh_frozen::Vector{Float64}
    Kd_Cl::Vector{Float64}
    Kd_Na::Vector{Float64}
    Kd_K::Vector{Float64}
    Kd_Ca::Vector{Float64}
    Kd_Mg::Vector{Float64}
    Kd_SO4::Vector{Float64}
    Kd_Al::Vector{Float64}
    # DLM: Cl⁻ adsorbed through ≡SiOCaCl (ternary)
    S_Cl_dlm::Vector{Float64}
    S_Na_dlm::Vector{Float64}
    S_K_dlm::Vector{Float64}
    S_Ca_dlm::Vector{Float64}
    S_Mg_dlm::Vector{Float64}
    dlm::DLMTernaryParams
    kin::KineticParams
    diff::IonicDiffusivities
    mat::CementMaterial
    env::ExposureConditions
    c_cl_init::Float64
    c_na_init::Float64
    c_k_init::Float64
    c_ca_init::Float64
    c_mg_init::Float64
    c_so4_init::Float64
    c_si_init::Float64
    c_al_init::Float64
end

# ── cemdata18 chemical system initialisation ──────────────────────────────────

function _init_chemistry_ternary()
    data_path = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    isfile(data_path) || error("cemdata18 introuvable : $data_path")
    @info "Loading cemdata18…"
    substances = build_species(data_path)
    dict_sp = Dict(symbol(s) => s for s in substances)

    seeds = split("Portlandite ettringite monosulphate12 C3S C2S H2O@ Ca+2 OH- Cl- Na+ K+ H+ Mg+2 Al+3 SO4-2 SiO2@")
    aq_species = speciation(substances, seeds;
        aggregate_state=[AS_AQUEOUS],
        exclude_species=EXCLUDE_NONCSHQ_T,
    )

    cshq_found = filter(n -> haskey(dict_sp, n), collect(CSHQ_EM_T))
    length(cshq_found) == 4 || @warn "CSHQ: only $(length(cshq_found))/4 end-members found"
    ss_cshq = SolidSolutionPhase("CSHQ", [dict_sp[em] for em in cshq_found]; model=IdealSolidSolutionModel())

    cash_found = filter(n -> haskey(dict_sp, n), collect(CASH_EM_T))
    has_cash = length(cash_found) == 3
    if has_cash
        ss_cash = SolidSolutionPhase("CASH3T", [dict_sp[em] for em in cash_found]; model=IdealSolidSolutionModel())
        @info "C-A-S-H (CSH3T) enabled" n_em = length(cash_found)
    else
        @warn "CSH3T missing from cemdata18 ($(length(cash_found))/3 end-members) — C-A-S-H disabled"
    end

    msh_found = filter(n -> haskey(dict_sp, n), collect(MSH_EM_T))
    has_msh = !isempty(msh_found)
    if has_msh
        ss_msh = SolidSolutionPhase("MSH", [dict_sp[em] for em in msh_found]; model=IdealSolidSolutionModel())
    else
        @warn "M-S-H missing from cemdata18"
    end

    ldh_found = filter(n -> haskey(dict_sp, n), collect(LDH_EM_T))
    has_ldh = !isempty(ldh_found)
    if has_ldh
        ss_ldh = SolidSolutionPhase("LDH", [dict_sp[em] for em in ldh_found]; model=IdealSolidSolutionModel())
    else
        @warn "LDH-OH missing from cemdata18"
    end

    # AFm solid solution (anion substitution SO₄²⁻ ↔ 2Cl⁻ in the interlayer).
    # Lets Friedel's salt form as an end-member of a mixture even when the pure
    # C4AClH10 phase would be undersaturated (IP < 1).
    afm_found = filter(n -> haskey(dict_sp, n), collect(AFM_EM_T))
    has_afm_ss = length(afm_found) == length(AFM_EM_T)
    if has_afm_ss
        ss_afm = SolidSolutionPhase("AFmSS", [dict_sp[n] for n in afm_found]; model=IdealSolidSolutionModel())
        @info "AFm-SS activee (monosulfo <-> Friedel)"
    else
        @warn "AFm-SS disabled: end-members missing from cemdata18"
    end

    solid_names_hyd = String["C3S", "C2S", "C3A", "C4AF", "Gp", cshq_found...,
        (has_cash ? cash_found : String[])...,
        "Portlandite", "ettringite", "monosulphate12"]
    solid_hyd = [dict_sp[n] for n in solid_names_hyd if haskey(dict_sp, n)]
    ss_hyd = SolidSolutionPhase[ss_cshq]
    has_cash && push!(ss_hyd, ss_cash)
    cs_hyd = ChemicalSystem(vcat(collect(aq_species), solid_hyd), CEMDATA_PRIMARIES; solid_solutions=ss_hyd)

    has_friedels = haskey(dict_sp, "C4AClH10")
    has_brucite = haskey(dict_sp, "Brc")       # cemdata18 name: "Brc" (not "Brucite")
    has_friedels || @warn "C4AClH10 absent"
    has_brucite || @warn "Brc (Brucite) missing from cemdata18"

    has_gyp = haskey(dict_sp, GYPS_NAME_T)
    has_gyp || @warn "$(GYPS_NAME_T) (gypsum CaSO₄·2H₂O) missing from cemdata18"

    # The end-members of a SolidSolutionPhase must also appear in the species vector
    # passed to ChemicalSystem, alongside their SolidSolutionPhase.
    solid_names_tr = String[cshq_found...,
        (has_cash ? cash_found : String[])...,
        "Portlandite", "ettringite", "monosulphate12",
        "C4AClH10", "Brc",
        (has_gyp ? [GYPS_NAME_T] : String[])...,
        (has_msh ? msh_found : String[])...,
        (has_ldh ? ldh_found : String[])...]
    solid_tr = [dict_sp[n] for n in solid_names_tr if haskey(dict_sp, n)]
    ss_tr = SolidSolutionPhase[ss_cshq]
    has_cash && push!(ss_tr, ss_cash)
    has_msh && push!(ss_tr, ss_msh)
    has_ldh && push!(ss_tr, ss_ldh)
    has_afm_ss && push!(ss_tr, ss_afm)
    cs_tr = ChemicalSystem(vcat(collect(aq_species), solid_tr), CEMDATA_PRIMARIES; solid_solutions=ss_tr)

    return cs_hyd, cs_tr, has_afm_ss, has_friedels, has_brucite, has_msh, has_ldh, has_gyp, has_cash
end

PoroMechanics.nspecies(::CementTernaryModel) = 8
PoroMechanics.species_names(::CementTernaryModel) = [:c_Cl, :c_Na, :c_K, :c_Ca, :c_Mg, :c_SO4, :c_Si, :c_Al]

# ── Oh-Jang tortuosity ────────────────────────────────────────────────────────

"""
    _tortuosity_t(phi, m::CementTernaryModel)

Oh-Jang tortuosity of the cement paste, from the package constitutive layer.
Saturated medium: S_l = 1, so the saturation factor is one.
"""
function _tortuosity_t(phi, m::CementTernaryModel)
    tr = m.mat.transport
    oj = OhJang(; phi_c = tr.phi_c, n = tr.n_OJ, ds = tr.ds_OJ, tau_agg = tr.tau_agg)
    return tortuosity(oj, phi, 1)
end

@inline function _node_idx_t(x::Float64, m::CementTernaryModel)
    N = length(m.phi) - 1
    return clamp(round(Int, x / (m.L / N)) + 1, 1, N + 1)
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

function PoroMechanics.storage!(f, u, node, m::CementTernaryModel, ::Any)
    i = _node_idx_t(node.coord[1], m)
    phi = m.phi[i]
    f[ICL_T] = (phi + m.Kd_Cl[i]) * u[ICL_T]
    f[INA_T] = (phi + m.Kd_Na[i]) * u[INA_T]
    f[IK_T] = (phi + m.Kd_K[i]) * u[IK_T]
    f[ICA_T] = (phi + m.Kd_Ca[i]) * u[ICA_T]
    f[IMG_T] = (phi + m.Kd_Mg[i]) * u[IMG_T]
    f[ISO4_T] = (phi + m.Kd_SO4[i]) * u[ISO4_T]
    f[ISI_T] = phi * u[ISI_T]
    f[IAL_T] = phi * u[IAL_T]
end

function PoroMechanics.flux!(f, u, edge, m::CementTernaryModel, ::Any)
    x_mid = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    i = _node_idx_t(x_mid, m)
    phi_i = (i < length(m.phi)) ? (m.phi[i] + m.phi[i+1]) / 2 : m.phi[i]
    tau = _tortuosity_t(phi_i, m)

    f[ICL_T]  = m.diff.D_Cl  * tau * (u[ICL_T,  1] - u[ICL_T,  2])
    f[INA_T]  = m.diff.D_Na  * tau * (u[INA_T,  1] - u[INA_T,  2])
    f[IK_T]   = m.diff.D_K   * tau * (u[IK_T,   1] - u[IK_T,   2])
    f[ICA_T]  = m.diff.D_Ca  * tau * (u[ICA_T,  1] - u[ICA_T,  2])
    f[IMG_T]  = m.diff.D_Mg  * tau * (u[IMG_T,  1] - u[IMG_T,  2])
    f[ISO4_T] = m.diff.D_SO4 * tau * (u[ISO4_T, 1] - u[ISO4_T, 2])
    f[ISI_T]  = m.diff.D_Si  * tau * (u[ISI_T,  1] - u[ISI_T,  2])
    f[IAL_T]  = m.diff.D_Al  * tau * (u[IAL_T,  1] - u[IAL_T,  2])
end

function PoroMechanics.bcondition!(f, u, bnode, m::CementTernaryModel, ::Any)
    boundary_dirichlet!(f, u, bnode; species=ICL_T,  region=1, value=m.env.c_Cl)
    boundary_dirichlet!(f, u, bnode; species=INA_T,  region=1, value=m.env.c_Na)
    boundary_dirichlet!(f, u, bnode; species=IK_T,   region=1, value=m.env.c_K)
    boundary_dirichlet!(f, u, bnode; species=ICA_T,  region=1, value=m.env.c_Ca)
    boundary_dirichlet!(f, u, bnode; species=IMG_T,  region=1, value=m.env.c_Mg)
    boundary_dirichlet!(f, u, bnode; species=ISO4_T, region=1, value=m.env.c_SO4)
    boundary_dirichlet!(f, u, bnode; species=ISI_T,  region=1, value=m.env.c_Si)
    boundary_dirichlet!(f, u, bnode; species=IAL_T,  region=1, value=m.env.c_Al)
end

# ── SNIA chemical step ────────────────────────────────────────────────────────

function chemistry_step_ternary!(
    m::CementTernaryModel, u::Matrix, cs,
    has_afm_ss::Bool, has_friedels::Bool, has_brucite::Bool, has_msh::Bool, has_ldh::Bool,
    has_gyp::Bool, has_cash::Bool,
    Δt_snia::Float64=Inf,
)
    N = size(u, 2)
    T_q = m.env.T_K * us"K"

    vm = MolarVolumes()
    ε = 1.0e-15
    skip_cl  = 1e-3 * m.env.c_Cl
    skip_mg  = 1e-3 * m.env.c_Mg
    skip_so4 = 1e-3 * m.env.c_SO4

    Threads.@threads for i in 2:N
        phi_i = m.phi[i]
        C_Cl = max(u[ICL_T, i], 0.0)
        C_Na = max(u[INA_T, i], 0.0)
        C_K = max(u[IK_T, i], 0.0)
        C_Ca = max(u[ICA_T, i], 0.0)
        C_Mg = max(u[IMG_T, i], 0.0)
        C_SO4 = max(u[ISO4_T, i], 0.0)

        if C_Cl < skip_cl && C_Mg < skip_mg && C_SO4 < skip_so4 &&
           m.n_fs[i] < 1e-8 && m.n_brc[i] < 1e-8
            continue
        end

        C_Si = max(u[ISI_T, i], 0.0)
        C_Al = max(u[IAL_T, i], 0.0)
        n_cl = C_Cl * phi_i * V_REV_T
        n_na = C_Na * phi_i * V_REV_T
        n_k = C_K * phi_i * V_REV_T
        n_ca = C_Ca * phi_i * V_REV_T
        n_mg = C_Mg * phi_i * V_REV_T
        n_so4 = C_SO4 * phi_i * V_REV_T
        n_si = C_Si * phi_i * V_REV_T
        n_al = C_Al * phi_i * V_REV_T
        # Al(OH)₄⁻ carries a −1 charge, like Cl⁻
        n_oh = max(n_na + n_k + 2.0 * n_ca + 2.0 * n_mg - n_cl - 2.0 * n_so4 - n_al, 1.0e-20)
        n_water = phi_i * V_REV_T * 55_500.0

        state = ChemicalState(cs; T=T_q)
        set_quantity!(state, "H2O@", n_water * us"mol")
        set_quantity!(state, "OH-", n_oh * us"mol")
        set_quantity!(state, "Cl-", n_cl * us"mol")
        set_quantity!(state, "Na+", n_na * us"mol")
        set_quantity!(state, "K+", n_k * us"mol")
        set_quantity!(state, "Ca+2", n_ca * us"mol")
        set_quantity!(state, "Mg+2", n_mg * us"mol")
        set_quantity!(state, "SO4-2", n_so4 * us"mol")
        set_quantity!(state, "SiO2@", n_si * us"mol")
        set_quantity!(state, "Al+3", n_al * us"mol")
        set_quantity!(state, "Portlandite", m.n_ch[i] * V_REV_T * us"mol")
        set_quantity!(state, "ettringite", m.n_ett[i] * V_REV_T * us"mol")
        set_quantity!(state, "monosulphate12", m.n_ms[i] * V_REV_T * us"mol")
        set_quantity!(state, "CSHQ-TobH", m.n_csh_tobh[i] * V_REV_T * us"mol")
        set_quantity!(state, "CSHQ-TobD", m.n_csh_tobd[i] * V_REV_T * us"mol")
        set_quantity!(state, "CSHQ-JenH", m.n_csh_jenh[i] * V_REV_T * us"mol")
        set_quantity!(state, "CSHQ-JenD", m.n_csh_jend[i] * V_REV_T * us"mol")
        if has_cash
            set_quantity!(state, "CSH3T-TobH", m.n_cash_tobh[i] * V_REV_T * us"mol")
            set_quantity!(state, "CSH3T-T5C",  m.n_cash_t5c[i]  * V_REV_T * us"mol")
            set_quantity!(state, "CSH3T-T2C",  m.n_cash_t2c[i]  * V_REV_T * us"mol")
        end
        (has_afm_ss || has_friedels) && set_quantity!(state, "C4AClH10", m.n_fs[i] * V_REV_T * us"mol")
        has_brucite && set_quantity!(state, "Brc", m.n_brc[i] * V_REV_T * us"mol")
        has_gyp && set_quantity!(state, GYPS_NAME_T, m.n_gyp[i] * V_REV_T * us"mol")
        if has_msh
            set_quantity!(state, MSH_EM_T[1], m.n_msh_08[i] * V_REV_T * us"mol")
            length(MSH_EM_T) > 1 && set_quantity!(state, MSH_EM_T[2], m.n_msh_13[i] * V_REV_T * us"mol")
        end
        if has_ldh
            set_quantity!(state, LDH_EM_T[1], m.n_ldh_m4[i] * V_REV_T * us"mol")
            length(LDH_EM_T) > 1 && set_quantity!(state, LDH_EM_T[2], m.n_ldh_m6[i] * V_REV_T * us"mol")
            length(LDH_EM_T) > 2 && set_quantity!(state, LDH_EM_T[3], m.n_ldh_m8[i] * V_REV_T * us"mol")
        end

        local state_eq
        try
            state_eq = equilibrate(state, OptimaOptimizer(tol=1e-7, verbose=false))
        catch e
            @warn "equilibrate failed at node $i" exception = e
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
        c_mg_new = ustrip(moles(state_eq, "Mg+2")) / V_liq
        c_si_new = sum(n * ustrip(moles(state_eq, sp)) for (sp, n) in SI_AQ_SPECIES_T
                       if ustrip(moles(state_eq, sp)) > 0.0) / V_liq
        _al_mol(sp) = try; ustrip(moles(state_eq, sp)); catch; 0.0; end
        c_al_new = sum(n * max(_al_mol(sp), 0.0) for (sp, n) in AL_AQ_SPECIES_T) / V_liq

        n_ch_new = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_T, 0.0)
        n_ett_new = max(ustrip(moles(state_eq, "ettringite")) / V_REV_T, 0.0)
        n_ms_new = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_T, 0.0)
        n_csh_tobh_new = max(ustrip(moles(state_eq, "CSHQ-TobH")) / V_REV_T, 0.0)
        n_csh_tobd_new = max(ustrip(moles(state_eq, "CSHQ-TobD")) / V_REV_T, 0.0)
        n_csh_jenh_new = max(ustrip(moles(state_eq, "CSHQ-JenH")) / V_REV_T, 0.0)
        n_csh_jend_new = max(ustrip(moles(state_eq, "CSHQ-JenD")) / V_REV_T, 0.0)
        n_cash_tobh_new = has_cash ? max(ustrip(moles(state_eq, "CSH3T-TobH")) / V_REV_T, 0.0) : 0.0
        n_cash_t5c_new  = has_cash ? max(ustrip(moles(state_eq, "CSH3T-T5C"))  / V_REV_T, 0.0) : 0.0
        n_cash_t2c_new  = has_cash ? max(ustrip(moles(state_eq, "CSH3T-T2C"))  / V_REV_T, 0.0) : 0.0
        n_fs_new = (has_afm_ss || has_friedels) ? max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_T, 0.0) : 0.0
        n_brc_new = has_brucite ? max(ustrip(moles(state_eq, "Brc")) / V_REV_T, 0.0) : 0.0
        n_gyp_new = has_gyp ? max(ustrip(moles(state_eq, GYPS_NAME_T)) / V_REV_T, 0.0) : 0.0
        n_msh_08_new = has_msh ? max(ustrip(moles(state_eq, MSH_EM_T[1])) / V_REV_T, 0.0) : 0.0
        n_msh_13_new = (has_msh && length(MSH_EM_T) > 1) ? max(ustrip(moles(state_eq, MSH_EM_T[2])) / V_REV_T, 0.0) : 0.0
        n_ldh_m4_new = has_ldh ? max(ustrip(moles(state_eq, LDH_EM_T[1])) / V_REV_T, 0.0) : 0.0
        n_ldh_m6_new = (has_ldh && length(LDH_EM_T) > 1) ? max(ustrip(moles(state_eq, LDH_EM_T[2])) / V_REV_T, 0.0) : 0.0
        n_ldh_m8_new = (has_ldh && length(LDH_EM_T) > 2) ? max(ustrip(moles(state_eq, LDH_EM_T[3])) / V_REV_T, 0.0) : 0.0

        # ── Dissolution/precipitation kinetics ────────────────────────────────
        # Caps each mineral change at what is reachable within Δt_snia.
        # Concentrations are corrected by mass balance (dissolution stoichiometry).
        # Convention: stoich > 0 = ion released into solution on dissolution.
        if isfinite(Δt_snia)
            kin = m.kin

            # Largest kinetically reachable change over Δt [mol/m³_concrete]
            # r_max = k [mol/m²/s] × A_s [m²/g] × M [g/mol] × n [mol/m³_concrete]
            r_CH = kin.k_CH * kin.A_CH * kin.M_CH * max(m.n_ch[i], 1e-20) * Δt_snia
            r_Ett = kin.k_Ett * kin.A_Ett * kin.M_Ett * max(m.n_ett[i], 1e-20) * Δt_snia
            r_MS = kin.k_MS * kin.A_MS * kin.M_MS * max(m.n_ms[i], 1e-20) * Δt_snia
            r_FS = kin.k_FS * kin.A_FS * kin.M_FS * max(m.n_fs[i], 1e-20) * Δt_snia
            r_Brc = kin.k_Brc * kin.A_Brc * kin.M_Brc * max(m.n_brc[i], 1e-20) * Δt_snia
            r_TobH = kin.k_CSH * kin.A_CSH * kin.M_TobH * max(m.n_csh_tobh[i], 1e-20) * Δt_snia
            r_TobD = kin.k_CSH * kin.A_CSH * kin.M_TobD * max(m.n_csh_tobd[i], 1e-20) * Δt_snia
            r_JenH = kin.k_CSH * kin.A_CSH * kin.M_JenH * max(m.n_csh_jenh[i], 1e-20) * Δt_snia
            r_JenD = kin.k_CSH * kin.A_CSH * kin.M_JenD * max(m.n_csh_jend[i], 1e-20) * Δt_snia
            r_CATobH = has_cash ? kin.k_CASH * kin.A_CASH * kin.M_CASH_TobH * max(m.n_cash_tobh[i], 1e-20) * Δt_snia : 0.0
            r_CAT5C  = has_cash ? kin.k_CASH * kin.A_CASH * kin.M_CASH_T5C  * max(m.n_cash_t5c[i],  1e-20) * Δt_snia : 0.0
            r_CAT2C  = has_cash ? kin.k_CASH * kin.A_CASH * kin.M_CASH_T2C  * max(m.n_cash_t2c[i],  1e-20) * Δt_snia : 0.0

            # Kinetically limited changes (clamping the equilibrium change)
            Δn_CH = clamp(n_ch_new - m.n_ch[i], -r_CH, r_CH)
            Δn_Ett = clamp(n_ett_new - m.n_ett[i], -r_Ett, r_Ett)
            Δn_MS = clamp(n_ms_new - m.n_ms[i], -r_MS, r_MS)
            Δn_FS = clamp(n_fs_new - m.n_fs[i], -r_FS, r_FS)
            Δn_Brc = clamp(n_brc_new - m.n_brc[i], -r_Brc, r_Brc)
            Δn_TobH = clamp(n_csh_tobh_new - m.n_csh_tobh[i], -r_TobH, r_TobH)
            Δn_TobD = clamp(n_csh_tobd_new - m.n_csh_tobd[i], -r_TobD, r_TobD)
            Δn_JenH = clamp(n_csh_jenh_new - m.n_csh_jenh[i], -r_JenH, r_JenH)
            Δn_JenD = clamp(n_csh_jend_new - m.n_csh_jend[i], -r_JenD, r_JenD)
            Δn_CATobH = clamp(n_cash_tobh_new - m.n_cash_tobh[i], -r_CATobH, r_CATobH)
            Δn_CAT5C  = clamp(n_cash_t5c_new  - m.n_cash_t5c[i],  -r_CAT5C,  r_CAT5C)
            Δn_CAT2C  = clamp(n_cash_t2c_new  - m.n_cash_t2c[i],  -r_CAT2C,  r_CAT2C)

            # Gap between equilibrium and kinetics (= Δn_eq - Δn_kin)
            # Positive: less dissolved / more precipitated than equilibrium.
            # Negative: less precipitated / more dissolved than equilibrium.
            δ_CH = (n_ch_new - m.n_ch[i]) - Δn_CH
            δ_Ett = (n_ett_new - m.n_ett[i]) - Δn_Ett
            δ_MS = (n_ms_new - m.n_ms[i]) - Δn_MS
            δ_FS = (n_fs_new - m.n_fs[i]) - Δn_FS
            δ_Brc = (n_brc_new - m.n_brc[i]) - Δn_Brc
            δ_TobH   = (n_csh_tobh_new  - m.n_csh_tobh[i])  - Δn_TobH
            δ_TobD   = (n_csh_tobd_new  - m.n_csh_tobd[i])  - Δn_TobD
            δ_JenH   = (n_csh_jenh_new  - m.n_csh_jenh[i])  - Δn_JenH
            δ_JenD   = (n_csh_jend_new  - m.n_csh_jend[i])  - Δn_JenD
            δ_CATobH = (n_cash_tobh_new - m.n_cash_tobh[i]) - Δn_CATobH
            δ_CAT5C  = (n_cash_t5c_new  - m.n_cash_t5c[i])  - Δn_CAT5C
            δ_CAT2C  = (n_cash_t2c_new  - m.n_cash_t2c[i])  - Δn_CAT2C

            # New mineral amounts (kinetically limited)
            n_ch_new = max(m.n_ch[i] + Δn_CH, 0.0)
            n_ett_new = max(m.n_ett[i] + Δn_Ett, 0.0)
            n_ms_new = max(m.n_ms[i] + Δn_MS, 0.0)
            n_fs_new = max(m.n_fs[i] + Δn_FS, 0.0)
            n_brc_new = max(m.n_brc[i] + Δn_Brc, 0.0)
            n_csh_tobh_new = max(m.n_csh_tobh[i] + Δn_TobH, 0.0)
            n_csh_tobd_new = max(m.n_csh_tobd[i] + Δn_TobD, 0.0)
            n_csh_jenh_new = max(m.n_csh_jenh[i] + Δn_JenH, 0.0)
            n_csh_jend_new = max(m.n_csh_jend[i] + Δn_JenD, 0.0)
            n_cash_tobh_new = max(m.n_cash_tobh[i] + Δn_CATobH, 0.0)
            n_cash_t5c_new  = max(m.n_cash_t5c[i]  + Δn_CAT5C,  0.0)
            n_cash_t2c_new  = max(m.n_cash_t2c[i]  + Δn_CAT2C,  0.0)

            # Correction of the aqueous concentrations by mass balance.
            # Δc_ion = (Δn_eq − Δn_kin) × stoich / (V_liq / V_REV_T)
            # where (Δn_eq − Δn_kin) = δ_*  [mol/m³_concrete]
            # Dissolution stoichiometry (mol released into solution per mol dissolved):
            #   Ca : CH=1, Ettringite=6, MS=4, TobH=5/6, TobD=5/6, JenH=9/6, JenD=10/6, FS=4
            #   Mg : Brucite=1
            #   SO₄: Ettringite=3, MS=1
            #   Cl : FS=2 (Cl released when Friedel's salt dissolves)
            #   Si : TobH=TobD=JenH=JenD=1
            fac_kin = V_REV_T / V_liq   # [m³_concrete / m³_solution] → dimensionless

            Δc_ca = (δ_CH * 1.0 + δ_Ett * 6.0 + δ_MS * 4.0 + δ_FS * 4.0
                     + δ_TobH * (5.0 / 6.0) + δ_TobD * (5.0 / 6.0)
                     + δ_JenH * (9.0 / 6.0) + δ_JenD * (10.0 / 6.0)
                     + δ_CATobH * XCAS_CASH_TOBH + δ_CAT5C * XCAS_CASH_T5C + δ_CAT2C * XCAS_CASH_T2C) * fac_kin
            Δc_mg = δ_Brc * 1.0 * fac_kin
            Δc_so4 = (δ_Ett * 3.0 + δ_MS * 1.0) * fac_kin
            Δc_cl = δ_FS * 2.0 * fac_kin
            Δc_si = (δ_TobH + δ_TobD + δ_JenH + δ_JenD
                     + δ_CATobH * (1.0 - AL_STOICH_CASH_TOBH) + δ_CAT5C * (1.0 - AL_STOICH_CASH_T5C) + δ_CAT2C * (1.0 - AL_STOICH_CASH_T2C)) * fac_kin
            Δc_al = (δ_CATobH * AL_STOICH_CASH_TOBH + δ_CAT5C * AL_STOICH_CASH_T5C + δ_CAT2C * AL_STOICH_CASH_T2C) * fac_kin

            c_ca_new = max(c_ca_new + Δc_ca, 0.0)
            c_mg_new = max(c_mg_new + Δc_mg, 0.0)
            c_so4_new = max(c_so4_new + Δc_so4, 0.0)
            c_cl_new = max(c_cl_new + Δc_cl, 0.0)
            c_si_new = max(c_si_new + Δc_si, 0.0)
            c_al_new = max(c_al_new + Δc_al, 0.0)
        end

        # ── Porosity ──────────────────────────────────────────────────────────
        phi_new = (phi_i
                   - (n_ch_new - m.n_ch[i])             * vm.Vm_CH
                   - (n_ett_new - m.n_ett[i])            * vm.Vm_ett
                   - (n_ms_new - m.n_ms[i])              * vm.Vm_ms
                   - (n_fs_new - m.n_fs[i])              * vm.Vm_fs
                   - (n_brc_new - m.n_brc[i])            * vm.Vm_brc
                   - (n_csh_tobh_new - m.n_csh_tobh[i]) * vm.Vm_TobH
                   - (n_csh_tobd_new - m.n_csh_tobd[i]) * vm.Vm_TobD
                   - (n_csh_jenh_new - m.n_csh_jenh[i]) * vm.Vm_JenH
                   - (n_csh_jend_new - m.n_csh_jend[i]) * vm.Vm_JenD
                   - (n_cash_tobh_new - m.n_cash_tobh[i]) * vm.Vm_CASH_TobH
                   - (n_cash_t5c_new  - m.n_cash_t5c[i])  * vm.Vm_CASH_T5C
                   - (n_cash_t2c_new  - m.n_cash_t2c[i])  * vm.Vm_CASH_T2C
                   - (n_msh_08_new - m.n_msh_08[i])     * vm.Vm_MSH_08
                   - (n_msh_13_new - m.n_msh_13[i])     * vm.Vm_MSH_13
                   - (n_ldh_m4_new - m.n_ldh_m4[i])     * vm.Vm_LDH_M4
                   - (n_ldh_m6_new - m.n_ldh_m6[i])     * vm.Vm_LDH_M6
                   - (n_ldh_m8_new - m.n_ldh_m8[i])     * vm.Vm_LDH_M8
                   - (n_gyp_new    - m.n_gyp[i])         * vm.Vm_Gyp)
        m.phi[i] = clamp(phi_new, 1e-4, 0.999)
        m.n_ch[i] = n_ch_new
        m.n_ett[i] = n_ett_new
        m.n_ms[i] = n_ms_new
        m.n_fs[i] = n_fs_new
        m.n_brc[i] = n_brc_new
        m.n_csh_tobh[i] = n_csh_tobh_new
        m.n_csh_tobd[i] = n_csh_tobd_new
        m.n_csh_jenh[i] = n_csh_jenh_new
        m.n_csh_jend[i] = n_csh_jend_new
        m.n_cash_tobh[i] = n_cash_tobh_new
        m.n_cash_t5c[i]  = n_cash_t5c_new
        m.n_cash_t2c[i]  = n_cash_t2c_new
        m.n_msh_08[i] = n_msh_08_new
        m.n_msh_13[i] = n_msh_13_new
        m.n_ldh_m4[i] = n_ldh_m4_new
        m.n_ldh_m6[i] = n_ldh_m6_new
        m.n_ldh_m8[i] = n_ldh_m8_new
        m.n_gyp[i]    = n_gyp_new
        m.c_oh_frozen[i] = max(c_oh_new, 1e-12)

        # ── Friedel Kd (numerical secant) ─────────────────────────────────────
        # With AFm-SS: computed as soon as total AFm is present (even at n_fs ≈ 0),
        # because the SO₄²⁻/Cl⁻ interlayer substitution happens continuously.
        dn_fs_dc = 0.0
        if (has_afm_ss && n_ms_new + n_fs_new > 1e-10) || (!has_afm_ss && has_friedels && n_fs_new > 1e-10)
            δ_cl = max(1.0, c_cl_new * 1e-2)
            phi_p = m.phi[i]
            n_cl_p = max(c_cl_new + δ_cl, 0.0) * phi_p * V_REV_T
            n_oh_p = max(c_na_new * phi_p * V_REV_T + c_k_new * phi_p * V_REV_T
                         + 2.0 * c_ca_new * phi_p * V_REV_T + 2.0 * c_mg_new * phi_p * V_REV_T
                         -
                         n_cl_p - 2.0 * c_so4_new * phi_p * V_REV_T - c_al_new * phi_p * V_REV_T, 1.0e-20)
            try
                state_p = ChemicalState(cs; T=T_q)
                set_quantity!(state_p, "H2O@", phi_p * V_REV_T * 55_500.0 * us"mol")
                set_quantity!(state_p, "OH-", n_oh_p * us"mol")
                set_quantity!(state_p, "Cl-", n_cl_p * us"mol")
                set_quantity!(state_p, "Na+", c_na_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "K+", c_k_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "Ca+2", c_ca_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "Mg+2", c_mg_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "SO4-2", c_so4_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "SiO2@", c_si_new * phi_p * V_REV_T * us"mol")
                set_quantity!(state_p, "Portlandite", n_ch_new * V_REV_T * us"mol")
                set_quantity!(state_p, "ettringite", n_ett_new * V_REV_T * us"mol")
                set_quantity!(state_p, "monosulphate12", n_ms_new * V_REV_T * us"mol")
                set_quantity!(state_p, "CSHQ-TobH", n_csh_tobh_new * V_REV_T * us"mol")
                set_quantity!(state_p, "CSHQ-TobD", n_csh_tobd_new * V_REV_T * us"mol")
                set_quantity!(state_p, "CSHQ-JenH", n_csh_jenh_new * V_REV_T * us"mol")
                set_quantity!(state_p, "CSHQ-JenD", n_csh_jend_new * V_REV_T * us"mol")
                if has_cash
                    set_quantity!(state_p, "CSH3T-TobH", n_cash_tobh_new * V_REV_T * us"mol")
                    set_quantity!(state_p, "CSH3T-T5C",  n_cash_t5c_new  * V_REV_T * us"mol")
                    set_quantity!(state_p, "CSH3T-T2C",  n_cash_t2c_new  * V_REV_T * us"mol")
                end
                set_quantity!(state_p, "C4AClH10", n_fs_new * V_REV_T * us"mol")
                has_brucite && set_quantity!(state_p, "Brc", n_brc_new * V_REV_T * us"mol")
                has_gyp && set_quantity!(state_p, GYPS_NAME_T, n_gyp_new * V_REV_T * us"mol")
                if has_msh
                    set_quantity!(state_p, MSH_EM_T[1], n_msh_08_new * V_REV_T * us"mol")
                    length(MSH_EM_T) > 1 && set_quantity!(state_p, MSH_EM_T[2], n_msh_13_new * V_REV_T * us"mol")
                end
                if has_ldh
                    set_quantity!(state_p, LDH_EM_T[1], n_ldh_m4_new * V_REV_T * us"mol")
                    length(LDH_EM_T) > 1 && set_quantity!(state_p, LDH_EM_T[2], n_ldh_m6_new * V_REV_T * us"mol")
                    length(LDH_EM_T) > 2 && set_quantity!(state_p, LDH_EM_T[3], n_ldh_m8_new * V_REV_T * us"mol")
                end
                state_p_eq = equilibrate(state_p, OptimaOptimizer(tol=1e-7, verbose=false))
                V_liq_p = ustrip(uconvert(us"m^3", state_p_eq.V_phases[].liquid))
                if V_liq_p > 1e-15
                    n_fs_p = max(ustrip(moles(state_p_eq, "C4AClH10")) / V_REV_T, 0.0)
                    dn_fs_dc = max(n_fs_p - n_fs_new, 0.0) / δ_cl
                end
            catch
            end
        end

        # ── Ternary DLM (CSHQ + C-A-S-H combined) ─────────────────────────────
        n_csh_cshq = n_csh_tobh_new + n_csh_tobd_new + n_csh_jenh_new + n_csh_jend_new
        n_csh_cash = n_cash_tobh_new + n_cash_t5c_new + n_cash_t2c_new
        n_csh_i = n_csh_cshq + n_csh_cash
        x_cas_cshq = n_csh_cshq > 1e-6 ? (
            (5 / 6 * n_csh_tobh_new + 5 / 6 * n_csh_tobd_new + 9 / 6 * n_csh_jenh_new + 10 / 6 * n_csh_jend_new) / n_csh_cshq
        ) : 1.5
        x_cas_cash = n_csh_cash > 1e-6 ? (
            XCAS_CASH_TOBH * n_cash_tobh_new + XCAS_CASH_T5C * n_cash_t5c_new + XCAS_CASH_T2C * n_cash_t2c_new
        ) / n_csh_cash : 1.5
        x_cas_i = n_csh_i > 1e-6 ? (x_cas_cshq * n_csh_cshq + x_cas_cash * n_csh_cash) / n_csh_i : 1.5
        dlm_i = m.dlm
        n_csh_dlm = dlm_i.n_csh0 > 0.0 ? dlm_i.n_csh0 : n_csh_i
        dS_Cl_dc = 0.0
        if n_csh_dlm > 0.0
            _, S_Cl_i, S_Na_i, S_K_i, S_Ca_i, S_Mg_i = solve_dlm_ternary(
                max(c_cl_new, 0.0), max(c_na_new, 0.0), max(c_k_new, 0.0),
                max(c_ca_new, 0.0), max(c_mg_new, 0.0), max(c_oh_new, ε),
                n_csh_dlm, x_cas_i; dlm=dlm_i, T_K=m.env.T_K,
            )
            m.S_Cl_dlm[i] = max(S_Cl_i, 0.0)
            m.S_Na_dlm[i] = max(S_Na_i, 0.0)
            m.S_K_dlm[i] = max(S_K_i, 0.0)
            m.S_Ca_dlm[i] = max(S_Ca_i, 0.0)
            m.S_Mg_dlm[i] = max(S_Mg_i, 0.0)
            # ∂S_Cl/∂c_Cl (c_Ca held fixed) for the DLM Kd
            δ_dlm = max(1.0e-3, c_cl_new * 1e-3)
            _, S_Cl_p, _, _, _, _ = solve_dlm_ternary(
                max(c_cl_new + δ_dlm, 0.0), max(c_na_new, 0.0), max(c_k_new, 0.0),
                max(c_ca_new, 0.0), max(c_mg_new, 0.0), max(c_oh_new, ε),
                n_csh_dlm, x_cas_i; dlm=dlm_i, T_K=m.env.T_K,
            )
            dS_Cl_dc = (max(S_Cl_p, 0.0) - m.S_Cl_dlm[i]) / δ_dlm
        else
            m.S_Cl_dlm[i] = 0.0
            m.S_Na_dlm[i] = 0.0
            m.S_K_dlm[i] = 0.0
            m.S_Ca_dlm[i] = 0.0
            m.S_Mg_dlm[i] = 0.0
        end

        m.Kd_Cl[i] = 2.0 * dn_fs_dc + dS_Cl_dc
        m.Kd_Na[i] = m.S_Na_dlm[i] / max(c_na_new, ε)
        m.Kd_K[i] = m.S_K_dlm[i] / max(c_k_new, ε)
        m.Kd_Ca[i] = m.S_Ca_dlm[i] / max(c_ca_new, ε)
        m.Kd_Mg[i] = m.S_Mg_dlm[i] / max(c_mg_new, ε)
        m.Kd_Al[i] = 0.0

        # ── Kd_SO4 : ∂(3·n_ett + n_ms + n_gyp) / ∂c_SO4 ─────────────────────
        # ── Numerical secant — same structure as the Friedel Kd_Cl above.
        kd_so4_i = 0.0
        n_solid_so4_base = 3.0 * n_ett_new + n_ms_new + (has_gyp ? n_gyp_new : 0.0)
        if n_solid_so4_base > 1.0e-10
            δ_so4   = max(1.0, c_so4_new * 1e-2)
            phi_s   = m.phi[i]
            n_so4_p = max(c_so4_new + δ_so4, 0.0) * phi_s * V_REV_T
            n_oh_sp = max(c_na_new * phi_s * V_REV_T + c_k_new * phi_s * V_REV_T
                          + 2.0 * c_ca_new * phi_s * V_REV_T + 2.0 * c_mg_new * phi_s * V_REV_T
                          - c_cl_new * phi_s * V_REV_T - 2.0 * n_so4_p - c_al_new * phi_s * V_REV_T, 1.0e-20)
            try
                state_s = ChemicalState(cs; T=T_q)
                set_quantity!(state_s, "H2O@",  phi_s * V_REV_T * 55_500.0 * us"mol")
                set_quantity!(state_s, "OH-",   n_oh_sp * us"mol")
                set_quantity!(state_s, "Cl-",   c_cl_new  * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "Na+",   c_na_new  * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "K+",    c_k_new   * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "Ca+2",  c_ca_new  * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "Mg+2",  c_mg_new  * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "SO4-2", n_so4_p * us"mol")
                set_quantity!(state_s, "SiO2@", c_si_new  * phi_s * V_REV_T * us"mol")
                set_quantity!(state_s, "Portlandite",    n_ch_new      * V_REV_T * us"mol")
                set_quantity!(state_s, "ettringite",     n_ett_new     * V_REV_T * us"mol")
                set_quantity!(state_s, "monosulphate12", n_ms_new      * V_REV_T * us"mol")
                set_quantity!(state_s, "CSHQ-TobH",     n_csh_tobh_new * V_REV_T * us"mol")
                set_quantity!(state_s, "CSHQ-TobD",     n_csh_tobd_new * V_REV_T * us"mol")
                set_quantity!(state_s, "CSHQ-JenH",     n_csh_jenh_new * V_REV_T * us"mol")
                set_quantity!(state_s, "CSHQ-JenD",     n_csh_jend_new * V_REV_T * us"mol")
                if has_cash
                    set_quantity!(state_s, "CSH3T-TobH", n_cash_tobh_new * V_REV_T * us"mol")
                    set_quantity!(state_s, "CSH3T-T5C",  n_cash_t5c_new  * V_REV_T * us"mol")
                    set_quantity!(state_s, "CSH3T-T2C",  n_cash_t2c_new  * V_REV_T * us"mol")
                end
                (has_afm_ss || has_friedels) && set_quantity!(state_s, "C4AClH10", n_fs_new  * V_REV_T * us"mol")
                has_brucite && set_quantity!(state_s, "Brc",        n_brc_new * V_REV_T * us"mol")
                has_gyp     && set_quantity!(state_s, GYPS_NAME_T,  n_gyp_new * V_REV_T * us"mol")
                if has_msh
                    set_quantity!(state_s, MSH_EM_T[1], n_msh_08_new * V_REV_T * us"mol")
                    length(MSH_EM_T) > 1 && set_quantity!(state_s, MSH_EM_T[2], n_msh_13_new * V_REV_T * us"mol")
                end
                if has_ldh
                    set_quantity!(state_s, LDH_EM_T[1], n_ldh_m4_new * V_REV_T * us"mol")
                    length(LDH_EM_T) > 1 && set_quantity!(state_s, LDH_EM_T[2], n_ldh_m6_new * V_REV_T * us"mol")
                    length(LDH_EM_T) > 2 && set_quantity!(state_s, LDH_EM_T[3], n_ldh_m8_new * V_REV_T * us"mol")
                end
                state_s_eq = equilibrate(state_s, OptimaOptimizer(tol=1e-7, verbose=false))
                V_liq_s = ustrip(uconvert(us"m^3", state_s_eq.V_phases[].liquid))
                if V_liq_s > 1e-15
                    n_ett_sp  = max(ustrip(moles(state_s_eq, "ettringite"))      / V_REV_T, 0.0)
                    n_ms_sp   = max(ustrip(moles(state_s_eq, "monosulphate12"))  / V_REV_T, 0.0)
                    n_gyp_sp  = has_gyp ? max(ustrip(moles(state_s_eq, GYPS_NAME_T)) / V_REV_T, 0.0) : 0.0
                    n_solid_so4_pert = 3.0 * n_ett_sp + n_ms_sp + n_gyp_sp
                    kd_so4_i = max((n_solid_so4_pert - n_solid_so4_base) / δ_so4, 0.0)
                end
            catch
            end
        end
        m.Kd_SO4[i] = kd_so4_i

        u[ICL_T,  i] = max(c_cl_new,  0.0)
        u[INA_T,  i] = max(c_na_new,  0.0)
        u[IK_T,   i] = max(c_k_new,   0.0)
        u[ICA_T,  i] = max(c_ca_new,  0.0)
        u[IMG_T,  i] = max(c_mg_new,  0.0)
        u[ISO4_T, i] = max(c_so4_new, 0.0)
        u[ISI_T,  i] = max(c_si_new,  0.0)
        u[IAL_T,  i] = max(c_al_new,  0.0)
    end
end

"""
    plot_M100_ternary(results, grid; n_curves, with_toughreact, save_path)

Returns a tuple of 5 figures (solid lines = PoroMechanics.jl, grey dashes = TOUGHREACT/Thermoddem):

  fig_ions    (2×3) — Cl⁻ | Mg²⁺ | SO₄²⁻ | Na⁺ | Ca²⁺ | pH
  fig_csh     (2×3) — Si  | φ    | Ca/Si | portlandite | ettringite | CSHQ
  fig_solids  (2×3) — Brucite | Friedel | MonoSulfo | M-S-H | LDH-OH | Gypse
  fig_cl      (1×1) — Cl⁻ forms (free / DLM ≡SiOCaCl / Friedel / total) at the final time
  fig_dlm     (2×2) — Adsorption DLM ternaire : S_Cl | S_Na | S_K | S_Ca

The TOUGHREACT references are overlaid when `with_toughreact=true` and when the
`toughreact_ref()` function is reachable from the calling context (it lives in tran2018.jl).

If `save_path` is given (e.g. "results/fig.png"), the 5 figures are saved with the
suffixes `_ions`, `_csh`, `_solids`, `_cl`, `_dlm` inserted before the extension.
"""
function plot_M100_ternary(results, grid; n_curves=4, with_toughreact=false, save_path=nothing)

    x_dm = grid[Coordinates][1, :] .* 10.0
    t_yr_s = 3.1536e7
    n_t = length(results)
    idxs = unique(clamp.(round.(Int, range(1, n_t; length=n_curves)), 1, n_t))
    pal = [:steelblue, :darkorange, :crimson, :forestgreen, :purple, :teal]

    fmt = (
        titlefontsize=11,
        guidefontsize=9,
        tickfontsize=8,
        legendfontsize=7,
        left_margin=14 * Plots.mm,
        bottom_margin=5 * Plots.mm,
    )

    tr = nothing
    if with_toughreact
        try
            tr = toughreact_ref()
        catch
            @warn "toughreact_ref() unreachable — TOUGHREACT comparison disabled"
        end
    end
    tr_cols = [:gray72, :gray50, :gray28, :black]
    tr_lbl = k -> "$(tr.t_yr[k]) yr ◆TR"

    function add_tr!(p, key::Symbol)
        isnothing(tr) && return
        for k in eachindex(tr.t_yr)
            plot!(p, tr.x_dm, getfield(tr, key)[k];
                lw=1.5, ls=:dash, color=tr_cols[k], label=tr_lbl(k))
        end
    end

    tlbl(ti) = "t = $(round(results[ti][1] / t_yr_s; digits=2)) yr"

    # ── p1 : Cl⁻ ─────────────────────────────────────────────────────────────
    p1 = plot(; fmt..., xlabel="x [dm]", ylabel="Cl⁻ [mol/m³ sol.]",
        title="Cl⁻ libre", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p1, x_dm, results[ti][2][ICL_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p1, :cl)

    # ── p2 : Mg²⁺ ────────────────────────────────────────────────────────────
    p2 = plot(; fmt..., xlabel="x [dm]", ylabel="Mg²⁺ [mol/m³ sol.]",
        title="Mg²⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p2, x_dm, results[ti][2][IMG_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p2, :mg)

    # ── p3 : SO₄²⁻ ───────────────────────────────────────────────────────────
    p3 = plot(; fmt..., xlabel="x [dm]", ylabel="SO₄²⁻ [mol/m³ sol.]",
        title="SO₄²⁻", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p3, x_dm, results[ti][2][ISO4_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p3, :so4)

    # ── p4 : Na⁺ ─────────────────────────────────────────────────────────────
    p4 = plot(; fmt..., xlabel="x [dm]", ylabel="Na⁺ [mol/m³ sol.]",
        title="Na⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p4, x_dm, results[ti][2][INA_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p4, :na)

    # ── p5 : Ca²⁺ ────────────────────────────────────────────────────────────
    p5 = plot(; fmt..., xlabel="x [dm]", ylabel="Ca²⁺ [mol/m³ sol.]",
        title="Ca²⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p5, x_dm, results[ti][2][ICA_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p5, :ca)

    # ── p6 : pH ──────────────────────────────────────────────────────────────
    p6 = plot(; fmt..., xlabel="x [dm]", ylabel="pH",
        title="pH", legend=:bottomright)
    for (k, ti) in enumerate(idxs)
        c_oh_k = results[ti][9]
        pH_k = [14.0 + log10(max(c_oh_k[j], 1e-20) / 1000.0) for j in eachindex(x_dm)]
        plot!(p6, x_dm, pH_k; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p6, :pH)

    # ── p7: total dissolved Si ────────────────────────────────────────────────
    p7 = plot(; fmt..., xlabel="x [dm]", ylabel="Si [mol/m³ sol.]",
        title="Total dissolved Si", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p7, x_dm, results[ti][2][ISI_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p8: porosity ──────────────────────────────────────────────────────────
    p8 = plot(; fmt..., xlabel="x [dm]", ylabel="φ [-]",
        title="Porosity", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p8, x_dm, results[ti][3]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    hline!(p8, [0.182]; lw=1, ls=:dot, color=:gray50, label="φ₀")
    add_tr!(p8, :phi)

    # ── p9: Ca/Si ratio, CSHQ + C-A-S-H combined ──────────────────────────────
    cs_TobH, cs_TobD, cs_JenH, cs_JenD = 5 / 6, 5 / 6, 9 / 6, 10 / 6
    p9 = plot(; fmt..., xlabel="x [dm]", ylabel="Ca/Si [-]",
        title="Ca/Si ratio (CSHQ + C-A-S-H)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        tobh = results[ti][10]
        tobd = results[ti][11]
        jenh = results[ti][12]
        jend = results[ti][13]
        catobh = results[ti][24]
        cat5c  = results[ti][25]
        cat2c  = results[ti][26]
        total_cshq = tobh .+ tobd .+ jenh .+ jend
        total_cash = catobh .+ cat5c .+ cat2c
        total = total_cshq .+ total_cash
        cs_ratio = @. ifelse(total > 1e-6,
            (cs_TobH * tobh + cs_TobD * tobd + cs_JenH * jenh + cs_JenD * jend
             + XCAS_CASH_TOBH * catobh + XCAS_CASH_T5C * cat5c + XCAS_CASH_T2C * cat2c) / total, NaN)
        plot!(p9, x_dm, cs_ratio; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    hline!(p9, [cs_TobH]; lw=1, ls=:dot, color=:gray50, label="TobH/D limite")
    hline!(p9, [cs_JenD]; lw=1, ls=:dash, color=:gray30, label="JenD limite")

    # ── p10 : Portlandite ─────────────────────────────────────────────────────
    p10 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Portlandite Ca(OH)₂", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p10, x_dm, results[ti][4]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p10, :n_ch)

    # ── p11 : Ettringite ──────────────────────────────────────────────────────
    p11 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Ettringite", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p11, x_dm, results[ti][5]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p12 : CSHQ + C-A-S-H (totaux + end-members) ─────────────────────────
    p12 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="CSHQ + C-A-S-H (totaux + end-members)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        col = pal[mod1(k, end)]
        tobh  = results[ti][10]
        tobd  = results[ti][11]
        jenh  = results[ti][12]
        jend  = results[ti][13]
        catobh = results[ti][24]
        cat5c  = results[ti][25]
        cat2c  = results[ti][26]
        total_cshq = tobh .+ tobd .+ jenh .+ jend
        total_cash = catobh .+ cat5c .+ cat2c
        plot!(p12, x_dm, total_cshq; lw=2.5, color=col, label="CSHQ $(tlbl(ti))")
        plot!(p12, x_dm, total_cash; lw=2, ls=:dash, color=col, label="C-A-S-H $(tlbl(ti))")
        plot!(p12, x_dm, tobh; lw=1, ls=:dot, color=col, label="")
        plot!(p12, x_dm, jend; lw=1, ls=:dashdot, color=col, label="")
    end

    # ── p6b : Al total dissous ────────────────────────────────────────────────
    p6b = plot(; fmt..., xlabel="x [dm]", ylabel="Al [mol/m³ sol.]",
        title="Al total dissous", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p6b, x_dm, results[ti][2][IAL_T, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p13 : Brucite ─────────────────────────────────────────────────────────
    p13 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Brucite Mg(OH)₂", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p13, x_dm, results[ti][8]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p13, :n_brc)

    # ── p14: Friedel's salt ───────────────────────────────────────────────────
    p14 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Friedel's salt (C₄AClH₁₀)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p14, x_dm, results[ti][7]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p15 : Monosulfoaluminate ──────────────────────────────────────────────
    p15 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Monosulfoaluminate", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p15, x_dm, results[ti][6]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    if !isnothing(tr)
        for k in eachindex(tr.t_yr)
            plot!(p15, tr.x_dm, tr.n_ms[k]; lw=1.5, ls=:dashdot,
                color=tr_cols[k], label="MS $(tr_lbl(k))")
        end
    end

    # ── p16 : M-S-H (total + end-members) ────────────────────────────────────
    p16 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="M-S-H (M075SH / M15SH)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        col = pal[mod1(k, end)]
        msh08 = results[ti][14]
        msh13 = results[ti][15]
        total_msh = msh08 .+ msh13
        plot!(p16, x_dm, total_msh; lw=2.5, color=col, label="Total $(tlbl(ti))")
        plot!(p16, x_dm, msh08; lw=1, ls=:dash, color=col, label="Mg/Si=0.75 (M075SH)")
        plot!(p16, x_dm, msh13; lw=1, ls=:dot, color=col, label="Mg/Si=1.50 (M15SH)")
    end

    # ── p17 : LDH-OH (M4A/M6A/M8A) ──────────────────────────────────────────
    p17 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="LDH-OH (Mg-Al)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        col = pal[mod1(k, end)]
        ldh_m4 = results[ti][16]
        ldh_m6 = results[ti][17]
        ldh_m8 = results[ti][18]
        total_ldh = ldh_m4 .+ ldh_m6 .+ ldh_m8
        plot!(p17, x_dm, total_ldh; lw=2.5, color=col, label="Total $(tlbl(ti))")
        plot!(p17, x_dm, ldh_m4; lw=1, ls=:dash, color=col, label="M4A-OH-LDH $(tlbl(ti))")
        plot!(p17, x_dm, ldh_m6; lw=1, ls=:dot, color=col, label="M6A-OH-LDH $(tlbl(ti))")
        plot!(p17, x_dm, ldh_m8; lw=1, ls=:dashdot, color=col, label="M8A-OH-LDH $(tlbl(ti))")
    end

    # ── p18: split of the Cl⁻ forms at the final time ─────────────────────────
    M_Cl_ = 35.453
    fac_g_ = M_Cl_ * 100.0 / 359_000.0
    u_f_ = results[end][2]
    phi_f_ = results[end][3]
    n_fs_f_ = results[end][7]
    S_Cl_f_ = results[end][19]
    t_lbl_f_ = tlbl(length(results))
    C_libre_f = u_f_[ICL_T, :] .* phi_f_ .* fac_g_
    C_dlm_f = S_Cl_f_ .* fac_g_
    C_friedel_f = 2.0 .* n_fs_f_ .* fac_g_
    C_total_f = C_libre_f .+ C_dlm_f .+ C_friedel_f
    p18 = plot(; fmt..., xlabel="x [dm]", ylabel="Cl [g/100g ciment]",
        title="Cl⁻ forms — ternary complex ($t_lbl_f_)", legend=:topright)
    plot!(p18, x_dm, C_libre_f; lw=2, color=:steelblue, label="Cl libre")
    plot!(p18, x_dm, C_dlm_f; lw=2, color=:darkorange, label="Cl DLM (≡SiOCaCl)")
    plot!(p18, x_dm, C_friedel_f; lw=2, color=:crimson, label="Cl Friedel")
    plot!(p18, x_dm, C_total_f; lw=3, color=:black, ls=:dash, label="Cl total")

    # ── p19–p22: ternary DLM adsorption over time ─────────────────────────────
    # [19] S_Cl_dlm  [20] S_Na_dlm  [21] S_K_dlm  [22] S_Ca_dlm
    p19 = plot(; fmt..., xlabel="x [dm]", ylabel="S_Cl [mol/m³ concrete]",
        title="DLM-adsorbed Cl⁻ (≡SiOCaCl)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p19, x_dm, results[ti][19]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    p20 = plot(; fmt..., xlabel="x [dm]", ylabel="S_Na [mol/m³ concrete]",
        title="DLM-adsorbed Na⁺ (≡SiONa)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p20, x_dm, results[ti][20]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    p21 = plot(; fmt..., xlabel="x [dm]", ylabel="S_K [mol/m³ concrete]",
        title="DLM-adsorbed K⁺ (≡SiOK)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p21, x_dm, results[ti][21]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    p22 = plot(; fmt..., xlabel="x [dm]", ylabel="S_Ca [mol/m³ concrete]",
        title="DLM-adsorbed Ca²⁺ (≡SiOCa⁺)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p22, x_dm, results[ti][22]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p23 : Gypse secondaire CaSO₄·2H₂O ───────────────────────────────────
    p23 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Gypse secondaire (CaSO₄·2H₂O)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p23, x_dm, results[ti][23]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    fig_ions = plot(p1, p2, p3, p4, p5, p6, p6b; layout=(2, 4), size=(2400, 900))
    fig_csh = plot(p7, p8, p9, p10, p11, p12; layout=(2, 3), size=(1800, 900))
    fig_solids = plot(p13, p14, p15, p16, p17, p23; layout=(2, 3), size=(1800, 900))
    fig_cl = plot(p18; size=(900, 600))
    fig_dlm = plot(p19, p20, p21, p22; layout=(2, 2), size=(1200, 900))

    if save_path !== nothing
        base, ext = splitext(save_path)
        savefig(fig_ions, base * "_ions" * ext)
        savefig(fig_csh, base * "_csh" * ext)
        savefig(fig_solids, base * "_solids" * ext)
        savefig(fig_cl, base * "_cl" * ext)
        savefig(fig_dlm, base * "_dlm" * ext)
    end
    return (fig_ions, fig_csh, fig_solids, fig_cl, fig_dlm)
end
