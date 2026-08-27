# Marks 2015 example — seawater attack on M100 concrete (CEM I 52.5N)
#
# Experimental case: Tran, Soive et al. 2026
# Simulation TOUGHREACT  : R_38_Marks2015.py
# Thermodynamic database: cemdata18 — Lothenbach et al. (2019). Cem. Concr. Res., 115, 472–506.
#
# M100 concrete mix:
#   CEM I 52,5N  : 359 kg/m³    (Na₂O=0.18%, K₂O=0.28%, SiO₂=20.06%, Al₂O₃=5.3%,
#                                 Fe₂O₃=2.11%, CaO=66.3%, SO₃=3.3%, CO₂=0.18%)
#   Eau          : 156 kg/m³     (E/C = 0.435)
#   Porosity     : φ = 0.182
#   D_app (Cl⁻) : 1.45×10⁻¹² m²/s
#
# Hydratation (Phase 1) :
#   Thermodynamic equilibrium (no hydration kinetics) starting from the mineral
#   contents calibrated on the TOUGHREACT/Thermoddem reference (solid.out t=0):
#     Portlandite    : 0.05036 m³/m³  → 1524 mol/m³ concrete
#     Ettringite     : 0.00910 m³/m³  →   12.9 mol/m³
#     Monosulfoalum. : 0.03419 m³/m³  →  110.6 mol/m³
#   The call to equilibrate() recomputes the matching pore solution.
#
# Reactive transport (Phase 2 — SNIA):
#   6 primary species: Cl⁻, Na⁺, K⁺, Ca²⁺, Mg²⁺, SO₄²⁻
#   Chemistry: equilibrate() (Gibbs) node by node
#   Solids: portlandite, ettringite, monosulphoaluminate, Friedel's salt, brucite
#   Kd    : Friedel secant for Cl⁻  (no DLM in this phase)
#
# Boundary solution (Atlantic seawater at 15 °C):
#   Na⁺=459, K⁺=9.71, Ca²⁺=9.97, Mg²⁺=52.2, SO₄²⁻=27.6, Cl⁻=546 mol/m³
#
# Usage :
#   julia --project examples/chloride_ingress/run_Marks2015.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver

# ── Transport species indices ─────────────────────────────────────────────────
const ICL_M = 1   # Cl⁻   z = −1
const INA_M = 2   # Na⁺   z = +1
const IK_M = 3   # K⁺    z = +1
const ICA_M = 4   # Ca²⁺  z = +2
const IMG_M = 5   # Mg²⁺  z = +2
const ISO4_M = 6   # SO₄²⁻ z = −2
const ISI_M = 7   # total dissolved Si (SiO₂@, HSiO₃⁻, SiO₃²⁻, complexes…) mean z ~ 0

# Aqueous Si species in cemdata18 (summed to obtain the total Si).
# Si₄O₁₀⁴⁻ carries 4 Si atoms per formula — factor 4 in the sum.
const SI_AQ_SPECIES = [
    ("SiO2@", 1), ("HSiO3-", 1), ("SiO3-2", 1),
    ("Ca(HSiO3)+", 1), ("Mg(HSiO3)+", 1), ("CaSiO3@", 1),
    ("AlSiO5-3", 1), ("Si4O10-4", 4),
]

const V_REV_M = 1.0e-3   # [m³] representative elementary volume = 1 dm³

# ── End-member names (used for the ChemistryLab calls) ────────────────────────
const CSHQ_EM = ("CSHQ-TobD", "CSHQ-TobH", "CSHQ-JenH", "CSHQ-JenD")
const MSH_EM  = ("M075SH", "M15SH")
const LDH_EM  = ("M4A-OH-LDH", "M6A-OH-LDH", "M8A-OH-LDH")

# Alternative C-S-H phases, excluded so they do not interfere with CSHQ
const EXCLUDE_NONCSHQ = [
    "CSH3T-TobH", "CSH3T-T5C", "CSH3T-T2C",
    "ECSH1-TobCa", "ECSH1-SH", "ECSH1-NaSH", "ECSH1-KSH", "ECSH1-SrSH",
    "ECSH2-TobCa", "ECSH2-JenCa", "ECSH2-NaSH", "ECSH2-KSH",
    "ECSH2-SrSH", "ECSH2-SrSH(ACW)", "Tob-I", "Tob-II", "Jennite",
    "TobH-CNASHss", "T5C-CNASHss", "T2C-CNASHss",
]

include("physdata.jl")

# ════════════════════════════════════════════════════════════════════════════════
# Material data
# ════════════════════════════════════════════════════════════════════════════════

"Mineralogical composition of the clinker (Bogue mass fractions) and alkali content."
Base.@kwdef struct ClinkerComposition
    na2o_frac::Float64 = 0.0018   # fraction massique Na₂O  (CEM I 52,5N)
    k2o_frac::Float64  = 0.0028   # fraction massique K₂O
    f_C3S::Float64     = 0.6942   # alite
    f_C2S::Float64     = 0.0515   # belite
    f_C3A::Float64     = 0.1047   # aluminate tricalcique
    f_C4AF::Float64    = 0.0642   # ferroaluminate
    f_Gp::Float64      = 0.0709   # gypse
end

"Concrete mix: binder content and water/cement ratio."
Base.@kwdef struct MixDesign
    m_clinker::Float64 = 359.0   # [kg/m³_concrete]
    wc::Float64        = 0.435   # eau/ciment massique
end

"""
Material-specific transport parameters.

  `phi0`    : porosity accessible to transport (measured value, used for the
              Oh-Jang tortuosity and as the initial state).
  `phi_c`   : capillary percolation threshold (Oh-Jang 2004).
  `n_OJ`    : exponent of the Oh-Jang model.
  `ds_OJ`   : normalised gel/capillary diffusivity.
  `tau_agg` : aggregate tortuosity factor (calibrated on the measured D_app).
  `n_csh0`  : fixed C-S-H content [mol/m³_concrete] for the DLM.
              0.0 (default) → computed dynamically after each Gibbs step.
"""
Base.@kwdef struct TransportModel
    phi0::Float64    = 0.182   # measured M100 porosity
    phi_c::Float64   = 0.10    # percolation threshold
    n_OJ::Float64    = 2.0     # exposant Oh-Jang
    ds_OJ::Float64   = 0.015   # gel/capillary diffusivity
    tau_agg::Float64 = 0.15    # aggregate tortuosity (calibrated on M100, D_app=1.45e-12)
    n_csh0::Float64  = 0.0     # [mol_CSH/m³_concrete] — 0 = dynamique
end

"All data specific to the cementitious material under study."
Base.@kwdef struct CementMaterial
    clinker::ClinkerComposition = ClinkerComposition()
    mix::MixDesign              = MixDesign()
    transport::TransportModel   = TransportModel()
end

# ════════════════════════════════════════════════════════════════════════════════
# Test-specific data (exposure environment)
# ════════════════════════════════════════════════════════════════════════════════

"""
Exposure conditions: temperature and composition of the aggressive solution.
Defaults = Atlantic seawater at 15 °C (Thermoddem 2023).
"""
Base.@kwdef struct ExposureConditions
    T_K::Float64   = 293.15   # temperature [K]
    c_Cl::Float64  = 546.0    # [mol/m³_water]
    c_Na::Float64  = 459.0
    c_K::Float64   = 9.71
    c_Ca::Float64  = 9.97
    c_Mg::Float64  = 52.2
    c_SO4::Float64 = 27.6
    c_Si::Float64  = 0.005    # total surface Atlantic Si (~5 µmol/L)
end

function _k_na_dlm(dlm::DLMConstants, x_cas::Float64)
    xT, xJ = 0.83, 1.67
    x_c = clamp(x_cas, xT, xJ)
    return dlm.K_Na_Jen + (dlm.K_Na_Tob - dlm.K_Na_Jen) * (xJ - x_c) / (xJ - xT)
end

"""
    solve_dlm_marks(c_Cl, c_Na, c_K, c_Ca, c_Mg, c_OH, n_csh, x_cas; dlm, T_K)

DLM equilibrium for the C-S-H with the Mg²⁺ extension (seawater).

!!! note
    This belongs in ChemistryLab.jl — surface complexation is chemistry, not transport.
    It lives here only until ChemistryLab.jl exposes it.

Bisection on β ∈ [−10, 10] of σ₀(β) = σ_DL(β).
Retourne `(β, S_Cl, S_Na, S_K, S_Ca, S_Mg)` en mol/m³_concrete.
Returns zeros when `n_csh ≤ 0` (DLM disabled).
"""
function solve_dlm_marks(
    c_Cl::Float64, c_Na::Float64, c_K::Float64, c_Ca::Float64,
    c_Mg::Float64, c_OH::Float64, n_csh::Float64, x_cas::Float64;
    dlm::DLMConstants,
    T_K::Float64=293.15,
)
    n_csh ≤ 0.0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    c_H = dlm.Kw_SI / max(c_OH, 1.0e-20)
    I = max(0.5 * (c_Cl + c_Na + c_K + 4.0 * c_Ca + 4.0 * c_Mg + c_OH + c_H), 1.0)

    Ka1 = dlm.Ka1
    KCa = dlm.K_Ca
    KMg = dlm.K_Mg
    KCl = dlm.K_OHCl
    KNa = _k_na_dlm(dlm, x_cas)
    KK = KNa   # K⁺: same constant as Na⁺ (Tran 2018)

    @inline A(β) = (1.0
                    + Ka1 * exp(β) / c_H
                    + KCa * Ka1 * c_Ca * exp(-β) / c_H
                    + KMg * Ka1 * c_Mg * exp(-β) / c_H
                    + KCl * c_Cl * exp(β)
                    + (KNa * c_Na + KK * c_K) * Ka1 / c_H)

    @inline B(β) = (KCa * Ka1 * c_Ca * exp(-β) / c_H
                    +
                    KMg * Ka1 * c_Mg * exp(-β) / c_H
                    -
                    Ka1 * exp(β) / c_H
                    -
                    KCl * c_Cl * exp(β))

    F_val = 96485.0
    R_val = 8.314
    σ_cap = sqrt(8.0 * 8.854e-12 * dlm.eps_r * R_val * T_K * I)

    f(β) = F_val * dlm.Gamma_max * B(β) / A(β) - σ_cap * sinh(β / 2.0)

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
    X = dlm.Gamma_max / A(β)

    theta_OHCl = KCl * X * c_Cl * exp(β)
    theta_OCa = KCa * Ka1 * X * c_Ca * exp(-β) / c_H
    theta_OMg = KMg * Ka1 * X * c_Mg * exp(-β) / c_H
    theta_ONa = KNa * Ka1 * X * c_Na / c_H
    theta_OK = KK * Ka1 * X * c_K / c_H

    fac = dlm.a_s * n_csh
    S_Cl = theta_OHCl * fac
    S_Na = theta_ONa * fac
    S_K = theta_OK * fac
    S_Ca = theta_OCa * fac
    S_Mg = theta_OMg * fac

    return β, S_Cl, S_Na, S_K, S_Ca, S_Mg
end

# ── Model ─────────────────────────────────────────────────────────────────────

"""
State of the M100 concrete model — seawater attack (Marks 2015).

Transport: 7 primary species (Cl⁻, Na⁺, K⁺, Ca²⁺, Mg²⁺, SO₄²⁻, Si), Fick's law.
Chemistry: Gibbs through equilibrate(), node by node (SNIA).
Solids   : portlandite, ettringite, monosulphoaluminate, Friedel's salt, brucite,
            M-S-H (solution solide M075SH/M15SH), LDH-OH (M4A/M6A/M8A-OH-LDH).

The input data is grouped into immutable structs:
  `diff` : free-water diffusion coefficients (properties of the ions)
  `vm`   : cemdata18 molar volumes
  `dlm`  : DLM constants from the literature
  `mat`  : material data (composition, mix, transport)
  `env`  : exposure conditions (seawater, temperature)
"""
mutable struct Marks2015Model <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64

    # ── Immutable data ────────────────────────────────────────────────────────
    diff::IonicDiffusivities
    vm::MolarVolumes
    dlm::DLMConstants

    # ── Material and environment data ─────────────────────────────────────────
    mat::CementMaterial
    env::ExposureConditions

    # ── Spatial fields [mol/m³_concrete] ──────────────────────────────────────
    phi::Vector{Float64}
    n_ch::Vector{Float64}        # Portlandite Ca(OH)₂
    n_ett::Vector{Float64}       # Ettringite
    n_ms::Vector{Float64}        # Monosulfoaluminate
    n_fs::Vector{Float64}        # Friedel's salt C₄AClH₁₀
    n_brc::Vector{Float64}       # Brucite Mg(OH)₂
    n_csh_tobh::Vector{Float64}  # CSHQ-TobH C₀.₆₇SH₁.₅
    n_csh_tobd::Vector{Float64}  # CSHQ-TobD C₀.₆₇SH₀.₅
    n_csh_jenh::Vector{Float64}  # CSHQ-JenH C₁.₅SH₂.₅
    n_csh_jend::Vector{Float64}  # CSHQ-JenD C₁.₅SH₀.₈₃
    n_msh_08::Vector{Float64}    # M075SH Mg/Si=0.75
    n_msh_13::Vector{Float64}    # M15SH  Mg/Si=1.50
    n_ldh_m4::Vector{Float64}    # M4A-OH-LDH Mg/Al=2
    n_ldh_m6::Vector{Float64}    # M6A-OH-LDH Mg/Al=3
    n_ldh_m8::Vector{Float64}    # M8A-OH-LDH Mg/Al=4
    c_oh_frozen::Vector{Float64} # local OH⁻ after chemistry [mol/m³_water]

    # ── Effective K_d (secant, updated after each chemistry step) ─────────────
    Kd_Cl::Vector{Float64}
    Kd_Na::Vector{Float64}
    Kd_K::Vector{Float64}
    Kd_Ca::Vector{Float64}
    Kd_Mg::Vector{Float64}
    Kd_SO4::Vector{Float64}

    # ── DLM: concentrations adsorbed on C-S-H [mol/m³_concrete] ───────────────
    S_Cl_dlm::Vector{Float64}   # Cl⁻  → ≡SiOHCl⁻
    S_Na_dlm::Vector{Float64}   # Na⁺  → ≡SiONa
    S_K_dlm::Vector{Float64}    # K⁺   → ≡SiOK
    S_Ca_dlm::Vector{Float64}   # Ca²⁺ → ≡SiOCa⁺
    S_Mg_dlm::Vector{Float64}   # Mg²⁺ → ≡SiOMg⁺

    # ── Initial conditions (OPC pore solution, computed by _hydrate_m100) ─────
    c_cl_init::Float64
    c_na_init::Float64
    c_k_init::Float64
    c_ca_init::Float64
    c_mg_init::Float64
    c_so4_init::Float64
    c_si_init::Float64
end

# ── Chemistry initialisation ──────────────────────────────────────────────────

"""
    _init_chemistry_marks2015() -> (cs_hyd, cs_tr, has_friedels, has_brucite, has_msh, has_ldh)

Builds two distinct chemical systems from cemdata18:

  `cs_hyd` — Phase 1 (hydratation) :
    Phases clinker anhydres (C3S, C2S, C3A, C4AF, Gp) + produits d'hydratation
    standards (Portlandite, CSHQ, Ettringite, Monosulfoaluminate).

  `cs_tr` — Phase 3 (reactive transport):
    Hydration products + phases induced by seawater:
      - Friedel (C4AClH10), Brucite
      - M-S-H ss (M075SH Mg/Si=0.75, M15SH Mg/Si=1.50) — cemdata18
      - LDH-OH ss (M4A/M6A/M8A-OH-LDH) — cemdata18
"""
function _init_chemistry_marks2015()
    data_path = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    isfile(data_path) || error("cemdata18 introuvable : $data_path")
    @info "Loading cemdata18…"
    substances = build_species(data_path)
    dict_sp = Dict(symbol(s) => s for s in substances)

    # "SiO2@" is the cemdata18 name of dissolved silicic acid (H₄SiO₄ in PHREEQC).
    seeds = split("Portlandite ettringite monosulphate12 C3S C2S H2O@ Ca+2 OH- Cl- Na+ K+ H+ Mg+2 Al+3 SO4-2 SiO2@")
    aq_species = speciation(substances, seeds;
        aggregate_state=[AS_AQUEOUS],
        exclude_species=EXCLUDE_NONCSHQ,
    )

    # ── Solutions solides CSHQ ────────────────────────────────────────────────
    cshq_found = filter(n -> haskey(dict_sp, n), collect(CSHQ_EM))
    length(cshq_found) == 4 || @warn "CSHQ: only $(length(cshq_found))/4 end-members found"
    ss_cshq = SolidSolutionPhase(
        "CSHQ",
        [dict_sp[em] for em in cshq_found];
        model=IdealSolidSolutionModel(),
    )

    # ── Solution solide M-S-H (M075SH, M15SH — cemdata18) ───────────────────
    msh_found = filter(n -> haskey(dict_sp, n), collect(MSH_EM))
    has_msh = !isempty(msh_found)
    if has_msh
        length(msh_found) < 2 && @warn "M-S-H: only $(length(msh_found))/2 end-members found ($(join(msh_found,", ")))"
        ss_msh = SolidSolutionPhase("MSH", [dict_sp[em] for em in msh_found]; model=IdealSolidSolutionModel())
    else
        @warn "M-S-H missing from cemdata18 (names looked up: $(join(MSH_EM,", "))) — CSHQ decalcification not modelled"
    end

    # ── Solution solide LDH-OH (M4A/M6A/M8A-OH-LDH — cemdata18) ─────────────
    ldh_found = filter(n -> haskey(dict_sp, n), collect(LDH_EM))
    has_ldh = !isempty(ldh_found)
    if has_ldh
        length(ldh_found) < 3 && @warn "LDH-OH: only $(length(ldh_found))/3 end-members found ($(join(ldh_found,", ")))"
        ss_ldh = SolidSolutionPhase("LDH", [dict_sp[em] for em in ldh_found]; model=IdealSolidSolutionModel())
    else
        @warn "LDH-OH missing from cemdata18 (names looked up: $(join(LDH_EM,", "))) — Mg/Al fixation not modelled"
    end

    # ── Hydration system: clinker + standard hydrates (no Friedel/brucite/MSH/LDH) ─
    solid_names_hyd = [
        "C3S", "C2S", "C3A", "C4AF", "Gp",
        cshq_found...,
        "Portlandite", "ettringite", "monosulphate12",
    ]
    solid_hyd = [dict_sp[n] for n in solid_names_hyd if haskey(dict_sp, n)]
    missing_hyd = filter(n -> !haskey(dict_sp, n), solid_names_hyd)
    isempty(missing_hyd) || @warn "Hydratation — phases absentes : $(join(missing_hyd, ", "))"
    cs_hyd = ChemicalSystem(
        vcat(collect(aq_species), solid_hyd), CEMDATA_PRIMARIES;
        solid_solutions=[ss_cshq],
    )

    # ── Transport system: hydrates + seawater phases + M-S-H + LDH ────────────
    has_friedels = haskey(dict_sp, "C4AClH10")
    has_brucite = haskey(dict_sp, "Brucite")
    has_friedels || @warn "C4AClH10 missing — Cl⁻ (Friedel) binding disabled"
    has_brucite || @warn "Brucite missing — Mg(OH)₂ precipitation disabled"
    solid_names_tr = String[
        cshq_found...,
        "Portlandite", "ettringite", "monosulphate12",
        "C4AClH10", "Brucite",
        (has_msh ? msh_found : String[])...,
        (has_ldh ? ldh_found : String[])...,
    ]
    solid_tr = [dict_sp[n] for n in solid_names_tr if haskey(dict_sp, n)]
    missing_tr = filter(n -> !haskey(dict_sp, n), solid_names_tr)
    isempty(missing_tr) || @warn "Transport — phases absentes : $(join(missing_tr, ", "))"

    ss_tr = SolidSolutionPhase[ss_cshq]
    has_msh && push!(ss_tr, ss_msh)
    has_ldh && push!(ss_tr, ss_ldh)

    cs_tr = ChemicalSystem(
        vcat(collect(aq_species), solid_tr), CEMDATA_PRIMARIES;
        solid_solutions=ss_tr,
    )

    @info "Chemical systems" n_aq = length(aq_species) n_solid_hyd = length(solid_hyd) n_solid_tr = length(solid_tr) has_msh has_ldh
    return cs_hyd, cs_tr, has_friedels, has_brucite, has_msh, has_ldh
end

# ── Hydration: computing the initial phase assemblage ─────────────────────────

"""
    _hydrate_m100(cs, mat, env) → NamedTuple

Computes the initial state of the concrete by thermodynamic hydration equilibrium.

Starts from the clinker composition (Bogue phases) + mixing water and calls
equilibrate() to obtain the stable phase assemblage, including the pore
solide CSHQ (TobD, TobH, JenH, JenD).

Parameters:
  cs  : chemical system (includes CSHQ)
  mat : material data (CementMaterial)
  env : exposure conditions — only the temperature T_K is used here

Returns: a NamedTuple with (phi, n_ch, n_ett, n_ms, n_fs, n_brc,
         n_csh_tobh, n_csh_tobd, n_csh_jenh, n_csh_jend,
         n_msh_08, n_msh_13, n_ldh_m4, n_ldh_m6, n_ldh_m8,
         c_ca, c_cl, c_na, c_k, c_oh, c_so4, c_mg, c_si)
"""
function _hydrate_m100(cs, mat::CementMaterial, env::ExposureConditions)
    m_clinker_kgm3 = mat.mix.m_clinker
    wc             = mat.mix.wc
    phi0           = mat.transport.phi0
    T_K            = env.T_K
    na2o_frac      = mat.clinker.na2o_frac
    k2o_frac       = mat.clinker.k2o_frac
    f_C3S          = mat.clinker.f_C3S
    f_C2S          = mat.clinker.f_C2S
    f_C3A          = mat.clinker.f_C3A
    f_C4AF         = mat.clinker.f_C4AF
    f_Gp           = mat.clinker.f_Gp

    # Masses molaires [kg/mol]
    phases_bogue = (
        ("C3S", f_C3S, 0.22831),
        ("C2S", f_C2S, 0.17224),
        ("C3A", f_C3A, 0.27019),
        ("C4AF", f_C4AF, 0.48596),
        ("Gp", f_Gp, 0.17217),
    )

    m_cem = m_clinker_kgm3 * V_REV_M    # kg of cement per REV
    m_eau = wc * m_cem                   # mixing water [kg/REV]
    n_eau = m_eau / 0.018015             # [mol/REV]

    M_Na2O, M_K2O = 61.98e-3, 94.20e-3
    n_na = 2.0 * (m_cem * na2o_frac / M_Na2O)
    n_k = 2.0 * (m_cem * k2o_frac / M_K2O)

    T_q = T_K * us"K"
    state = ChemicalState(cs; T=T_q)

    # Clinker anhydre
    for (name, frac, mw) in phases_bogue
        n = m_cem * frac / mw
        set_quantity!(state, name, n * us"mol")
    end
    # Mixing water + alkalis
    set_quantity!(state, "H2O@", n_eau * us"mol")
    set_quantity!(state, "Na+", n_na * us"mol")
    set_quantity!(state, "K+", n_k * us"mol")
    set_quantity!(state, "OH-", (n_na + n_k) * us"mol")   # initial alkali charge
    set_quantity!(state, "Cl-", 1e-16 * us"mol")
    set_quantity!(state, "Mg+2", 1e-16 * us"mol")

    @info "_hydrate_m100: M100 hydration equilibrium (Bogue + mixing water)…"
    local state_eq
    try
        state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))
    catch e
        error("M100 hydration equilibrate failed: $e")
    end

    V_liq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
    V_liq < 1e-15 && error("Zero liquid volume after the M100 hydration")

    get_n(name) = max(ustrip(moles(state_eq, name)) / V_REV_M, 0.0)
    get_c(name) = ustrip(moles(state_eq, name)) / V_liq

    n_ch = get_n("Portlandite")
    n_ett = get_n("ettringite")
    n_ms = get_n("monosulphate12")
    n_csh_tobh = get_n("CSHQ-TobH")
    n_csh_tobd = get_n("CSHQ-TobD")
    n_csh_jenh = get_n("CSHQ-JenH")
    n_csh_jend = get_n("CSHQ-JenD")

    c_ca = get_c("Ca+2")
    c_cl = max(get_c("Cl-"), 0.0)
    c_na = get_c("Na+")
    c_k = get_c("K+")
    c_oh = get_c("OH-")
    c_so4 = get_c("SO4-2")
    c_mg = get_c("Mg+2")
    # Total dissolved Si: sum over all aqueous Si species
    c_si = sum(n * ustrip(moles(state_eq, sp)) for (sp, n) in SI_AQ_SPECIES
               if ustrip(moles(state_eq, sp)) > 0.0) / V_liq

    n_csh_total = n_csh_tobh + n_csh_tobd + n_csh_jenh + n_csh_jend
    pH = 14.0 + log10(max(c_oh, 1e-20) / 1000.0)
    @info "Hydratation M100 :" pH = round(pH; digits=2) n_CH = round(n_ch; digits=0) n_CSH = round(n_csh_total; digits=0) n_ett = round(n_ett; digits=1) n_ms = round(n_ms; digits=1) c_Si = round(c_si; sigdigits=3)

    # phi0 (measured) is used for the transport — not the computed porosity
    # M-S-H and LDH = 0 initially (they only form on contact with seawater)
    return (;
        phi=phi0,
        n_ch=n_ch,
        n_ett=n_ett,
        n_ms=n_ms,
        n_fs=0.0,
        n_brc=0.0,
        n_csh_tobh=n_csh_tobh,
        n_csh_tobd=n_csh_tobd,
        n_csh_jenh=n_csh_jenh,
        n_csh_jend=n_csh_jend,
        n_msh_08=0.0,
        n_msh_13=0.0,
        n_ldh_m4=0.0,
        n_ldh_m6=0.0,
        n_ldh_m8=0.0,
        c_ca=c_ca,
        c_cl=c_cl,
        c_na=c_na,
        c_k=c_k,
        c_oh=c_oh,
        c_so4=c_so4,
        c_mg=c_mg,
        c_si=c_si,
    )
end

"""
    Marks2015Model(N_nodes, cs_hyd; mat, env, diff, vm, dlm)

Constructor with thermodynamic hydration initialisation (Bogue phases + equilibrate()).

  `mat`  : material data (CementMaterial) — composition, mix, transport
  `env`  : exposure conditions (ExposureConditions) — seawater, temperature
  `diff` : diffusion coefficients (IonicDiffusivities) — Atkins/Oelkers values
  `vm`   : molar volumes (MolarVolumes) — cemdata18
  `dlm`  : DLM constants (DLMConstants) — Tran & Soive 2018
"""
function Marks2015Model(
    N_nodes::Int, cs_hyd;
    mat::CementMaterial      = CementMaterial(),
    env::ExposureConditions  = ExposureConditions(),
    diff::IonicDiffusivities = IonicDiffusivities(),
    vm::MolarVolumes         = MolarVolumes(),
    dlm::DLMConstants        = DLMConstants(),
)
    ic = _hydrate_m100(cs_hyd, mat, env)
    N  = N_nodes

    # ── DLM: initial state (OPC pore solution after hydration) ────────────────
    n_csh_init = ic.n_csh_tobh + ic.n_csh_tobd + ic.n_csh_jenh + ic.n_csh_jend
    x_cas_init = n_csh_init > 1e-6 ? (
        (5 / 6 * ic.n_csh_tobh + 5 / 6 * ic.n_csh_tobd +
         9 / 6 * ic.n_csh_jenh + 10 / 6 * ic.n_csh_jend) / n_csh_init
    ) : 1.5
    n_csh_dlm_init = mat.transport.n_csh0 > 0.0 ? mat.transport.n_csh0 : n_csh_init
    _, S_Cl0, S_Na0, S_K0, S_Ca0, S_Mg0 = solve_dlm_marks(
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca, ic.c_mg, ic.c_oh,
        n_csh_dlm_init, x_cas_init; dlm=dlm, T_K=env.T_K,
    )
    S_Cl0 = max(S_Cl0, 0.0)
    S_Na0 = max(S_Na0, 0.0)
    S_K0  = max(S_K0,  0.0)
    S_Ca0 = max(S_Ca0, 0.0)
    S_Mg0 = max(S_Mg0, 0.0)
    n_csh_dlm_init > 0.0 && @info "DLM initial :" n_csh = round(n_csh_dlm_init; digits=0) x_cas = round(x_cas_init; digits=2) S_Cl = round(S_Cl0; sigdigits=3) S_Na = round(S_Na0; sigdigits=3) S_Mg = round(S_Mg0; sigdigits=3)

    return Marks2015Model(
        0.10,                  # L [m] — 10 cm
        diff, vm, dlm,         # immutable data
        mat, env,              # material and environment data
        fill(ic.phi, N),
        fill(ic.n_ch, N),
        fill(ic.n_ett, N),
        fill(ic.n_ms, N),
        fill(ic.n_fs, N),
        fill(ic.n_brc, N),
        fill(ic.n_csh_tobh, N),
        fill(ic.n_csh_tobd, N),
        fill(ic.n_csh_jenh, N),
        fill(ic.n_csh_jend, N),
        fill(ic.n_msh_08, N),
        fill(ic.n_msh_13, N),
        fill(ic.n_ldh_m4, N),
        fill(ic.n_ldh_m6, N),
        fill(ic.n_ldh_m8, N),
        fill(ic.c_oh, N),
        fill(0.0, N),          # Kd_Cl  — initially zero (no Cl)
        fill(0.0, N),          # Kd_Na
        fill(0.0, N),          # Kd_K
        fill(0.0, N),          # Kd_Ca
        fill(0.0, N),          # Kd_Mg
        fill(0.0, N),          # Kd_SO4
        fill(S_Cl0, N),        # S_Cl_dlm
        fill(S_Na0, N),        # S_Na_dlm
        fill(S_K0,  N),        # S_K_dlm
        fill(S_Ca0, N),        # S_Ca_dlm
        fill(S_Mg0, N),        # S_Mg_dlm
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca, ic.c_mg, ic.c_so4, ic.c_si,
    )
end

PoroMechanics.nspecies(::Marks2015Model) = 7
PoroMechanics.species_names(::Marks2015Model) = [:c_Cl, :c_Na, :c_K, :c_Ca, :c_Mg, :c_SO4, :c_Si]

# ── Fonctions utilitaires ─────────────────────────────────────────────────────

"""
    _tortuosity_m(phi, m::Marks2015Model)

Oh-Jang tortuosity of the cement paste, from the package constitutive layer.
Saturated medium: S_l = 1, so the saturation factor is one.
"""
function _tortuosity_m(phi, m::Marks2015Model)
    tr = m.mat.transport
    oj = OhJang(; phi_c = tr.phi_c, n = tr.n_OJ, ds = tr.ds_OJ, tau_agg = tr.tau_agg)
    return tortuosity(oj, phi, 1)
end

@inline function _node_idx_m(x::Float64, m::Marks2015Model)
    N = length(m.phi) - 1
    return clamp(round(Int, x / (m.L / N)) + 1, 1, N + 1)
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

function PoroMechanics.storage!(f, u, node, m::Marks2015Model, ::Any)
    i = _node_idx_m(node.coord[1], m)
    phi = m.phi[i]
    f[ICL_M] = (phi + m.Kd_Cl[i]) * u[ICL_M]
    f[INA_M] = (phi + m.Kd_Na[i]) * u[INA_M]
    f[IK_M] = (phi + m.Kd_K[i]) * u[IK_M]
    f[ICA_M] = (phi + m.Kd_Ca[i]) * u[ICA_M]
    f[IMG_M] = (phi + m.Kd_Mg[i]) * u[IMG_M]
    f[ISO4_M] = (phi + m.Kd_SO4[i]) * u[ISO4_M]
    f[ISI_M] = phi * u[ISI_M]   # no Si Kd in this model
end

function PoroMechanics.flux!(f, u, edge, m::Marks2015Model, ::Any)
    x_mid = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    i = _node_idx_m(x_mid, m)
    phi_i = (i < length(m.phi)) ? (m.phi[i] + m.phi[i+1]) / 2 : m.phi[i]
    tau = _tortuosity_m(phi_i, m)
    f[ICL_M]  = m.diff.D_Cl  * tau * (u[ICL_M,  1] - u[ICL_M,  2])
    f[INA_M]  = m.diff.D_Na  * tau * (u[INA_M,  1] - u[INA_M,  2])
    f[IK_M]   = m.diff.D_K   * tau * (u[IK_M,   1] - u[IK_M,   2])
    f[ICA_M]  = m.diff.D_Ca  * tau * (u[ICA_M,  1] - u[ICA_M,  2])
    f[IMG_M]  = m.diff.D_Mg  * tau * (u[IMG_M,  1] - u[IMG_M,  2])
    f[ISO4_M] = m.diff.D_SO4 * tau * (u[ISO4_M, 1] - u[ISO4_M, 2])
    f[ISI_M]  = m.diff.D_Si  * tau * (u[ISI_M,  1] - u[ISI_M,  2])
end

function PoroMechanics.bcondition!(f, u, bnode, m::Marks2015Model, ::Any)
    boundary_dirichlet!(f, u, bnode; species=ICL_M,  region=1, value=m.env.c_Cl)
    boundary_dirichlet!(f, u, bnode; species=INA_M,  region=1, value=m.env.c_Na)
    boundary_dirichlet!(f, u, bnode; species=IK_M,   region=1, value=m.env.c_K)
    boundary_dirichlet!(f, u, bnode; species=ICA_M,  region=1, value=m.env.c_Ca)
    boundary_dirichlet!(f, u, bnode; species=IMG_M,  region=1, value=m.env.c_Mg)
    boundary_dirichlet!(f, u, bnode; species=ISO4_M, region=1, value=m.env.c_SO4)
    boundary_dirichlet!(f, u, bnode; species=ISI_M,  region=1, value=m.env.c_Si)
end

# ── SNIA chemical step ────────────────────────────────────────────────────────

"""
    chemistry_step_marks2015!(m, u, cs, has_friedels, has_brucite)

SNIA chemical step — Marks 2015.

For every interior node (i ≥ 2):
  1. Primary charge balance → n_OH.
  2. equilibrate() (Gibbs) → redistribution thermodynamique.
  3. Update: free concentrations, solids, porosity, Kd_Cl (Friedel tangent).

The key reactions with seawater:
  - Mg²⁺ + Ca(OH)₂ → Mg(OH)₂ (brucite) + Ca²⁺  (decalcification)
  - Cl⁻ + monosulphoaluminate → Friedel's salt + SO₄²⁻  (Cl⁻ binding)
  - SO₄²⁻ + Ca(OH)₂ + Al → Ettringite  (sulfatation)
"""
function chemistry_step_marks2015!(
    m::Marks2015Model, u::Matrix, cs,
    has_friedels::Bool, has_brucite::Bool, has_msh::Bool, has_ldh::Bool
)
    N   = size(u, 2)
    T_q = m.env.T_K * us"K"
    vm  = m.vm   # local alias for readability

    ε = 1.0e-15
    # "Seawater has not arrived yet" threshold: 0.1 % of the Dirichlet BCs
    skip_cl  = 1e-3 * m.env.c_Cl
    skip_mg  = 1e-3 * m.env.c_Mg
    skip_so4 = 1e-3 * m.env.c_SO4

    Threads.@threads for i in 2:N
        phi_i = m.phi[i]
        C_Cl = max(u[ICL_M, i], 0.0)
        C_Na = max(u[INA_M, i], 0.0)
        C_K = max(u[IK_M, i], 0.0)
        C_Ca = max(u[ICA_M, i], 0.0)
        C_Mg = max(u[IMG_M, i], 0.0)
        C_SO4 = max(u[ISO4_M, i], 0.0)

        # Untouched node: the front has not arrived and no reactive hydrates formed
        if C_Cl < skip_cl && C_Mg < skip_mg && C_SO4 < skip_so4 &&
           m.n_fs[i] < 1e-8 && m.n_brc[i] < 1e-8
            continue
        end

        C_Si = max(u[ISI_M, i], 0.0)
        n_cl = C_Cl * phi_i * V_REV_M
        n_na = C_Na * phi_i * V_REV_M
        n_k = C_K * phi_i * V_REV_M
        n_ca = C_Ca * phi_i * V_REV_M
        n_mg = C_Mg * phi_i * V_REV_M
        n_so4 = C_SO4 * phi_i * V_REV_M
        n_si = C_Si * phi_i * V_REV_M
        # Electroneutrality: OH⁻ = Na⁺ + K⁺ + 2Ca²⁺ + 2Mg²⁺ - Cl⁻ - 2SO₄²⁻
        n_oh = max(n_na + n_k + 2.0 * n_ca + 2.0 * n_mg - n_cl - 2.0 * n_so4, 1.0e-20)
        n_water = phi_i * V_REV_M * 55_500.0

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
        set_quantity!(state, "Portlandite", m.n_ch[i] * V_REV_M * us"mol")
        set_quantity!(state, "ettringite", m.n_ett[i] * V_REV_M * us"mol")
        set_quantity!(state, "monosulphate12", m.n_ms[i] * V_REV_M * us"mol")
        set_quantity!(state, "CSHQ-TobH", m.n_csh_tobh[i] * V_REV_M * us"mol")
        set_quantity!(state, "CSHQ-TobD", m.n_csh_tobd[i] * V_REV_M * us"mol")
        set_quantity!(state, "CSHQ-JenH", m.n_csh_jenh[i] * V_REV_M * us"mol")
        set_quantity!(state, "CSHQ-JenD", m.n_csh_jend[i] * V_REV_M * us"mol")
        has_friedels && set_quantity!(state, "C4AClH10", m.n_fs[i] * V_REV_M * us"mol")
        has_brucite && set_quantity!(state, "Brucite", m.n_brc[i] * V_REV_M * us"mol")
        if has_msh
            set_quantity!(state, MSH_EM[1], m.n_msh_08[i] * V_REV_M * us"mol")
            length(MSH_EM) > 1 && set_quantity!(state, MSH_EM[2], m.n_msh_13[i] * V_REV_M * us"mol")
        end
        if has_ldh
            set_quantity!(state, LDH_EM[1], m.n_ldh_m4[i] * V_REV_M * us"mol")
            length(LDH_EM) > 1 && set_quantity!(state, LDH_EM[2], m.n_ldh_m6[i] * V_REV_M * us"mol")
            length(LDH_EM) > 2 && set_quantity!(state, LDH_EM[3], m.n_ldh_m8[i] * V_REV_M * us"mol")
        end

        local state_eq
        try
            state_eq = equilibrate(state, OptimaOptimizer(tol=1e-7, verbose=false))
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
        c_mg_new = ustrip(moles(state_eq, "Mg+2")) / V_liq
        c_si_new = sum(n * ustrip(moles(state_eq, sp)) for (sp, n) in SI_AQ_SPECIES
                       if ustrip(moles(state_eq, sp)) > 0.0) / V_liq

        n_ch_new = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_M, 0.0)
        n_ett_new = max(ustrip(moles(state_eq, "ettringite")) / V_REV_M, 0.0)
        n_ms_new = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_M, 0.0)
        n_csh_tobh_new = max(ustrip(moles(state_eq, "CSHQ-TobH")) / V_REV_M, 0.0)
        n_csh_tobd_new = max(ustrip(moles(state_eq, "CSHQ-TobD")) / V_REV_M, 0.0)
        n_csh_jenh_new = max(ustrip(moles(state_eq, "CSHQ-JenH")) / V_REV_M, 0.0)
        n_csh_jend_new = max(ustrip(moles(state_eq, "CSHQ-JenD")) / V_REV_M, 0.0)
        n_fs_new = has_friedels ? max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_M, 0.0) : 0.0
        n_brc_new = has_brucite ? max(ustrip(moles(state_eq, "Brucite")) / V_REV_M, 0.0) : 0.0
        n_msh_08_new = has_msh ? max(ustrip(moles(state_eq, MSH_EM[1])) / V_REV_M, 0.0) : 0.0
        n_msh_13_new = (has_msh && length(MSH_EM) > 1) ? max(ustrip(moles(state_eq, MSH_EM[2])) / V_REV_M, 0.0) : 0.0
        n_ldh_m4_new = has_ldh ? max(ustrip(moles(state_eq, LDH_EM[1])) / V_REV_M, 0.0) : 0.0
        n_ldh_m6_new = (has_ldh && length(LDH_EM) > 1) ? max(ustrip(moles(state_eq, LDH_EM[2])) / V_REV_M, 0.0) : 0.0
        n_ldh_m8_new = (has_ldh && length(LDH_EM) > 2) ? max(ustrip(moles(state_eq, LDH_EM[3])) / V_REV_M, 0.0) : 0.0

        # ── Porosity update ───────────────────────────────────────────────────
        Δn_ch = n_ch_new - m.n_ch[i]
        Δn_ett = n_ett_new - m.n_ett[i]
        Δn_ms = n_ms_new - m.n_ms[i]
        Δn_fs = n_fs_new - m.n_fs[i]
        Δn_brc = n_brc_new - m.n_brc[i]
        Δn_csh_tobh = n_csh_tobh_new - m.n_csh_tobh[i]
        Δn_csh_tobd = n_csh_tobd_new - m.n_csh_tobd[i]
        Δn_csh_jenh = n_csh_jenh_new - m.n_csh_jenh[i]
        Δn_csh_jend = n_csh_jend_new - m.n_csh_jend[i]
        Δn_msh_08 = n_msh_08_new - m.n_msh_08[i]
        Δn_msh_13 = n_msh_13_new - m.n_msh_13[i]
        Δn_ldh_m4 = n_ldh_m4_new - m.n_ldh_m4[i]
        Δn_ldh_m6 = n_ldh_m6_new - m.n_ldh_m6[i]
        Δn_ldh_m8 = n_ldh_m8_new - m.n_ldh_m8[i]
        phi_new = (phi_i
                   - Δn_ch       * vm.Vm_CH
                   - Δn_ett      * vm.Vm_ett
                   - Δn_ms       * vm.Vm_ms
                   - Δn_fs       * vm.Vm_fs
                   - Δn_brc      * vm.Vm_brc
                   - Δn_csh_tobh * vm.Vm_TobH
                   - Δn_csh_tobd * vm.Vm_TobD
                   - Δn_csh_jenh * vm.Vm_JenH
                   - Δn_csh_jend * vm.Vm_JenD
                   - Δn_msh_08   * vm.Vm_MSH_08
                   - Δn_msh_13   * vm.Vm_MSH_13
                   - Δn_ldh_m4   * vm.Vm_LDH_M4
                   - Δn_ldh_m6   * vm.Vm_LDH_M6
                   - Δn_ldh_m8   * vm.Vm_LDH_M8)
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
        m.n_msh_08[i] = n_msh_08_new
        m.n_msh_13[i] = n_msh_13_new
        m.n_ldh_m4[i] = n_ldh_m4_new
        m.n_ldh_m6[i] = n_ldh_m6_new
        m.n_ldh_m8[i] = n_ldh_m8_new
        m.c_oh_frozen[i] = max(c_oh_new, 1e-12)

        # ── Friedel Kd_Cl: tangent through a perturbed Gibbs call ─────────────
        dn_fs_dc = 0.0
        if has_friedels && n_fs_new > 1e-10
            δ_cl = max(1.0, c_cl_new * 1e-2)
            phi_p = m.phi[i]
            n_cl_p = max(c_cl_new + δ_cl, 0.0) * phi_p * V_REV_M
            n_na_p = c_na_new * phi_p * V_REV_M
            n_k_p = c_k_new * phi_p * V_REV_M
            n_ca_p = c_ca_new * phi_p * V_REV_M
            n_mg_p = c_mg_new * phi_p * V_REV_M
            n_so4_p = c_so4_new * phi_p * V_REV_M
            n_si_p = c_si_new * phi_p * V_REV_M
            n_oh_p = max(n_na_p + n_k_p + 2.0 * n_ca_p + 2.0 * n_mg_p - n_cl_p - 2.0 * n_so4_p, 1.0e-20)
            n_fs_p = n_fs_new + δ_cl * n_fs_new / max(c_cl_new, ε)
            try
                state_p = ChemicalState(cs; T=T_q)
                set_quantity!(state_p, "H2O@", phi_p * V_REV_M * 55_500.0 * us"mol")
                set_quantity!(state_p, "OH-", n_oh_p * us"mol")
                set_quantity!(state_p, "Cl-", n_cl_p * us"mol")
                set_quantity!(state_p, "Na+", n_na_p * us"mol")
                set_quantity!(state_p, "K+", n_k_p * us"mol")
                set_quantity!(state_p, "Ca+2", n_ca_p * us"mol")
                set_quantity!(state_p, "Mg+2", n_mg_p * us"mol")
                set_quantity!(state_p, "SO4-2", n_so4_p * us"mol")
                set_quantity!(state_p, "SiO2@", n_si_p * us"mol")
                set_quantity!(state_p, "Portlandite", n_ch_new * V_REV_M * us"mol")
                set_quantity!(state_p, "ettringite", n_ett_new * V_REV_M * us"mol")
                set_quantity!(state_p, "monosulphate12", n_ms_new * V_REV_M * us"mol")
                set_quantity!(state_p, "CSHQ-TobH", n_csh_tobh_new * V_REV_M * us"mol")
                set_quantity!(state_p, "CSHQ-TobD", n_csh_tobd_new * V_REV_M * us"mol")
                set_quantity!(state_p, "CSHQ-JenH", n_csh_jenh_new * V_REV_M * us"mol")
                set_quantity!(state_p, "CSHQ-JenD", n_csh_jend_new * V_REV_M * us"mol")
                set_quantity!(state_p, "C4AClH10", n_fs_new * V_REV_M * us"mol")
                has_brucite && set_quantity!(state_p, "Brucite", n_brc_new * V_REV_M * us"mol")
                if has_msh
                    set_quantity!(state_p, MSH_EM[1], n_msh_08_new * V_REV_M * us"mol")
                    length(MSH_EM) > 1 && set_quantity!(state_p, MSH_EM[2], n_msh_13_new * V_REV_M * us"mol")
                end
                if has_ldh
                    set_quantity!(state_p, LDH_EM[1], n_ldh_m4_new * V_REV_M * us"mol")
                    length(LDH_EM) > 1 && set_quantity!(state_p, LDH_EM[2], n_ldh_m6_new * V_REV_M * us"mol")
                    length(LDH_EM) > 2 && set_quantity!(state_p, LDH_EM[3], n_ldh_m8_new * V_REV_M * us"mol")
                end
                state_p_eq = equilibrate(state_p, OptimaOptimizer(tol=1e-7, verbose=false))
                V_liq_p = ustrip(uconvert(us"m^3", state_p_eq.V_phases[].liquid))
                if V_liq_p > 1e-15
                    n_fs_p = max(ustrip(moles(state_p_eq, "C4AClH10")) / V_REV_M, 0.0)
                end
            catch
                # keep n_fs_p = the secant value
            end
            dn_fs_dc = max(n_fs_p - n_fs_new, 0.0) / δ_cl
        end

        # ── DLM surface complexation (Tran 2018, Mg²⁺ extension) ─────────────
        n_csh_i = n_csh_tobh_new + n_csh_tobd_new + n_csh_jenh_new + n_csh_jend_new
        x_cas_i = n_csh_i > 1e-6 ? (
            (5 / 6 * n_csh_tobh_new + 5 / 6 * n_csh_tobd_new + 9 / 6 * n_csh_jenh_new + 10 / 6 * n_csh_jend_new) / n_csh_i
        ) : 1.5
        dlm_i     = m.dlm
        n_csh_dlm = m.mat.transport.n_csh0 > 0.0 ? m.mat.transport.n_csh0 : n_csh_i
        dS_Cl_dc  = 0.0
        if n_csh_dlm > 0.0
            _, S_Cl_i, S_Na_i, S_K_i, S_Ca_i, S_Mg_i = solve_dlm_marks(
                max(c_cl_new, 0.0), max(c_na_new, 0.0), max(c_k_new, 0.0),
                max(c_ca_new, 0.0), max(c_mg_new, 0.0), max(c_oh_new, ε),
                n_csh_dlm, x_cas_i; dlm=dlm_i, T_K=m.env.T_K,
            )
            m.S_Cl_dlm[i] = max(S_Cl_i, 0.0)
            m.S_Na_dlm[i] = max(S_Na_i, 0.0)
            m.S_K_dlm[i]  = max(S_K_i,  0.0)
            m.S_Ca_dlm[i] = max(S_Ca_i, 0.0)
            m.S_Mg_dlm[i] = max(S_Mg_i, 0.0)
            δ_dlm = max(1.0e-3, c_cl_new * 1e-3)
            _, S_Cl_p, _, _, _, _ = solve_dlm_marks(
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
        m.Kd_SO4[i] = 0.0

        # ── Free concentrations after Gibbs ───────────────────────────────────
        u[ICL_M, i] = max(c_cl_new, 0.0)
        u[INA_M, i] = max(c_na_new, 0.0)
        u[IK_M, i] = max(c_k_new, 0.0)
        u[ICA_M, i] = max(c_ca_new, 0.0)
        u[IMG_M, i] = max(c_mg_new, 0.0)
        u[ISO4_M, i] = max(c_so4_new, 0.0)
        u[ISI_M, i] = max(c_si_new, 0.0)
    end
end

# ── SNIA solve ────────────────────────────────────────────────────────────────

"""
    run_Marks2015(; N, t_end, n_save, verbose, kwargs...) -> (results, model)

SNIA Marks 2015: Fick transport (6 species) + reactive Gibbs.

Returns:
- `results` : `[(t, u, phi, n_ch, n_ett, n_ms, n_fs, n_brc, c_oh), …]`
- `model`   : the final `Marks2015Model` state

Main parameters:
  N      : number of elements [100]
  t_end  : simulated time [s, 3.1536e7 = 1 year]
  n_save : number of outputs [12]
"""
function run_Marks2015(;
    N=100,
    t_end=3.1536e7,
    n_save=12,
    verbose=false,
    mat::CementMaterial      = CementMaterial(),
    env::ExposureConditions  = ExposureConditions(),
    diff::IonicDiffusivities = IonicDiffusivities(),
    vm::MolarVolumes         = MolarVolumes(),
    dlm::DLMConstants        = DLMConstants(),
)
    @info "Chemistry parallelism" n_threads = Threads.nthreads()

    # cs_hyd: clinker + hydrates — for the hydration computation (Phase 1)
    # cs_tr : hydrates + Friedel/brucite/M-S-H/LDH — for the reactive transport (Phase 3)
    cs_hyd, cs_tr, has_friedels, has_brucite, has_msh, has_ldh = _init_chemistry_marks2015()

    # Transport diagnostic (Cl⁻ D_app at the initial porosity)
    let tr = mat.transport
        oj = OhJang(; phi_c = tr.phi_c, n = tr.n_OJ, ds = tr.ds_OJ, tau_agg = tr.tau_agg)
        tau = tortuosity(oj, tr.phi0, 1)
        D_eff = diff.D_Cl * tau
        D_app = D_eff * tr.phi0 / (tr.phi0 + 1.0)   # Kd ≈ 1 (Marks 2015 value)
        println("── Initial Cl⁻ transport values (phi₀ = $(tr.phi0)) ────────────────────")
        println("   tau    = $tau  (tortuosity [-])")
        println("   D_eff  = $D_eff  [m²/s]")
        println("   D_app  = $D_app  [m²/s]")
        println("────────────────────────────────────────────────────────────────────────")
        @info "Transport M100 (phi=$(tr.phi0))" tau = round(tau; sigdigits=3) D_eff_Cl = round(D_eff; sigdigits=3) D_app_Cl = round(D_app; sigdigits=3)
    end

    m    = Marks2015Model(N + 1, cs_hyd; mat=mat, env=env, diff=diff, vm=vm, dlm=dlm)
    grid = simplexgrid(range(0.0, m.L; length=N + 1))

    _storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data)    = PoroMechanics.flux!(f, u, edge, m, data)
    _bcond!(f, u, bnode, data)  = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage=_storage!,
        flux=_flux!,
        bcondition=_bcond!,
        species=[ICL_M, INA_M, IK_M, ICA_M, IMG_M, ISO4_M, ISI_M],
    )

    inival = unknowns(sys)
    inival[ICL_M, :]  .= m.c_cl_init
    inival[INA_M, :]  .= m.c_na_init
    inival[IK_M,  :]  .= m.c_k_init
    inival[ICA_M, :]  .= m.c_ca_init
    inival[IMG_M, :]  .= m.c_mg_init
    inival[ISO4_M, :] .= m.c_so4_init
    inival[ISI_M, :]  .= m.c_si_init
    # Exposed face: seawater
    inival[ICL_M,  1] = m.env.c_Cl
    inival[INA_M,  1] = m.env.c_Na
    inival[IK_M,   1] = m.env.c_K
    inival[ICA_M,  1] = m.env.c_Ca
    inival[IMG_M,  1] = m.env.c_Mg
    inival[ISO4_M, 1] = m.env.c_SO4
    inival[ISI_M,  1] = m.env.c_Si

    ctrl = VoronoiFVM.SolverControl(;
        Δt=1.0,
        Δt_max=t_end / (4 * n_save),
        Δu_opt=0.5 * m.env.c_Cl,
        handle_exceptions=true,
        verbose=verbose,
    )

    tsave = range(0.0, t_end; length=n_save + 1)
    results = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    u_cur = copy(inival)

    for k in 2:lastindex(tsave)
        t0, t1 = tsave[k-1], tsave[k]
        @info "Transport seg $k" t_yr = round(t1 / 3.1536e7; digits=2)

        seg = solve(sys; inival=u_cur, times=[t0, t1], control=ctrl)
        u_cur = Matrix(seg[:, :, end])

        t_chem = @elapsed chemistry_step_marks2015!(m, u_cur, cs_tr, has_friedels, has_brucite, has_msh, has_ldh)
        @info "Chimie" seg = k - 1 t_chem_s = round(t_chem; digits=1) n_threads = Threads.nthreads()

        # Re-apply the seawater BCs (node 1 = exposed face)
        u_cur[ICL_M,  1] = m.env.c_Cl
        u_cur[INA_M,  1] = m.env.c_Na
        u_cur[IK_M,   1] = m.env.c_K
        u_cur[ICA_M,  1] = m.env.c_Ca
        u_cur[IMG_M,  1] = m.env.c_Mg
        u_cur[ISO4_M, 1] = m.env.c_SO4
        u_cur[ISI_M,  1] = m.env.c_Si
        m.c_oh_frozen[1] = 1.0    # free value at the exposed face
        m.Kd_Cl[1] = 0.0
        m.Kd_Mg[1] = 0.0
        m.Kd_SO4[1] = 0.0
        m.S_Cl_dlm[1] = 0.0
        m.S_Na_dlm[1] = 0.0
        m.S_K_dlm[1] = 0.0
        m.S_Ca_dlm[1] = 0.0
        m.S_Mg_dlm[1] = 0.0

        any(isnan, u_cur) && @warn "NaN after chemistry, seg $k"

        push!(results, (
            t1,               # [1]
            copy(u_cur),      # [2] concentrations (7 species × N nodes)
            copy(m.phi),      # [3]
            copy(m.n_ch),     # [4] Portlandite
            copy(m.n_ett),    # [5] Ettringite
            copy(m.n_ms),     # [6] Monosulfoaluminate
            copy(m.n_fs),     # [7] Friedel's salt
            copy(m.n_brc),    # [8] Brucite
            copy(m.c_oh_frozen), # [9] frozen OH⁻ → pH
            copy(m.n_csh_tobh),  # [10] CSHQ-TobH
            copy(m.n_csh_tobd),  # [11] CSHQ-TobD
            copy(m.n_csh_jenh),  # [12] CSHQ-JenH
            copy(m.n_csh_jend),  # [13] CSHQ-JenD
            copy(m.n_msh_08),    # [14] M-S-H (Mg/Si=0.8)
            copy(m.n_msh_13),    # [15] M-S-H (Mg/Si=1.3)
            copy(m.n_ldh_m4),    # [16] M4A-OH-LDH (Mg/Al=2)
            copy(m.n_ldh_m6),    # [17] M6A-OH-LDH (Mg/Al=3)
            copy(m.n_ldh_m8),    # [18] M8A-OH-LDH (Mg/Al=4)
            copy(m.S_Cl_dlm),    # [19] DLM-adsorbed Cl [mol/m³_concrete]
            copy(m.S_Na_dlm),    # [20] DLM-adsorbed Na
            copy(m.S_K_dlm),     # [21] DLM-adsorbed K
            copy(m.S_Ca_dlm),    # [22] DLM-adsorbed Ca
            copy(m.S_Mg_dlm),    # [23] DLM-adsorbed Mg
        ))

        @info "SNIA" seg = k - 1 φ_mean = round(mean(m.phi); sigdigits=4) n_FS0 = round(m.n_fs[2]; sigdigits=4) n_BRC0 = round(m.n_brc[2]; sigdigits=4)

        let phi_vec = m.phi, tr = m.mat.transport, D_Cl = m.diff.D_Cl
            oj_diag = OhJang(; phi_c = tr.phi_c, n = tr.n_OJ, ds = tr.ds_OJ, tau_agg = tr.tau_agg)
            N_nodes = length(phi_vec)
            t_yr_k  = round(t1 / 3.1536e7; digits=2)
            println("   ── Cl⁻ transport  [seg $(k-1), t = $t_yr_k yr] ─────────────────────────────")
            @printf("   %-8s  %-10s  %-10s  %-14s  %-14s\n", "x [cm]", "phi", "tau", "D_eff [m²/s]", "D_app [m²/s]")
            step = max(1, (N_nodes - 1) ÷ 10)
            for i in 1:step:N_nodes
                phi_i   = phi_vec[i]
                tau_i   = tortuosity(oj_diag, phi_i, 1)
                D_eff_i = D_Cl * tau_i
                D_app_i = D_eff_i * phi_i / (phi_i + 1.0)
                x_cm    = (i - 1) * m.L / (N_nodes - 1) * 100
                @printf("   %6.1f    %.6f    %.6f    %.4e      %.4e\n", x_cm, phi_i, tau_i, D_eff_i, D_app_i)
            end
            println("   ─────────────────────────────────────────────────────────────────────────────")
        end
    end

    return results, m
end

# ── TOUGHREACT reference data (R_38_Marks2015, Thermoddem 2023) ───────────────
#
# Source : solid.out, zones t = 0.25 / 0.50 / 0.75 / 1.00 yr.
# x_dm   : 0.00, 0.01, …, 1.00 dm (101 points, 1 mm spacing).
# Units  : volume fractions (TOUGHREACT) converted to mol/m³_concrete.
#   Portlandite  n_ch  — Vm = 33.06e-6 m³/mol
#   Monosulfoal. n_ms  — Vm = 309e-6   m³/mol
#   Brucite      n_brc — Vm = 24.63e-6 m³/mol
# Note   : Friedel = 0 everywhere (TOUGHREACT has no surface complexation).
#          Chrysotile (M-S-H) is significant at x = 0–1 mm (7–8 % vol) but is
#          not modelled here, for lack of explicit C-S-H tracking.

"""
    toughreact_ref() → NamedTuple

Returns the TOUGHREACT reference profiles (Marks 2015, M100 + seawater)
at t = 0.25, 0.50, 0.75, 1.00 yr.  Fields:

  `t_yr`  – vector of the 4 times [yr]
  `x_dm`  – abscisses [dm], 101 points, 0 → 1 dm
  `phi`   – vector of 4 vectors φ [-]
  `n_ch`  – vector of 4 vectors portlandite [mol/m³_concrete]
  `n_ms`  – vector of 4 vectors monosulphoaluminate [mol/m³_concrete]
  `n_brc` – vector of 4 vectors brucite [mol/m³_concrete]
  `cl`    – vector of 4 vectors Cl⁻ [mol/m³_solution]
  `na`    – vector of 4 vectors Na⁺ [mol/m³_solution]
  `k`     – vector of 4 vectors K⁺  [mol/m³_solution]
  `ca`    – vector of 4 vectors Ca²⁺ [mol/m³_solution]
  `so4`   – vector of 4 vectors SO₄²⁻ [mol/m³_solution]
  `mg`    – vector of 4 vectors Mg²⁺ [mol/m³_solution]
  `pH`    – vector of 4 vectors pH [-]

Ionic source: `plot_365.25.xls` (Thermoddem), resampled on the Julia grid.
"""
function toughreact_ref()
    x_dm = collect(range(0.0, 1.0; length=101))

    # ── Porosity φ [-] ────────────────────────────────────────────────────────
    phi_025 = vcat(Float64[
            0.15998, 0.18930, 0.17612, 0.17779, 0.18214, 0.18211,
            0.18181, 0.18166, 0.18151, 0.18144, 0.18143, 0.18143,
            0.18145, 0.18153, 0.18168, 0.18178, 0.18179, 0.18180,
            0.18180, 0.18180, 0.18181, 0.18181, 0.18181, 0.18181,
            0.18181,
        ], fill(0.18181, 76))

    phi_050 = vcat(Float64[
            0.15897, 0.15164, 0.16994, 0.17121, 0.17733, 0.18176,
            0.18159, 0.18141, 0.18118, 0.18101, 0.18091, 0.18085,
            0.18082, 0.18082, 0.18082, 0.18084, 0.18087, 0.18100,
            0.18127, 0.18143, 0.18154, 0.18157, 0.18158, 0.18159,
            0.18159, 0.18160, 0.18160, 0.18160, 0.18160, 0.18160,
            0.18160, 0.18160,
        ], fill(0.18161, 69))

    phi_075 = vcat(Float64[
            0.15796, 0.13309, 0.16969, 0.16558, 0.16947, 0.18101,
            0.18119, 0.18107, 0.18084, 0.18065, 0.18050, 0.18039,
            0.18034, 0.18029, 0.18027, 0.18027, 0.18028, 0.18030,
            0.18032, 0.18035, 0.18061, 0.18096, 0.18121, 0.18127,
            0.18134, 0.18138, 0.18138, 0.18139, 0.18139, 0.18140,
            0.18140, 0.18140, 0.18140,
        ], fill(0.18141, 68))

    phi_100 = vcat(Float64[
            0.15627, 0.12828, 0.19308, 0.16084, 0.16453, 0.18013,
            0.18073, 0.18068, 0.18051, 0.18032, 0.18016, 0.18002,
            0.17995, 0.17987, 0.17982, 0.17979, 0.17979, 0.17979,
            0.17980, 0.17982, 0.17984, 0.17989, 0.18029, 0.18056,
            0.18086, 0.18100, 0.18106, 0.18114, 0.18120, 0.18121,
            0.18121, 0.18121, 0.18122, 0.18122, 0.18122, 0.18122,
            0.18122,
        ], fill(0.18123, 64))

    # ── Portlandite Ca(OH)₂ [mol/m³_concrete] — Vm = 33.06e-6 m³/mol ───────────
    n_ch_025 = vcat(Float64[
            0.0, 0.0, 1457.0, 1435.9, 1477.0, 1492.4, 1503.0, 1508.2, 1513.9,
            1517.2, 1519.4, 1520.9, 1521.5, 1523.0, 1524.5, 1523.9, 1523.6, 1523.3,
            1523.3, 1523.0, 1523.0, 1523.0, 1523.0, 1523.0, 1523.0,
        ], fill(1523.0, 76))

    n_ch_050 = vcat(Float64[
            0.0, 0.0, 1422.3, 1411.1, 1435.9, 1471.9, 1481.2, 1487.6, 1495.8, 1502.1,
            1506.7, 1510.3, 1512.1, 1514.2, 1516.0, 1517.2, 1518.1, 1520.6, 1523.9, 1524.8,
            1523.3, 1522.1, 1521.8, 1521.8, 1521.5, 1521.5, 1521.5, 1521.5, 1521.2, 1521.2,
            1521.2, 1521.2,
        ], fill(1521.2, 69))

    n_ch_075 = vcat(Float64[
            0.0, 0.0, 1246.5, 1394.7, 1384.5, 1463.4, 1469.4, 1474.0, 1481.5, 1488.8,
            1494.6, 1499.4, 1502.1, 1505.4, 1508.2, 1510.3, 1511.5, 1513.0, 1514.5, 1515.4,
            1520.0, 1524.5, 1525.4, 1523.9, 1521.8, 1520.6, 1520.3, 1520.3, 1520.0, 1520.0,
            1520.0, 1520.0,
        ], fill(1519.7, 69))

    n_ch_100 = vcat(Float64[
            0.0, 0.0, 386.3, 1380.8, 1358.7, 1458.9, 1462.5, 1465.5, 1471.6, 1478.2,
            1484.0, 1489.4, 1492.7, 1496.7, 1500.3, 1503.0, 1504.8, 1507.0, 1508.8, 1510.0,
            1511.2, 1513.3, 1520.0, 1523.9, 1526.9, 1524.8, 1523.3, 1520.9, 1519.1, 1518.8,
            1518.8, 1518.8, 1518.5, 1518.5, 1518.5, 1518.5, 1518.5,
        ], fill(1518.1, 64))

    # ── Monosulfoaluminate [mol/m³_concrete] — Vm = 309e-6 m³/mol ──────────────
    n_ms_025 = vcat(Float64[
            91.6, 91.6, 91.7, 92.5, 93.4, 94.1, 95.0, 95.6, 96.5, 97.4, 98.4, 99.3,
            99.8, 100.6, 101.3, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5, 101.5,
            101.5, 101.5, 101.5,
        ], fill(101.6, 76))

    n_ms_050 = vcat(Float64[
            75.8, 75.8, 75.9, 76.6, 77.8, 78.5, 79.4, 79.9, 80.9, 81.8,
            82.8, 83.9, 84.6, 85.8, 86.9, 88.0, 88.6, 89.6, 90.6, 91.0,
        ], fill(91.1, 81))

    n_ms_075 = vcat(Float64[
            62.7, 62.7, 62.8, 63.4, 64.6, 65.3, 66.1, 66.7, 67.5, 68.4,
            69.4, 70.5, 71.2, 72.3, 73.5, 74.7, 75.5, 76.8, 77.9, 78.6,
            79.8, 80.7, 81.2, 81.3,
        ], fill(81.3, 77))

    n_ms_100 = vcat(Float64[
            51.9, 51.9, 52.0, 52.5, 53.5, 54.4, 55.0, 55.5, 56.3, 57.2,
            58.1, 59.0, 59.6, 60.7, 61.8, 63.0, 63.9, 65.1, 66.4, 67.2,
            68.4, 69.5, 70.7, 71.4, 72.0, 72.2, 72.2,
        ], fill(72.2, 74))

    # ── Brucite Mg(OH)₂ [mol/m³_concrete] — Vm = 24.63e-6 m³/mol ──────────────
    # Confined to the first 2 cells (x = 0 and 1 mm) — M-S-H dominates beyond
    # that in TOUGHREACT (not reproduced here).
    n_brc_025 = vcat([2668.3, 1651.6], zeros(99))
    n_brc_050 = vcat([2442.1, 3348.8], zeros(99))
    n_brc_075 = vcat([2184.7, 4238.7], zeros(99))
    n_brc_100 = vcat([1932.6, 4543.2], zeros(99))

    # ── Ioniques [mol/m³_solution] — source : plot_365.25.xls ────────────────
    # Linear resampling (250 non-uniform cells → 101 points over 0-1 dm).
    # Value at x=0: TOUGHREACT boundary node (x=-0.01 mm, seawater).

    # Cl⁻
    cl_025 = vcat(Float64[
            545.10, 474.17, 403.02, 338.33, 282.32, 233.05, 189.67, 151.66, 118.60, 90.41,
            66.83, 47.72, 32.57, 21.61, 14.55, 9.92, 6.55, 4.21, 2.60, 1.56,
            0.91, 0.52, 0.28, 0.15, 0.08, 0.04, 0.02, 0.01,
        ], fill(0.00, 73))
    cl_050 = vcat(Float64[
            545.10, 489.79, 437.90, 385.47, 338.31, 296.41, 258.08, 223.08, 191.09, 161.95,
            135.72, 112.27, 91.09, 72.61, 56.66, 42.92, 31.44, 22.75, 16.88, 13.15,
            10.30, 7.94, 6.01, 4.47, 3.28, 2.36, 1.68, 1.18, 0.81, 0.55,
            0.37, 0.24, 0.16, 0.10, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01,
        ], fill(0.00, 61))
    cl_075 = vcat(Float64[
            545.10, 498.10, 455.03, 407.48, 364.35, 326.74, 292.00, 259.63, 229.64, 201.72,
            175.98, 152.34, 130.34, 110.49, 92.48, 76.20, 61.60, 48.91, 37.72, 28.38,
            21.38, 16.89, 13.95, 11.62, 9.56, 7.77, 6.26, 5.00, 3.93, 3.07,
            2.37, 1.81, 1.37, 1.03, 0.76, 0.56, 0.41, 0.30, 0.21, 0.15,
            0.10, 0.07, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01,
        ], fill(0.00, 53))
    cl_100 = vcat(Float64[
            545.10, 499.57, 463.74, 420.57, 380.38, 345.55, 313.51, 283.25, 254.90, 228.36,
            203.42, 180.23, 158.34, 138.20, 119.64, 102.49, 86.65, 72.46, 59.51, 47.99,
            37.75, 29.04, 22.44, 18.09, 15.31, 13.20, 11.28, 9.58, 8.05, 6.72,
            5.57, 4.59, 3.75, 3.04, 2.44, 1.95, 1.55, 1.22, 0.95, 0.74,
            0.57, 0.44, 0.33, 0.25, 0.19, 0.14, 0.10, 0.08, 0.05, 0.04,
            0.03, 0.02, 0.01, 0.01, 0.01,
        ], fill(0.00, 46))

    # Na⁺
    na_025 = vcat(Float64[
            458.20, 434.18, 410.03, 385.49, 361.12, 336.91, 313.30, 290.79, 270.19, 251.72,
            235.71, 222.27, 211.10, 202.17, 195.41, 190.33, 186.51, 183.72, 181.68, 180.28,
            179.30, 178.66, 178.22, 177.90, 177.80, 177.70, 177.70, 177.70, 177.70, 177.70,
            177.70, 177.70, 177.70, 177.74,
        ], fill(177.80, 67))
    na_050 = vcat(Float64[
            458.20, 439.24, 421.91, 403.34, 385.36, 368.01, 350.67, 333.36, 316.28, 299.86,
            284.14, 269.48, 255.79, 243.44, 232.43, 222.58, 213.95, 206.74, 200.61, 195.62,
            191.60, 188.28, 185.56, 183.36, 181.56, 180.10, 178.98, 178.09, 177.33, 176.81,
            176.44, 176.18, 175.92, 175.76, 175.69, 175.60, 175.50, 175.50, 175.50, 175.50,
            175.50, 175.50, 175.50, 175.50, 175.50, 175.50, 175.58,
        ], fill(175.60, 54))
    na_075 = vcat(Float64[
            458.20, 441.49, 427.43, 411.40, 395.78, 381.22, 366.79, 352.37, 337.96, 323.68,
            309.78, 296.40, 283.32, 271.04, 259.63, 248.92, 239.05, 230.13, 221.98, 214.76,
            208.40, 203.00, 198.39, 194.53, 191.20, 188.31, 185.76, 183.65, 181.86, 180.31,
            179.00, 177.91, 177.05, 176.27, 175.69, 175.17, 174.78, 174.42, 174.16, 174.01,
            173.85, 173.70, 173.64, 173.60, 173.50, 173.50, 173.50, 173.50, 173.50, 173.40,
        ], fill(173.50, 51))
    na_100 = vcat(Float64[
            458.20, 442.71, 430.77, 416.14, 401.92, 388.91, 376.39, 363.76, 351.13, 338.51,
            326.01, 313.77, 301.76, 290.13, 279.10, 268.42, 258.33, 249.05, 240.26, 232.17,
            224.80, 218.13, 212.11, 206.80, 202.23, 198.25, 194.74, 191.75, 189.06, 186.62,
            184.52, 182.56, 180.89, 179.48, 178.18, 177.14, 176.21, 175.35, 174.68, 174.11,
            173.60, 173.19, 172.88, 172.58, 172.33, 172.18, 172.02, 171.90, 171.80, 171.70,
            171.70, 171.60, 171.60, 171.60, 171.60, 171.50, 171.50, 171.50, 171.50, 171.50,
            171.50, 171.50,
        ], fill(171.60, 39))

    # K⁺
    k_025 = vcat(Float64[
            9.69, 22.95, 35.83, 48.47, 60.25, 71.21, 81.30, 90.50, 98.80, 106.22,
            112.93, 118.97, 124.57, 129.60, 134.28, 138.62, 142.61, 146.04, 149.15, 151.77,
            154.00, 155.81, 157.39, 158.62, 159.59, 160.32, 160.91, 161.37, 161.73, 162.00,
            162.16, 162.30, 162.40, 162.50, 162.50, 162.56,
        ], fill(162.60, 65))
    k_050 = vcat(Float64[
            9.69, 19.59, 28.51, 37.89, 46.77, 55.05, 62.97, 70.53, 77.66, 84.35,
            90.59, 96.34, 101.75, 106.71, 111.28, 115.62, 119.73, 123.51, 127.17, 130.66,
            134.00, 137.05, 139.98, 142.55, 144.96, 147.12, 149.04, 150.71, 152.27, 153.59,
            154.74, 155.69, 156.55, 157.23, 157.91, 158.36, 158.82, 159.18, 159.44, 159.69,
            159.85, 160.01, 160.16, 160.30, 160.37, 160.40, 160.47, 160.50, 160.50, 160.53,
        ], fill(160.60, 51))
    k_075 = vcat(Float64[
            9.69, 18.22, 25.23, 33.13, 40.69, 47.56, 54.18, 60.62, 66.81, 72.75,
            78.40, 83.72, 88.81, 93.57, 98.02, 102.22, 106.23, 109.92, 113.47, 116.86,
            120.10, 123.15, 126.22, 129.16, 131.93, 134.59, 137.04, 139.25, 141.34, 143.28,
            145.02, 146.64, 148.11, 149.38, 150.62, 151.66, 152.57, 153.45, 154.18, 154.79,
            155.40, 155.92, 156.33, 156.72, 157.04, 157.32, 157.57, 157.73, 157.90, 158.03,
            158.18, 158.30, 158.38, 158.40, 158.50, 158.52, 158.60, 158.60, 158.60,
        ], fill(158.70, 42))
    k_100 = vcat(Float64[
            9.69, 17.31, 23.08, 30.12, 36.95, 43.05, 48.85, 54.54, 60.06, 65.40,
            70.56, 75.47, 80.22, 84.72, 88.97, 93.03, 96.90, 100.52, 104.01, 107.26,
            110.40, 113.45, 116.32, 119.15, 121.93, 124.59, 127.25, 129.72, 132.11, 134.28,
            136.32, 138.26, 139.99, 141.67, 143.13, 144.52, 145.82, 147.01, 148.05, 149.08,
            149.95, 150.82, 151.53, 152.15, 152.74, 153.35, 153.75, 154.23, 154.58, 154.93,
            155.18, 155.43, 155.68, 155.82, 156.00, 156.12, 156.30, 156.40, 156.50, 156.60,
            156.60, 156.70, 156.70, 156.80, 156.80, 156.80, 156.90, 156.90, 156.90, 156.90,
            156.90, 156.90, 156.92,
        ], fill(157.00, 28))

    # Ca²⁺
    ca_025 = vcat(Float64[
            9.9530, 52.0057, 28.0186, 15.9411, 11.2263, 9.0955, 8.0679, 7.6285, 7.5441, 7.7007,
            8.0331, 8.5027, 9.1084, 9.8528, 10.7262, 11.5000, 12.1008, 12.5684, 12.9414, 13.2238,
            13.4300, 13.5774, 13.6814, 13.7611, 13.8073, 13.8439, 13.8705, 13.8868, 13.9000, 13.9100,
            13.9200, 13.9200, 13.9300, 13.9300, 13.9300, 13.9300, 13.9300, 13.9300,
        ], fill(13.9400, 63))
    ca_050 = vcat(Float64[
            9.9530, 51.6527, 37.3519, 23.1148, 15.8159, 12.1509, 9.9599, 8.6322, 7.8355, 7.3779,
            7.1458, 7.0736, 7.1240, 7.2724, 7.5062, 7.8183, 8.2091, 8.7239, 9.4305, 10.2103,
            10.8200, 11.2458, 11.6154, 11.9154, 12.1757, 12.3876, 12.5626, 12.6974, 12.8137, 12.8992,
            12.9721, 13.0323, 13.0768, 13.1089, 13.1400, 13.1564, 13.1723, 13.1882, 13.2000, 13.2095,
            13.2151, 13.2200, 13.2200, 13.2300, 13.2300, 13.2300, 13.2375,
        ], fill(13.2400, 54))
    ca_075 = vcat(Float64[
            9.9530, 48.8322, 42.4413, 27.3873, 18.8305, 14.5252, 11.7448, 9.8992, 8.6785, 7.8675,
            7.3350, 6.9973, 6.7982, 6.7111, 6.7138, 6.7887, 6.9277, 7.1252, 7.3812, 7.7047,
            8.1820, 8.8387, 9.5746, 10.1797, 10.6049, 10.8855, 11.1347, 11.3547, 11.5439, 11.7184,
            11.8603, 11.9815, 12.0937, 12.1779, 12.2515, 12.3128, 12.3646, 12.4164, 12.4477, 12.4795,
            12.5051, 12.5208, 12.5400, 12.5518, 12.5670, 12.5723, 12.5800, 12.5900, 12.5977, 12.6000,
            12.6077, 12.6100, 12.6100, 12.6124, 12.6200, 12.6200, 12.6200, 12.6200, 12.6200, 12.6200,
            12.6200, 12.6200, 12.6200, 12.6200,
        ], fill(12.6300, 37))
    ca_100 = vcat(Float64[
            9.9530, 46.0714, 45.3280, 30.5032, 21.1758, 16.4502, 13.3033, 11.0929, 9.5604, 8.4923,
            7.7458, 7.2254, 6.8634, 6.6271, 6.4893, 6.4272, 6.4303, 6.4901, 6.6021, 6.7615,
            6.9660, 7.2300, 7.6433, 8.2246, 8.9206, 9.5645, 10.0679, 10.4053, 10.6005, 10.7584,
            10.9103, 11.0515, 11.1821, 11.2979, 11.4021, 11.4956, 11.5769, 11.6446, 11.7077, 11.7590,
            11.8051, 11.8415, 11.8762, 11.9035, 11.9270, 11.9523, 11.9700, 11.9828, 11.9977, 12.0100,
            12.0200, 12.0300, 12.0376, 12.0400, 12.0500, 12.0500, 12.0563, 12.0600, 12.0600, 12.0600,
            12.0700, 12.0700, 12.0700, 12.0700, 12.0700, 12.0700, 12.0793,
        ], fill(12.0800, 34))

    # SO₄²⁻
    so4_025 = vcat(Float64[
            27.5500, 15.3649, 5.3875, 0.9447, 0.2777, 0.2628, 0.2531, 0.2310, 0.2012, 0.1688,
            0.1367, 0.1054, 0.0728, 0.0478, 0.0439, 0.0451, 0.0426, 0.0409, 0.0397, 0.0389,
            0.0383, 0.0380, 0.0378, 0.0377, 0.0376,
        ], fill(0.0375, 76))
    so4_050 = vcat(Float64[
            27.5500, 17.2144, 8.0486, 2.2052, 0.2546, 0.2230, 0.2363, 0.2423, 0.2387, 0.2272,
            0.2101, 0.1899, 0.1678, 0.1457, 0.1240, 0.1011, 0.0708, 0.0435, 0.0424, 0.0473,
            0.0458, 0.0438, 0.0423, 0.0411, 0.0402, 0.0395, 0.0390, 0.0386, 0.0383, 0.0380,
            0.0379, 0.0378, 0.0377, 0.0376, 0.0376, 0.0376,
        ], fill(0.0375, 65))
    so4_075 = vcat(Float64[
            27.5500, 18.0974, 9.3566, 3.1504, 0.3547, 0.1953, 0.2075, 0.2207, 0.2280, 0.2286,
            0.2233, 0.2133, 0.1997, 0.1838, 0.1666, 0.1487, 0.1306, 0.1128, 0.0916, 0.0565,
            0.0405, 0.0443, 0.0484, 0.0465, 0.0443, 0.0430, 0.0419, 0.0409, 0.0402, 0.0396,
            0.0391, 0.0387, 0.0383, 0.0381, 0.0379, 0.0377, 0.0376, 0.0375, 0.0374, 0.0374,
            0.0373, 0.0373, 0.0373,
        ], fill(0.0372, 58))
    so4_100 = vcat(Float64[
            27.5500, 18.0941, 10.6795, 4.1037, 0.5416, 0.1739, 0.1844, 0.1997, 0.2111, 0.2176,
            0.2191, 0.2161, 0.2093, 0.1993, 0.1870, 0.1730, 0.1580, 0.1426, 0.1269, 0.1111,
            0.0923, 0.0548, 0.0408, 0.0449, 0.0482, 0.0476, 0.0449, 0.0432, 0.0422, 0.0414,
            0.0407, 0.0401, 0.0396, 0.0391, 0.0387, 0.0384, 0.0381, 0.0379, 0.0377, 0.0376,
            0.0374, 0.0373, 0.0373, 0.0372, 0.0371, 0.0371, 0.0371, 0.0370, 0.0370, 0.0370,
        ], fill(0.0369, 51))

    # Mg²⁺ — near zero inside (brucite + M-S-H precipitation in TOUGHREACT)
    mg_025 = vcat([52.11], fill(0.00, 100))
    mg_050 = vcat([52.11], fill(0.00, 100))
    mg_075 = vcat([52.11], fill(0.00, 100))
    mg_100 = vcat([52.11], fill(0.00, 100))

    # pH
    pH_025 = vcat(Float64[
            8.0999, 12.6314, 12.7592, 12.8793, 12.9562, 13.0016, 13.0253, 13.0333, 13.0303, 13.0197,
            13.0041, 12.9856, 12.9648, 12.9426, 12.9199, 12.9018, 12.8888, 12.8791, 12.8719, 12.8667,
            12.8630, 12.8606, 12.8588, 12.8576, 12.8569, 12.8563, 12.8560, 12.8557, 12.8556, 12.8554,
            12.8553, 12.8553, 12.8552, 12.8552, 12.8552,
        ], fill(12.8551, 66))
    pH_050 = vcat(Float64[
            8.0999, 12.4573, 12.6980, 12.7978, 12.8800, 12.9385, 12.9825, 13.0132, 13.0328, 13.0435,
            13.0472, 13.0455, 13.0396, 13.0306, 13.0191, 13.0056, 12.9905, 12.9730, 12.9520, 12.9313,
            12.9162, 12.9063, 12.8981, 12.8915, 12.8862, 12.8819, 12.8785, 12.8759, 12.8738, 12.8722,
            12.8710, 12.8700, 12.8692, 12.8687, 12.8683, 12.8679, 12.8677, 12.8675, 12.8674, 12.8672,
            12.8671, 12.8671, 12.8670, 12.8669, 12.8669, 12.8669,
        ], fill(12.8668, 55))
    pH_075 = vcat(Float64[
            8.0999, 12.2355, 12.6715, 12.7615, 12.8415, 12.8989, 12.9461, 12.9839, 13.0127, 13.0333,
            13.0471, 13.0553, 13.0589, 13.0586, 13.0551, 13.0490, 13.0406, 13.0306, 13.0190, 13.0058,
            12.9888, 12.9681, 12.9473, 12.9313, 12.9208, 12.9139, 12.9080, 12.9030, 12.8987, 12.8950,
            12.8920, 12.8894, 12.8873, 12.8855, 12.8841, 12.8829, 12.8819, 12.8812, 12.8805, 12.8800,
            12.8795, 12.8792, 12.8789, 12.8787, 12.8785, 12.8784, 12.8783, 12.8782, 12.8781, 12.8780,
            12.8779, 12.8779, 12.8778, 12.8778, 12.8778, 12.8778, 12.8777, 12.8777, 12.8777, 12.8777,
            12.8777, 12.8777, 12.8777, 12.8777, 12.8777, 12.8777, 12.8777, 12.8777, 12.8777, 12.8777,
            12.8777, 12.8777, 12.8777,
        ], fill(12.8776, 28))
    pH_100 = vcat(Float64[
            8.0999, 12.0557, 12.6583, 12.7386, 12.8157, 12.8711, 12.9184, 12.9589, 12.9919, 13.0178,
            13.0374, 13.0515, 13.0612, 13.0668, 13.0690, 13.0683, 13.0650, 13.0598, 13.0526, 13.0439,
            13.0339, 13.0222, 13.0063, 12.9865, 12.9652, 12.9471, 12.9339, 12.9253, 12.9203, 12.9162,
            12.9125, 12.9091, 12.9062, 12.9036, 12.9013, 12.8993, 12.8975, 12.8960, 12.8947, 12.8936,
            12.8927, 12.8920, 12.8913, 12.8908, 12.8903, 12.8900, 12.8896, 12.8894, 12.8891, 12.8889,
            12.8887, 12.8886, 12.8885, 12.8884, 12.8883, 12.8883, 12.8882, 12.8881, 12.8881, 12.8880,
            12.8880, 12.8880, 12.8880, 12.8879, 12.8879, 12.8879, 12.8879, 12.8879, 12.8879, 12.8879,
            12.8879,
        ], fill(12.8878, 30))

    return (;
        t_yr=[0.25, 0.50, 0.75, 1.00],
        x_dm=x_dm,
        phi=[phi_025, phi_050, phi_075, phi_100],
        n_ch=[n_ch_025, n_ch_050, n_ch_075, n_ch_100],
        n_ms=[n_ms_025, n_ms_050, n_ms_075, n_ms_100],
        n_brc=[n_brc_025, n_brc_050, n_brc_075, n_brc_100],
        cl=[cl_025, cl_050, cl_075, cl_100],
        na=[na_025, na_050, na_075, na_100],
        k=[k_025, k_050, k_075, k_100],
        ca=[ca_025, ca_050, ca_075, ca_100],
        so4=[so4_025, so4_050, so4_075, so4_100],
        mg=[mg_025, mg_050, mg_075, mg_100],
        pH=[pH_025, pH_050, pH_075, pH_100],
    )
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_toughreact(results, grid)

Comparison table against the TOUGHREACT reference profiles (R_38_Marks2015).
Prints free Cl⁻, pH, brucite, Friedel's salt and porosity.
"""
function compare_toughreact(results, grid)
    M_Cl = 35.453
    m_clinker = 359_000.0     # g/m³_concrete
    fac = M_Cl * 100.0 / m_clinker

    t = results[end][1]
    u = results[end][2]
    phi = results[end][3]
    n_fs = results[end][7]
    n_brc = results[end][8]
    c_oh = results[end][9]
    S_Cl_dlm = length(results[end]) >= 18 ? results[end][18] : zeros(size(u, 2))

    x_m = grid[Coordinates][1, :]
    x_dm = x_m .* 10.0

    C_Cl_mol_L = u[ICL_M, :] ./ 1000.0
    C_Mg_mol_L = u[IMG_M, :] ./ 1000.0

    println("\nProfile at t = $(round(t/3.1536e7; digits=2)) year(s) — Marks 2015 (M100 + seawater)")
    println("="^110)
    @printf("%-8s  %-12s  %-12s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
        "x [dm]", "Cl⁻ [mol/L]", "Mg²⁺ [mol/L]", "pH", "φ [-]", "n_FS", "n_Brucite", "S_Cl_DLM")
    println("-"^110)
    for xr in [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075, 0.100]
        idx = argmin(abs.(x_dm .- xr))
        pH_i = 14.0 + log10(max(c_oh[idx], 1e-20) / 1000.0)
        @printf("%-8.4f  %-12.4f  %-12.6f  %-10.2f  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n",
            xr, C_Cl_mol_L[idx], C_Mg_mol_L[idx], pH_i, phi[idx], n_fs[idx], n_brc[idx], S_Cl_dlm[idx])
    end
    println("="^110)

    idx_front = findlast(C_Cl_mol_L .> 1e-3)
    x_front = idx_front !== nothing ? x_dm[idx_front] * 100.0 : NaN
    @printf("\nFront Cl⁻ ≈ %.1f mm\n", x_front)

    @printf("\nBilan Cl en g/100g ciment :\n")
    @printf("  %-8s  %-12s  %-12s  %-12s  %-12s\n", "x [dm]", "Cl libre", "Cl DLM", "Cl Friedel", "Total")
    println("  " * "-"^60)
    for xr in [0.005, 0.010, 0.020, 0.030, 0.050]
        idx = argmin(abs.(x_dm .- xr))
        cl_libre = u[ICL_M, idx] * phi[idx] * fac
        cl_dlm = S_Cl_dlm[idx] * fac
        cl_friedel = 2.0 * n_fs[idx] * fac
        cl_total = cl_libre + cl_dlm + cl_friedel
        @printf("  %-8.3f  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n",
            xr, cl_libre, cl_dlm, cl_friedel, cl_total)
    end
end

"""
    plot_Marks2015(results, grid; n_curves, with_toughreact, save_path)

Returns a tuple of 4 figures (solid lines = PoroMechanics.jl, grey dashes = TOUGHREACT/Thermoddem):

  fig_ions    (2×3) — p1 Cl⁻ | p2 Mg²⁺ | p3 SO₄²⁻ | p4 Na⁺ | p5 Ca²⁺ | p6 pH
  fig_csh     (2×3) — p7 Si | p8 φ | p9 Ca/Si | p10 portlandite | p11 ettringite | p12 CSHQ
  fig_solids  (2×3) — p13 Brucite | p14 Friedel | p15 MonoSulfo | p16 M-S-H | p17 Hydrotalcite
  fig_cl      (1×1) — p18 Formes Cl⁻ (libre / DLM / Friedel / total)

If `save_path` is given (e.g. "results/fig.png"), the 4 figures are saved with the
suffixes `_ions`, `_csh`, `_solids`, `_cl` inserted before the extension.
"""
function plot_Marks2015(results, grid; n_curves=4, with_toughreact=true, save_path=nothing)
    x_dm = grid[Coordinates][1, :] .* 10.0
    t_yr_s = 3.1536e7
    n_t = length(results)
    idxs = unique(clamp.(round.(Int, range(1, n_t; length=n_curves)), 1, n_t))
    pal = [:steelblue, :darkorange, :crimson, :forestgreen, :purple, :teal]

    # ── Formatage commun ──────────────────────────────────────────────────────────
    # left_margin keeps the y label from spilling into the neighbouring panel.
    fmt = (
        titlefontsize=11,
        guidefontsize=9,
        tickfontsize=8,
        legendfontsize=7,
        left_margin=14 * Plots.mm,
        bottom_margin=5 * Plots.mm,
    )

    # ── TOUGHREACT reference ──────────────────────────────────────────────────
    tr = with_toughreact ? toughreact_ref() : nothing
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

    # ── p1 : Cl⁻ ─────────────────────────────────────────────────────────────────
    p1 = plot(; fmt..., xlabel="x [dm]", ylabel="Cl⁻ [mol/m³ sol.]",
        title="Cl⁻ libre", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p1, x_dm, results[ti][2][ICL_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p1, :cl)

    # ── p2 : Mg²⁺ ────────────────────────────────────────────────────────────────
    p2 = plot(; fmt..., xlabel="x [dm]", ylabel="Mg²⁺ [mol/m³ sol.]",
        title="Mg²⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p2, x_dm, results[ti][2][IMG_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p2, :mg)

    # ── p3 : SO₄²⁻ ───────────────────────────────────────────────────────────────
    p3 = plot(; fmt..., xlabel="x [dm]", ylabel="SO₄²⁻ [mol/m³ sol.]",
        title="SO₄²⁻", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p3, x_dm, results[ti][2][ISO4_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p3, :so4)

    # ── p4 : Na⁺ ─────────────────────────────────────────────────────────────────
    p4 = plot(; fmt..., xlabel="x [dm]", ylabel="Na⁺ [mol/m³ sol.]",
        title="Na⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p4, x_dm, results[ti][2][INA_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p4, :na)

    # ── p5 : Ca²⁺ ────────────────────────────────────────────────────────────────
    p5 = plot(; fmt..., xlabel="x [dm]", ylabel="Ca²⁺ [mol/m³ sol.]",
        title="Ca²⁺", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p5, x_dm, results[ti][2][ICA_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p5, :ca)

    # ── p6 : pH ──────────────────────────────────────────────────────────────────
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
        plot!(p7, x_dm, results[ti][2][ISI_M, :]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p8: porosity ──────────────────────────────────────────────────────────
    p8 = plot(; fmt..., xlabel="x [dm]", ylabel="φ [-]",
        title="Porosity", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p8, x_dm, results[ti][3]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    hline!(p8, [0.182]; lw=1, ls=:dot, color=:gray50, label="φ₀")
    add_tr!(p8, :phi)

    # ── p9: CSHQ Ca/Si ratio ──────────────────────────────────────────────────
    # End-members cemdata18 (Lothenbach & Nonat 2015) :
    #   TobH Ca/Si = 5/6 ≈ 0.83,  TobD Ca/Si = 5/6 ≈ 0.83
    #   JenH Ca/Si = 9/6 = 1.50,  JenD Ca/Si = 10/6 ≈ 1.67
    cs_TobH, cs_TobD, cs_JenH, cs_JenD = 5 / 6, 5 / 6, 9 / 6, 10 / 6
    p9 = plot(; fmt..., xlabel="x [dm]", ylabel="Ca/Si [-]",
        title="Ca/Si ratio of the CSHQ", legend=:topright)
    for (k, ti) in enumerate(idxs)
        tobh = results[ti][10]
        tobd = results[ti][11]
        jenh = results[ti][12]
        jend = results[ti][13]
        total = tobh .+ tobd .+ jenh .+ jend
        cs_ratio = @. ifelse(total > 1e-6,
            (cs_TobH * tobh + cs_TobD * tobd + cs_JenH * jenh + cs_JenD * jend) / total, NaN)
        plot!(p9, x_dm, cs_ratio; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    hline!(p9, [cs_TobH]; lw=1, ls=:dot, color=:gray50, label="TobH/D limite")
    hline!(p9, [cs_JenD]; lw=1, ls=:dash, color=:gray30, label="JenD limite")

    # ── p10 : Portlandite ─────────────────────────────────────────────────────────
    p10 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Portlandite Ca(OH)₂", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p10, x_dm, results[ti][4]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end
    add_tr!(p10, :n_ch)

    # ── p11 : Ettringite ──────────────────────────────────────────────────────────
    p11 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="Ettringite", legend=:topright)
    for (k, ti) in enumerate(idxs)
        plot!(p11, x_dm, results[ti][5]; lw=2, color=pal[mod1(k, end)], label=tlbl(ti))
    end

    # ── p12 : CSHQ total + end-members ───────────────────────────────────────────
    p12 = plot(; fmt..., xlabel="x [dm]", ylabel="mol/m³ concrete",
        title="CSHQ (total + end-members)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        col = pal[mod1(k, end)]
        tobh = results[ti][10]
        tobd = results[ti][11]
        jenh = results[ti][12]
        jend = results[ti][13]
        total_csh = tobh .+ tobd .+ jenh .+ jend
        plot!(p12, x_dm, total_csh; lw=2.5, color=col, label="Total $(tlbl(ti))")
        plot!(p12, x_dm, tobh; lw=1, ls=:dash, color=col, label="TobH")
        plot!(p12, x_dm, tobd; lw=1, ls=:dot, color=col, label="TobD")
        plot!(p12, x_dm, jenh; lw=1, ls=:dashdot, color=col, label="JenH")
        plot!(p12, x_dm, jend; lw=1, ls=:dashdotdot, color=col, label="JenD")
    end

    # ── p13 : Brucite ─────────────────────────────────────────────────────────────
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

    # ── p15 : Monosulfoaluminate ──────────────────────────────────────────────────
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

    # ── p16 : M-S-H (total + end-members) ────────────────────────────────────────
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

    # ── p17 : LDH-OH (M4A/M6A/M8A) ──────────────────────────────────────────────
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
        plot!(p17, x_dm, ldh_m6; lw=1, ls=:dot,  color=col, label="M6A-OH-LDH $(tlbl(ti))")
        plot!(p17, x_dm, ldh_m8; lw=1, ls=:dashdot, color=col, label="M8A-OH-LDH $(tlbl(ti))")
    end

    # ── p18: split of the Cl⁻ binding at the final time ───────────────────────
    M_Cl_ = 35.453
    m_cem_ = 359_000.0
    fac_g_ = M_Cl_ * 100.0 / m_cem_
    u_f_ = results[end][2]
    phi_f_ = results[end][3]
    n_fs_f_ = results[end][7]
    S_Cl_f_ = length(results[end]) >= 18 ? results[end][18] : zeros(length(x_dm))
    t_lbl_f_ = tlbl(length(results))
    C_libre_f = u_f_[ICL_M, :] .* phi_f_ .* fac_g_
    C_dlm_f = S_Cl_f_ .* fac_g_
    C_friedel_f = 2.0 .* n_fs_f_ .* fac_g_
    C_total_f = C_libre_f .+ C_dlm_f .+ C_friedel_f
    p18 = plot(; fmt..., xlabel="x [dm]", ylabel="Cl [g/100g ciment]",
        title="Cl forms ($t_lbl_f_)", legend=:topright)
    plot!(p18, x_dm, C_libre_f; lw=2, color=:steelblue, label="Cl libre")
    plot!(p18, x_dm, C_dlm_f; lw=2, color=:darkorange, label="Cl DLM C-S-H")
    plot!(p18, x_dm, C_friedel_f; lw=2, color=:crimson, label="Cl Friedel")
    plot!(p18, x_dm, C_total_f; lw=3, color=:black, ls=:dash, label="Cl total")

    fig_ions = plot(p1, p2, p3, p4, p5, p6; layout=(2, 3), size=(1800, 900))
    fig_csh = plot(p7, p8, p9, p10, p11, p12; layout=(2, 3), size=(1800, 900))
    p_empty = plot(; axis=false, grid=false, legend=false)
    fig_solids = plot(p13, p14, p15, p16, p17, p_empty; layout=(2, 3), size=(1800, 900))
    fig_cl = plot(p18; size=(900, 600))

    if save_path !== nothing
        base, ext = splitext(save_path)
        savefig(fig_ions, base * "_ions" * ext)
        savefig(fig_csh, base * "_csh" * ext)
        savefig(fig_solids, base * "_solids" * ext)
        savefig(fig_cl, base * "_cl" * ext)
    end
    return (fig_ions, fig_csh, fig_solids, fig_cl)
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Marks 2015 — M100 concrete + seawater (SNIA: Fick + Gibbs)"

    # M100 concrete — CEM I 52.5N (Marks 2015, Table 1)
    # Oxydes : CaO=66.3%, SiO₂=20.06%, Al₂O₃=5.3%, Fe₂O₃=2.11%, SO₃=3.3%
    # → Bogue fractions computed with the standard EN 196-2 equations
    results, m_fin = run_Marks2015(;
        N=100,
        t_end=3.1536e7,   # 1 year
        n_save=12,
        mat=CementMaterial(
            clinker=ClinkerComposition(
                na2o_frac=0.0018,
                k2o_frac=0.0028,
                f_C3S=0.6942,
                f_C2S=0.0515,
                f_C3A=0.1047,
                f_C4AF=0.0642,
                f_Gp=0.0709,
            ),
            mix=MixDesign(m_clinker=359.0, wc=0.435),
            transport=TransportModel(phi0=0.182),
        ),
        env=ExposureConditions(T_K=293.15),
    )

    grid_ref = simplexgrid(range(0.0, 0.10; length=101))
    compare_toughreact(results, grid_ref)

    try
        using Plots
        figs = plot_Marks2015(results, grid_ref, save_path="./examples/chloride_ingress/fig_Marks2015_1an.png")
        for f in figs
            display(f)
        end
    catch e
        @warn "Plots.jl not available" exception = e
    end
end
