# Chloricem example — Phase 3: TOUGHREACT / SNIA on 4 primary ions
#
# TOUGHREACT approach (Xu et al., 2004 / Steefel & MacQuarrie, 1996):
#
#   1. Transport: pure Fick's law on the 4 primary cemdata18 species:
#        Cl⁻  (ICL=1), Na⁺ (INA=2), K⁺ (IK=3), Ca²⁺ (ICA=4)
#      No electric potential ψ — no electromigration.
#      OH⁻ is NOT a transported variable.
#
#   2. Chemistry (SNIA): after each transport segment, at every node:
#      a. OH⁻ is recovered from the primary charge balance:
#           n_OH ≈ n_Na + n_K + 2·n_Ca − n_Cl  (primary EN, dilute solution)
#      b. equilibrate() (Gibbs) redistributes freely between aqueous species
#         and solids. No manual correction.
#      c. u[ICL..ICA] is read back from the full speciation.
#      d. c_oh_frozen is updated node by node (for logs/plots).
#
#   3. Friedel's salt: thermodynamic trap for Cl⁻ through C4AClH10.
#
# Advantages over Phase 2b / the previous Phase 3:
#   - No ψ → no artificial electrostatic barrier on Cl⁻
#   - OH⁻ consistent with EN after each chemistry step (no frozen c_oh_frozen)
#   - Faithful implementation of TOUGHREACT: Fick transport + Gibbs chemistry
#
# Usage :
#   julia --project examples/Chloricem/run_3.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver

# ── Indices of the primary ions (TOUGHREACT / cemdata18) ──────────────────────
# Transport: pure Fick's law on the 4 primary ions.
# No ψ in the transport — electroneutrality is restored by equilibrate() after
# each segment (SNIA / TOUGHREACT approach).
# OH⁻ is derived from the charge balance by equilibrate(), not transported.

const ICL = 1   # c_Cl    z = -1
const INA = 2   # c_Na    z = +1
const IK = 3   # c_K     z = +1
const ICA = 4   # c_Ca    z = +2

# ── Representative elementary volume ──────────────────────────────────────────
const V_REV_3 = 1.0e-3   # [m³]  = 1 dm³

# ── Model ─────────────────────────────────────────────────────────────────────

"""
Parameters of the Chloricem model — Phase 3 (TOUGHREACT).

4 transported species: Cl⁻, Na⁺, K⁺, Ca²⁺ (pure Fick's law).
`c_oh_frozen`: c(OH⁻) from the last chemical step — for logs and plots only, it
plays no part in the transport.
"""
mutable struct ChloriceModel3 <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64

    # ── Spatially varying fields (one per node) ───────────────────────────────
    phi::Vector{Float64}          # porosity [-]
    n_ch::Vector{Float64}         # portlandite     [mol/m³_concrete]
    n_ett::Vector{Float64}        # ettringite      [mol/m³_concrete]
    n_ms::Vector{Float64}         # monosulphate    [mol/m³_concrete]
    n_fs::Vector{Float64}         # Friedel's salt  [mol/m³_concrete]
    c_so4_local::Vector{Float64}  # c(SO4²⁻) [mol/m³_water] — not transported
    c_oh_frozen::Vector{Float64}  # c(OH⁻)   [mol/m³_water] — for logs/plots

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

    # ── Boundary conditions (x = 0: external NaCl solution) ───────────────────
    c_cl_BC::Float64
    c_na_BC::Float64
    c_k_BC::Float64
    c_ca_BC::Float64

    # ── Initial conditions (intact concrete at OPC equilibrium) ───────────────
    c_cl_init::Float64
    c_na_init::Float64
    c_k_init::Float64
    c_ca_init::Float64

    # ── Adsorption / ion exchange on C-S-H ────────────────────────────────────
    # Cl⁻  : Langmuir isotherm  S = S_max × K_L × c / (1 + K_L × c)
    #         [S_max] = mol/m³_concrete,  [K_L] = m³/mol
    # Na⁺, K⁺, Ca²⁺ : linear isotherm  S = K_d × c
    #         [K_d] = m³_water / m³_concrete  →  R = 1 + K_d / φ
    # These terms are the main retardation mechanism in TOUGHREACT for concrete
    # (on top of the chemical binding through Friedel's salt).
    S_max_Cl::Float64   # max Cl⁻ adsorption capacity  [mol/m³_concrete]
    K_L_Cl::Float64     # Langmuir affinity constant   [m³/mol]
    K_d_Na::Float64     # linear coefficient Na⁺       [m³/m³]
    K_d_K::Float64      # linear coefficient K⁺        [m³/m³]
    K_d_Ca::Float64     # linear coefficient Ca²⁺      [m³/m³]
end

"""
    compute_opc_ic(cs, has_friedels; phi0, T_K, n_ch0, n_ms0, n_ett0,
                   m_clinker, na2o_frac, k2o_frac)

Computes the initial equilibrium state of an OPC paste/concrete in the chemical
system of run_3.jl (Ca-Cl-Na-K-Al-S-H-O).

Strategy:
- The alkalis Na and K are given as mass fractions of Na₂O and K₂O in the
  clinker (rather than as imposed concentrations). This lets `equilibrate()`
  compute c(OH⁻), c(Na⁺) and c(K⁺) freely.
- Dissolution of the alkali oxides is represented properly:
    Na₂O + H₂O → 2 Na⁺ + 2 OH⁻
    K₂O  + H₂O → 2 K⁺  + 2 OH⁻
  The matching OH⁻ is injected explicitly (1 mol OH⁻ per mol of alkali cation)
  so that the elemental O balance is right. Without that term the solver
  balances the Na⁺ + K⁺ charge with Al(OH)₄⁻ instead of OH⁻.
- The solid phases (n_ch0, n_ms0, n_ett0) are passed directly as parameters;
  Al is implicitly set by the AFm phases.
- `equilibrate()` freely determines c(OH⁻), c(Ca²⁺) and c(SO₄²⁻).

Default parameters: typical OPC, C30/35 concrete (calibration from run_2b.jl).
  na2o_frac = 0.0016, k2o_frac = 0.0049, m_clinker = 350
  → c_na ≈ 149 mol/m³, c_k ≈ 301 mol/m³ (before equilibration)

Returns `NamedTuple` :
  (phi, n_ch, n_ett, n_ms, n_fs, c_ca, c_cl, c_na, c_k, c_oh, c_so4)
"""
function compute_opc_ic(
    cs,
    has_friedels::Bool;
    phi0=0.121,    # initial porosity [-]
    T_K=293.15,   # temperature [K]
    n_ch0=1640.0,   # portlandite    [mol/m³_concrete] — run_2b.jl
    n_ms0=100.0,    # monosulphate12 [mol/m³_concrete] — mature OPC (literature)
    n_ett0=0.0,      # ettringite     [mol/m³_concrete] — mature OPC (→ monosulphate)
    m_clinker=350.0,    # kg clinker / m³_concrete (typical C30/35 concrete)
    na2o_frac=0.0016,   # Na₂O mass fraction in the clinker (→ c_na ≈ 149 mol/m³)
    k2o_frac=0.0049,   # K₂O  mass fraction in the clinker (→ c_k  ≈ 301 mol/m³)
)
    T_q = T_K * us"K"
    V_liq = phi0 * V_REV_3   # m³ of water in the REV pores
    m_ck = m_clinker * V_REV_3   # kg of clinker in the REV

    # ── Alkalis: Na₂O + K₂O → complete dissolution ────────────────────────────
    # Na₂O + H₂O → 2 Na⁺ + 2 OH⁻   K₂O + H₂O → 2 K⁺ + 2 OH⁻
    M_Na2O = 61.98e-3    # kg/mol
    M_K2O = 94.20e-3    # kg/mol
    n_Na2O = na2o_frac * m_ck / M_Na2O   # mol of Na₂O in the REV
    n_K2O = k2o_frac * m_ck / M_K2O   # mol of K₂O  in the REV
    n_na = 2.0 * n_Na2O               # mol Na⁺
    n_k = 2.0 * n_K2O                # mol K⁺
    n_oh_alk = n_na + n_k                  # mol OH⁻ d'accompagnement (bilan O)
    n_h2o_ox = n_Na2O + n_K2O             # mol of H₂O consumed by the oxides

    # ── Molar budget per REV ──────────────────────────────────────────────────
    n_water = V_liq * 55_500.0 - n_h2o_ox  # available H₂O (after the oxide reaction)
    n_ch_mol = n_ch0 * V_REV_3
    n_ms_mol = n_ms0 * V_REV_3   # Al implicite : 2 × n_ms_mol
    n_ett_mol = n_ett0 * V_REV_3   # Al implicite : 2 × n_ett_mol

    state = ChemicalState(cs; T=T_q)
    set_quantity!(state, "H2O@", n_water * us"mol")
    set_quantity!(state, "Na+", n_na * us"mol")
    set_quantity!(state, "K+", n_k * us"mol")
    set_quantity!(state, "OH-", n_oh_alk * us"mol")  # O from Na₂O + K₂O
    set_quantity!(state, "Cl-", 1.0e-16 * us"mol")  # trace (intact concrete)
    set_quantity!(state, "Portlandite", n_ch_mol * us"mol")
    set_quantity!(state, "ettringite", n_ett_mol * us"mol")
    set_quantity!(state, "monosulphate12", n_ms_mol * us"mol")
    set_quantity!(state, "SO4-2", 1.0e-16 * us"mol")  # trace — S is in the solids

    @info "compute_opc_ic: initial OPC equilibrium computation…"
    local state_eq
    try
        state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))
    catch e
        error("OPC IC equilibrate failed: $e")
    end

    V_liq_eq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
    V_liq_eq < 1e-15 && error("Zero liquid volume after the OPC IC equilibrate")

    # ── Aqueous concentrations at equilibrium ─────────────────────────────────
    c_ca_eq = ustrip(moles(state_eq, "Ca+2")) / V_liq_eq
    c_cl_eq = ustrip(moles(state_eq, "Cl-")) / V_liq_eq
    c_na_eq = ustrip(moles(state_eq, "Na+")) / V_liq_eq
    c_k_eq = ustrip(moles(state_eq, "K+")) / V_liq_eq
    c_oh_eq = ustrip(moles(state_eq, "OH-")) / V_liq_eq
    c_so4_eq = ustrip(moles(state_eq, "SO4-2")) / V_liq_eq

    # ── Solid amounts at equilibrium ──────────────────────────────────────────
    n_ch_eq = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_3, 0.0)
    n_ett_eq = max(ustrip(moles(state_eq, "ettringite")) / V_REV_3, 0.0)
    n_ms_eq = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_3, 0.0)
    n_fs_eq = has_friedels ?
              max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_3, 0.0) : 0.0

    @info "Initial OPC equilibrium:" φ = round(phi0; digits=4) c_OH = round(c_oh_eq; sigdigits=4) c_Ca = round(c_ca_eq; sigdigits=4) n_CH = round(n_ch_eq; digits=0) n_ett = round(n_ett_eq; digits=0) n_ms = round(n_ms_eq; digits=0)

    return (
        phi=phi0,
        n_ch=n_ch_eq, n_ett=n_ett_eq, n_ms=n_ms_eq, n_fs=n_fs_eq,
        c_ca=c_ca_eq, c_cl=c_cl_eq, c_na=c_na_eq, c_k=c_k_eq,
        c_oh=c_oh_eq, c_so4=c_so4_eq,
    )
end

"""
    ChloriceModel3(N_nodes, cs, has_friedels; kwargs...) -> ChloriceModel3

Constructor with thermodynamic initialisation.
Calls `compute_opc_ic(cs, has_friedels; kwargs...)` — the `kwargs` are
`phi0, T_K, n_ch0, n_ms0, n_ett0, m_clinker, na2o_frac, k2o_frac`
(see `compute_opc_ic`). The alkalis Na₂O and K₂O are converted into
Na⁺ + OH⁻ and K⁺ + OH⁻ before equilibration.
"""
function ChloriceModel3(N_nodes::Int, cs, has_friedels::Bool; kwargs...)
    ic = compute_opc_ic(cs, has_friedels; kwargs...)
    return ChloriceModel3(
        0.05,                                    # L [m]
        fill(ic.phi, N_nodes),                 # phi
        fill(ic.n_ch, N_nodes),                 # n_ch
        fill(ic.n_ett, N_nodes),                 # n_ett
        fill(ic.n_ms, N_nodes),                 # n_ms
        fill(ic.n_fs, N_nodes),                 # n_fs
        fill(ic.c_so4, N_nodes),                 # c_so4_local
        fill(ic.c_oh, N_nodes),                 # c_oh_frozen (for logs)
        2.032e-9,                                # D_Cl
        1.334e-9,                                # D_Na
        1.957e-9,                                # D_K
        0.792e-9,                                # D_Ca
        0.18, 2.7, 2.0e-4, 0.27,                # Oh-Jang params
        96485.0, 8.314, 293.15,                  # Faraday, R_gas, T_K
        # BCs (x=0 : solution NaCl externe 0.523 M)
        523.0, 523.0, 1.0, 0.0,                 # c_cl_BC, c_na_BC, c_k_BC, c_ca_BC
        # ICs (intact concrete at OPC equilibrium)
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca,
        # C-S-H adsorption (initial values — to be calibrated on reference profiles)
        # Cl⁻ : Langmuir   R(c=100 mol/m³) ≈ 1 + S_max×K_L/(φ×(1+K_L×c)²) ≈ 5
        150.0, 0.008,                            # S_max_Cl [mol/m³], K_L_Cl [m³/mol]
        # Na⁺, K⁺, Ca²⁺ : linear   R = 1 + K_d/φ
        0.1, 0.08, 0.3,                          # K_d_Na, K_d_K, K_d_Ca [m³/m³]
    )
end

PoroMechanics.nspecies(::ChloriceModel3) = 4
PoroMechanics.species_names(::ChloriceModel3) = [:c_Cl, :c_Na, :c_K, :c_Ca]

# ── Fonctions utilitaires ─────────────────────────────────────────────────────

function tortuosity_OhJang_3(phi::T, m::ChloriceModel3) where {T<:Real}
    phi_cap = phi > 0 ? T(0.5) * phi : zero(T)
    phi_c = T(m.phi_c)
    n = T(m.n_OJ)
    ds = T(m.ds_OJ)
    dsn = ds^(1 / n)
    m_p = T(0.5) * ((phi_cap - phi_c) + dsn * (1 - phi_c - phi_cap)) / (1 - phi_c)
    tau_paste = (m_p + sqrt(m_p^2 + dsn * phi_c / (1 - phi_c)))^n
    return tau_paste * T(m.tau_agg)
end

@inline function _node_idx_3(x::Float64, m::ChloriceModel3)
    N = length(m.phi) - 1
    return clamp(round(Int, x / (m.L / N)) + 1, 1, N + 1)
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

function PoroMechanics.storage!(f, u, node, m::ChloriceModel3, ::Any)
    i   = _node_idx_3(node.coord[1], m)
    phi = m.phi[i]

    # ── Cl⁻ : in solution + Langmuir adsorption on C-S-H ──────────────────────
    # S_Cl = S_max × K_L × c / (1 + K_L × c)   [mol/m³_concrete]
    # ∂(φ c + S)/∂t = ∂/∂t[(φ + dS/dc)·c] — handled automatically by ForwardDiff
    c_cl = u[ICL]
    S_Cl = m.S_max_Cl * m.K_L_Cl * c_cl / (1 + m.K_L_Cl * c_cl)
    f[ICL] = phi * c_cl + S_Cl

    # ── Na⁺, K⁺, Ca²⁺ : in solution + linear adsorption (C-S-H exchange) ──────
    # S_i = K_d_i × c_i  →  stockage total = (φ + K_d_i) × c_i
    f[INA] = (phi + m.K_d_Na) * u[INA]
    f[IK]  = (phi + m.K_d_K)  * u[IK]
    f[ICA] = (phi + m.K_d_Ca) * u[ICA]
end

function PoroMechanics.flux!(f, u, edge, m::ChloriceModel3, ::Any)
    # Pure Fick's law — no electromigration (TOUGHREACT approach)
    x_mid = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    i = _node_idx_3(x_mid, m)
    phi_i = (i < length(m.phi)) ? (m.phi[i] + m.phi[i+1]) / 2 : m.phi[i]
    tau = tortuosity_OhJang_3(phi_i, m)

    f[ICL] = m.D_Cl * tau * (u[ICL, 1] - u[ICL, 2])
    f[INA] = m.D_Na * tau * (u[INA, 1] - u[INA, 2])
    f[IK]  = m.D_K  * tau * (u[IK,  1] - u[IK,  2])
    f[ICA] = m.D_Ca * tau * (u[ICA, 1] - u[ICA, 2])
end

function PoroMechanics.bcondition!(f, u, bnode, m::ChloriceModel3, ::Any)
    # x = 0 (region=1) : solution NaCl externe — 4 Dirichlet.
    # x = L (region=2): zero flux by default (VoronoiFVM).
    boundary_dirichlet!(f, u, bnode; species=ICL, region=1, value=m.c_cl_BC)
    boundary_dirichlet!(f, u, bnode; species=INA, region=1, value=m.c_na_BC)
    boundary_dirichlet!(f, u, bnode; species=IK, region=1, value=m.c_k_BC)
    boundary_dirichlet!(f, u, bnode; species=ICA, region=1, value=m.c_ca_BC)
end

# ── Initialisation ChemistryLab ───────────────────────────────────────────────

"""
    init_chemistry3() -> (cs, has_friedels)

Loads cemdata18 and builds the chemical system for the Phase 3 reactive
transport with Friedel's salt.

Elemental space: Ca-Cl-Na-K-Al-S-H-O.
Phases solides explicites : Portlandite, ettringite, monosulphate12, Friedel's salt.

Retourne :
- `cs`           : `ChemicalSystem`
- `dict_sp`      : name→`Species` dictionary (all cemdata18 substances)
- `has_friedels` : `true` if Friedel's salt is present in cemdata18
"""
function init_chemistry3()
    data_path = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    isfile(data_path) || error("cemdata18 introuvable : $data_path")

    @info "Loading cemdata18…"
    substances = build_species(data_path)
    dict_sp = Dict(symbol(s) => s for s in substances)

    # ── Aqueous species — elemental space Ca-Cl-Na-K-Al-S-H-O ────────────────
    # The seeds include Al+3 and SO4-2 to widen the elemental space.
    # Portlandite, ettringite and monosulphate12 also act as seeds to trigger
    # selection of the relevant solutes (Al, S, Ca).
    seeds = split("Portlandite ettringite monosulphate12 H2O@ Ca+2 OH- Cl- Na+ K+ H+ Al+3 SO4-2")
    aq_species = speciation(substances, seeds; aggregate_state=[AS_AQUEOUS])

    # ── Phases solides — liste explicite ──────────────────────────────────────
    # Portlandite    : Ca reservoir (dissolves as the pH drops)
    # ettringite     : Ca₆Al₂(SO₄)₃(OH)₁₂·26H₂O — early AFt phase
    # monosulphate12 : Ca₄Al₂(SO₄)(OH)₁₂·6H₂O — Cl-free AFm phase
    # C4AClH10       : Ca₄Al₂Cl₂(OH)₁₂·4H₂O  — thermodynamic trap for Cl⁻
    solid_names = ["Portlandite", "ettringite", "monosulphate12", "C4AClH10"]
    solid_species = [dict_sp[n] for n in solid_names if haskey(dict_sp, n)]
    missing_s = filter(n -> !haskey(dict_sp, n), solid_names)
    if !isempty(missing_s)
        ms = join(missing_s, ", ")
        @warn "Phases not found in cemdata18 (ignored): $ms"
    end

    has_friedels = haskey(dict_sp, "C4AClH10")
    has_friedels || @warn "C4AClH10 missing from cemdata18 — thermodynamic Cl⁻ binding disabled"

    species = vcat(collect(aq_species), solid_species)
    @info "Selected species" n_aq = length(aq_species) n_solid = length(solid_species)

    cs = ChemicalSystem(collect(species), CEMDATA_PRIMARIES)
    return cs, has_friedels
end

# ── Chemical step ─────────────────────────────────────────────────────────────

"""
    chemistry_step3!(m, u, cs, has_friedels)

SNIA chemical step — Phase 3 (TOUGHREACT).

For every interior node:
  1. The concentrations u[ICL..ICA, i] come from the Fick transport.
  2. OH⁻ is recovered from the primary charge balance (primary EN):
       n_OH = max(n_Na + n_K + 2·n_Ca − n_Cl, ε)
     This is the TOUGHREACT assumption: OH⁻ is a secondary species derived
     from EN over the 4 transported primary ions.
  3. equilibrate() (Gibbs) redistributes freely between aqueous species and
     solides. Aucune correction manuelle.
  4. u[ICL..ICA, i] is read back from the full speciation.
  5. m.c_oh_frozen[i] is updated (for logs/plots).
"""
function chemistry_step3!(m::ChloriceModel3, u::Matrix, cs, has_friedels::Bool)
    N = size(u, 2)
    T_q = m.T_K * us"K"

    # Molar volumes [m³/mol] for the porosity balance
    Vm_CH = 33.06e-6    # portlandite     Ca(OH)₂
    Vm_ett = 707.0e-6    # ettringite      Ca₆Al₂(SO₄)₃(OH)₁₂·26H₂O
    Vm_ms = 309.0e-6    # monosulfate12   Ca₄Al₂(SO₄)(OH)₁₂·6H₂O
    Vm_fs = 271.0e-6    # Friedel's salt  Ca₄Al₂Cl₂(OH)₁₂·4H₂O

    for i in 2:N
        phi_i = m.phi[i]

        # ── Concentrations from the transport (dissolved primary ions) ────────
        C_Cl = max(u[ICL, i], 0.0)
        C_Na = max(u[INA, i], 0.0)
        C_K = max(u[IK, i], 0.0)
        C_Ca = max(u[ICA, i], 0.0)

        # ── Molar budget per REV ──────────────────────────────────────────────
        n_cl_aq = C_Cl * phi_i * V_REV_3
        n_na = C_Na * phi_i * V_REV_3
        n_k = C_K * phi_i * V_REV_3
        n_ca_aq = C_Ca * phi_i * V_REV_3
        n_ch = m.n_ch[i] * V_REV_3
        n_ett = m.n_ett[i] * V_REV_3
        n_ms = m.n_ms[i] * V_REV_3
        n_fs = m.n_fs[i] * V_REV_3
        n_so4_aq = m.c_so4_local[i] * phi_i * V_REV_3

        # ── OH⁻ from the primary charge balance (TOUGHREACT assumption) ───────
        # Primary EN: -c_Cl + c_Na + c_K + 2·c_Ca = c_OH  (dilute solution)
        # Gives a c_OH consistent with the 4 transported primary ions.
        # equilibrate() then refines it freely from that starting state.
        n_oh_en = max(n_na + n_k + 2.0 * n_ca_aq - n_cl_aq, 1.0e-20)
        n_water = phi_i * V_REV_3 * 55_500.0

        # ── Fresh chemical state ──────────────────────────────────────────────
        state = ChemicalState(cs; T=T_q)
        set_quantity!(state, "H2O@", n_water * us"mol")
        set_quantity!(state, "OH-", n_oh_en * us"mol")  # EN primaire
        set_quantity!(state, "Cl-", n_cl_aq * us"mol")
        set_quantity!(state, "Na+", n_na * us"mol")
        set_quantity!(state, "K+", n_k * us"mol")
        set_quantity!(state, "Ca+2", n_ca_aq * us"mol")
        set_quantity!(state, "Portlandite", n_ch * us"mol")
        set_quantity!(state, "ettringite", n_ett * us"mol")
        set_quantity!(state, "monosulphate12", n_ms * us"mol")
        set_quantity!(state, "SO4-2", n_so4_aq * us"mol")
        if has_friedels
            set_quantity!(state, "C4AClH10", n_fs * us"mol")
        end

        # ── Thermodynamic equilibrium ─────────────────────────────────────────
        local state_eq
        try
            state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))
        catch e
            @warn "equilibrate failed at node $i — node left unchanged" exception = e
            continue
        end

        # ── Volume of the liquid phase ────────────────────────────────────────
        V_liq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
        if isnan(V_liq) || V_liq < 1e-15
            @warn "Zero/NaN liquid volume at node $i" V_liq
            continue
        end

        # ── Reading back the aqueous concentrations ───────────────────────────
        c_ca_new = ustrip(moles(state_eq, "Ca+2")) / V_liq
        c_cl_new = ustrip(moles(state_eq, "Cl-")) / V_liq
        c_na_new = ustrip(moles(state_eq, "Na+")) / V_liq
        c_k_new = ustrip(moles(state_eq, "K+")) / V_liq
        c_oh_new = ustrip(moles(state_eq, "OH-")) / V_liq
        c_so4_new = ustrip(moles(state_eq, "SO4-2")) / V_liq

        # ── Reading back the solid phases after equilibrium ───────────────────
        n_ch_new = max(ustrip(moles(state_eq, "Portlandite")) / V_REV_3, 0.0)
        n_ett_new = max(ustrip(moles(state_eq, "ettringite")) / V_REV_3, 0.0)
        n_ms_new = max(ustrip(moles(state_eq, "monosulphate12")) / V_REV_3, 0.0)
        n_fs_new = has_friedels ?
                   max(ustrip(moles(state_eq, "C4AClH10")) / V_REV_3, 0.0) : 0.0

        # ── Porosity update (CH + AFm contributions) ──────────────────────────
        Δn_ch = n_ch_new - m.n_ch[i]
        Δn_ett = n_ett_new - m.n_ett[i]
        Δn_ms = n_ms_new - m.n_ms[i]
        Δn_fs = n_fs_new - m.n_fs[i]
        phi_new = phi_i -
                  Δn_ch * Vm_CH -
                  Δn_ett * Vm_ett -
                  Δn_ms * Vm_ms -
                  Δn_fs * Vm_fs
        m.phi[i] = clamp(phi_new, 1e-4, 0.999)
        m.n_ch[i] = n_ch_new
        m.n_ett[i] = n_ett_new
        m.n_ms[i] = n_ms_new
        m.n_fs[i] = n_fs_new
        m.c_so4_local[i] = max(c_so4_new, 0.0)

        # ── Updating the transport concentrations ─────────────────────────────
        u[ICL, i] = max(c_cl_new, 0.0)
        u[INA, i] = max(c_na_new, 0.0)
        u[IK, i] = max(c_k_new, 0.0)
        u[ICA, i] = max(c_ca_new, 0.0)

        # ── c_oh_frozen: tracked for logs/plots (unused by the transport) ──────
        m.c_oh_frozen[i] = max(c_oh_new, 1e-12)
    end
end

# ── Solve (operator splitting) ────────────────────────────────────────────────

"""
    run_Chloricem3(; N, t_end, n_save, verbose) -> (results, model)

SNIA Chloricem — Phase 3 (TOUGHREACT): Fick transport on 4 primary ions,
chimie Gibbs via equilibrate().

# Returns
- `results` : `[(t, u_matrix, phi_vec, n_ch_vec, n_ett_vec, n_ms_vec, n_fs_vec, c_oh_vec), …]`
  where `c_oh_vec` = `m.c_oh_frozen` after the chemical step.
- `model`   : the final `ChloriceModel3` state
"""
function run_Chloricem3(;
    N=100,
    t_end=3.1536e7,   # 1 year [s]
    n_save=12,
    verbose=false,
)
    # ── ChemistryLab (once) — before the model, for the ICs ───────────────────
    cs, has_friedels = init_chemistry3()

    # ── Diagnostic transport ──────────────────────────────────────────────────
    # Computes τ and D_eff for the initial porosity, BEFORE any chemistry.
    let phi0 = 0.121
        phi_cap = 0.5 * phi0
        phi_c_ = 0.18
        n_ = 2.7
        ds_ = 2.0e-4
        tau_agg_ = 0.27
        dsn = ds_^(1 / n_)
        mp = 0.5 * ((phi_cap - phi_c_) + dsn * (1 - phi_c_ - phi_cap)) / (1 - phi_c_)
        tau_p = (mp + sqrt(mp^2 + dsn * phi_c_ / (1 - phi_c_)))^n_
        tau = tau_p * tau_agg_
        D_eff_Cl = 2.032e-9 * tau                        # [m²/s]
        D_app_Cl = D_eff_Cl / phi0                       # D_eff/φ (VoronoiFVM: storage=φ·c)
        t_1yr = 3.1536e7                                  # [s] — always 1 year for the comparison
        front_1yr_mm = 2 * sqrt(D_app_Cl * t_1yr) * 1e3 # mm, pure diffusion at 1 year
        R_needed = (front_1yr_mm / 7.5)^2                # retardation needed for a 7.5 mm/yr front
        @info "Transport diagnostic (phi=0.121, no chemical retardation, at 1 year)" tau = round(tau; sigdigits=3) D_eff_Cl_m2s = round(D_eff_Cl; sigdigits=3) D_app_Cl_m2s = round(D_app_Cl; sigdigits=3) front_pure_1yr_mm = round(front_1yr_mm; sigdigits=3)
        @info "  → For a front ≈ 7.5 mm/yr, the chemical retardation needed is R ≈ $(round(R_needed; sigdigits=2)) (Langmuir alone: R ≈ $(round(1 + 150*0.008/phi0; sigdigits=2)))"
    end

    # ── Model with thermodynamic ICs ──────────────────────────────────────────
    m = ChloriceModel3(N + 1, cs, has_friedels)
    grid = simplexgrid(range(0.0, m.L; length=N + 1))

    # ── VoronoiFVM adapters (4 species, no reaction) ──────────────────────────
    _storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data) = PoroMechanics.flux!(f, u, edge, m, data)
    _bcond!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage=_storage!,
        flux=_flux!,
        bcondition=_bcond!,
        species=[ICL, INA, IK, ICA],
    )

    # ── Condition initiale ────────────────────────────────────────────────────
    inival = unknowns(sys)
    inival[ICL, :] .= m.c_cl_init
    inival[INA, :] .= m.c_na_init
    inival[IK, :] .= m.c_k_init
    inival[ICA, :] .= m.c_ca_init
    # Nœud x = 0 : solution NaCl externe
    inival[ICL, 1] = m.c_cl_BC
    inival[INA, 1] = m.c_na_BC
    inival[IK, 1] = m.c_k_BC
    inival[ICA, 1] = m.c_ca_BC

    # ── Time step controller ──────────────────────────────────────────────────
    ctrl = VoronoiFVM.SolverControl(;
        Δt=1.0,
        Δt_max=t_end / (4 * n_save),
        Δu_opt=0.5 * m.c_cl_BC,
        handle_exceptions=true,
        verbose=verbose,
    )

    # ── SNIA loop ─────────────────────────────────────────────────────────────
    tsave = range(0.0, t_end; length=n_save + 1)
    # Tuple : (t, u, phi, n_ch, n_ett, n_ms, n_fs, c_oh_frozen)
    results = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    u_cur = copy(inival)

    for k in 2:lastindex(tsave)
        t0 = tsave[k-1]
        t1 = tsave[k]

        @info "Transport seg $k" t_yr = round(t1 / 3.1536e7; digits=2)

        # Step 1: Fick transport (VoronoiFVM)
        seg = solve(sys; inival=u_cur, times=[t0, t1], control=ctrl)
        u_cur = Matrix(seg[:, :, end])

        # Step 2: chemistry (ChemistryLab) — nodes 2..N+1
        chemistry_step3!(m, u_cur, cs, has_friedels)

        # Restore the BCs (x = 0: external solution)
        u_cur[ICL, 1] = m.c_cl_BC
        u_cur[INA, 1] = m.c_na_BC
        u_cur[IK, 1] = m.c_k_BC
        u_cur[ICA, 1] = m.c_ca_BC
        m.c_oh_frozen[1] = 1.0   # EN externe : c_Na + c_K − c_Cl ≈ 1 mol/m³

        if any(isnan, u_cur)
            idx = findall(isnan, u_cur)
            @warn "NaN after chemistry, seg $k" first = idx[1:min(5, end)]
        end

        push!(results, (
            t1,
            copy(u_cur),
            copy(m.phi),
            copy(m.n_ch),
            copy(m.n_ett),
            copy(m.n_ms),
            copy(m.n_fs),
            copy(m.c_oh_frozen),   # OH⁻ from the last chemical step
        ))

        @info "SNIA" seg = k - 1 φ_mean = round(mean(m.phi); sigdigits=4) n_CH0 = round(m.n_ch[2]; sigdigits=4) n_FS0 = round(m.n_fs[2]; sigdigits=4)
    end

    return results, m
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_reference_3(results, grid)

Prints the C_Cl and c_OH_frozen profiles at t_final.
Compares C_Cl with the Phase 2b C++ reference.
"""
function compare_reference_3(results, grid)
    t, u, phi, n_ch, n_ett, n_ms, n_fs, c_oh = results[end]
    x_m = grid[Coordinates][1, :]
    x_dm = x_m .* 10.0

    C_Cl = u[ICL, :] ./ 1000.0   # mol/dm³
    C_Ca = u[ICA, :] ./ 1000.0
    c_oh_dm3 = c_oh ./ 1000.0    # mol/dm³ — from m.c_oh_frozen

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    println("\nProfile at t = $(round(t/3.1536e7; digits=2)) year(s) — Phase 3 TOUGHREACT (Fick + Gibbs)")
    println("="^95)
    @printf("%-8s  %-14s  %-14s  %-12s  %-8s  %-8s  %-10s\n",
        "x [dm]", "C_Cl [mol/dm³]", "C++ ref", "c_OH [mol/dm³]", "C_Ca", "φ [-]", "n_FS [mol/m³]")
    println("-"^95)
    for (xr, cr) in zip(x_ref, c_ref)
        idx = argmin(abs.(x_dm .- xr))
        @printf("%-8.4f  %-14.4f  %-14.4f  %-12.4f  %-8.5f  %-8.4f  %-10.2f\n",
            xr, C_Cl[idx], cr, c_oh_dm3[idx], C_Ca[idx], phi[idx], n_fs[idx])
    end
    println("="^95)

    idx_front = findlast(C_Cl .> 1e-3)
    x_front = idx_front !== nothing ? x_dm[idx_front] : NaN

    # Chemical retardation: compare the n_fs formed with the C_Cl in solution
    @printf("\nChemical retardation (Friedel's salt):\n")
    @printf("  x [dm]   C_Cl_libre   n_FS    ratio n_FS/C_Cl_pore\n")
    for xr in [0.000, 0.005, 0.010, 0.020]
        idx_ = argmin(abs.(x_dm .- xr))
        c_cl_pore = C_Cl[idx_] * 1000.0 * phi[idx_]   # mol/m³_concrete — free Cl in solution
        @printf("  %-8.3f  %-12.2f %-8.2f  %-6.2f\n",
            xr, c_cl_pore, n_fs[idx_],
            c_cl_pore > 0.01 ? n_fs[idx_] / c_cl_pore : 0.0)
    end

    @printf("\nDiagnostics Phase 3 :\n")
    @printf("  C_Cl(x=0)  = %.4f mol/dm³  (ref: 0.523)\n", C_Cl[1])
    @printf("  c_OH(x=0)  = %.4f mol/dm³  (solution externe NaCl ≈ 0.001)\n", c_oh_dm3[1])
    @printf("  φ(x=0)     = %.4f  (initial = 0.121)\n", phi[1])
    @printf("  n_CH(x=0)  = %.1f mol/m³  (initial = 1640)\n", n_ch[1])
    @printf("  n_ETT(x=0) = %.1f mol/m³\n", n_ett[1])
    @printf("  n_MS(x=0)  = %.1f mol/m³\n", n_ms[1])
    @printf("  n_FS(x=0)  = %.2f mol/m³  (initially 0 — Cl⁻ trap)\n", n_fs[1])
    @printf("  Cl⁻ front  ≈ %.1f mm  (C++ ref ≈ 7.5 mm)\n", x_front * 100.0)
end

"""
    plot_Chloricem3(results, grid; n_curves=4, save_path=nothing)

Quatre panneaux :
  - Left     : C_Cl profiles (+ C++ reference)
  - Centre-L : porosity φ(x)
  - Centre-R : Friedel's salt n_fs(x) and monosulphate n_ms(x)
  - Right    : c_OH_frozen at the final time
"""
function plot_Chloricem3(results, grid;
    n_curves=4,
    save_path=nothing,
)
    x_dm = grid[Coordinates][1, :] .* 10.0
    t_yr = 3.1536e7

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    n_t = length(results)
    idxs = unique(clamp.(round.(Int, range(1, n_t; length=n_curves)), 1, n_t))
    pal = [:steelblue, :darkorange, :crimson, :forestgreen,
        :purple, :teal, :goldenrod, :indianred]

    p1 = plot(; xlabel="x [dm]", ylabel="C_Cl [mol/dm³]",
        title="Phase 3 — Cl⁻", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k, u_k = results[ti][1], results[ti][2]
        plot!(p1, x_dm, u_k[ICL, :] ./ 1000;
            lw=2, color=pal[mod1(k, end)],
            label="t = $(round(t_k/t_yr; digits=2)) yr")
    end
    scatter!(p1, x_ref, c_ref; ms=5, color=:black, label="C++ ref (1 yr)")

    p2 = plot(; xlabel="x [dm]", ylabel="φ [-]",
        title="Phase 3 — porosity", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k, phi_k = results[ti][1], results[ti][3]
        plot!(p2, x_dm, phi_k;
            lw=2, color=pal[mod1(k, end)],
            label="t = $(round(t_k/t_yr; digits=2)) yr")
    end
    hline!(p2, [0.121]; lw=1, ls=:dot, color=:gray50, label="φ₀")

    p3 = plot(; xlabel="x [dm]", ylabel="n [mol/m³_concrete]",
        title="Phase 3 — AFm (Friedel + MS)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k = results[ti][1]
        n_fs_k = results[ti][7]
        n_ms_k = results[ti][6]
        lbl = "t = $(round(t_k/t_yr; digits=2)) yr"
        plot!(p3, x_dm, n_fs_k; lw=2, color=pal[mod1(k, end)], label="FS $lbl")
        plot!(p3, x_dm, n_ms_k; lw=1, ls=:dash, color=pal[mod1(k, end)], label="MS $lbl")
    end

    t_fin = results[end][1]
    c_oh_f = results[end][8]   # c_oh_frozen from the last step
    p4 = plot(x_dm, c_oh_f ./ 1000;
        xlabel="x [dm]", ylabel="c_OH [mol/dm³]",
        title="Phase 3 — OH⁻  (t = $(round(t_fin/t_yr; digits=1)) yr)",
        lw=2, color=:darkorange, label="c_OH")

    fig = plot(p1, p2, p3, p4; layout=(1, 4), size=(1700, 430))
    save_path !== nothing && savefig(fig, save_path)
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Chloricem — Phase 3 TOUGHREACT: pure Fick on 4 primary ions + Gibbs chemistry"
    results, m_fin = run_Chloricem3(; verbose=false)

    grid_ref = simplexgrid(range(0.0, 0.05; length=101))
    compare_reference_3(results, grid_ref)

    try
        using Plots
        p = plot_Chloricem3(results, grid_ref)
        display(p)
    catch e
        @warn "Plots.jl non disponible" exception = e
    end
end
