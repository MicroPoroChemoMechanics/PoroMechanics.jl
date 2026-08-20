# Exemple Chloricem — Phase 2b : Nernst-Planck + dissolution CH (operator splitting)
#
# Extension of Phase 2a with portlandite (Ca(OH)₂) dissolution through ChemistryLab.jl.
#
# Strategy: operator splitting (SNIA — Sequential Non-Iterative Approach)
#   Step 1 — Transport (VoronoiFVM):
#     Solve the NP PDEs for Cl⁻, Na⁺, K⁺, OH⁻, Ca²⁺, ψ over [tₙ, tₙ₊₁].
#   Step 2 — Chemistry (ChemistryLab):
#     At every node, update the chemical state with the transported
#     concentrations, call equilibrate(), and read back φ_new, n_CH_new and the
#     new concentrations (after dissolution or precipitation of Ca(OH)₂).
#
# New physics compared with Phase 2a:
#   - Ca²⁺ is a 5th transported species (z = +2).
#   - φ(x, t) follows the dissolution of Ca(OH)₂ → D_eff(x, t) increases.
#   - n_CH(x, t) decreases near x = 0.
#
# Prerequisite:
#   ChemistryLab.jl must be available.
#   From the Julia REPL of the PoroMechanics.jl project:
#     import Pkg
#     Pkg.develop(path = raw"C:\Users\anthony.soive\Box\...\ChemistryLab.jl")
#     Pkg.add("DynamicQuantities")
#
# Usage :
#   julia --project examples/Chloricem/run_2b.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver

# ── Species indices ───────────────────────────────────────────────────────────

const ICL = 1   # Cl⁻    z = -1
const INA = 2   # Na⁺    z = +1
const IK  = 3   # K⁺     z = +1
const IOH = 4   # OH⁻    z = -1
const ICA = 5   # Ca²⁺   z = +2
const IPS = 6   # ψ      algebraic equation

const Z_IONS_2B = (-1.0, +1.0, +1.0, -1.0, +2.0)

# ── Representative elementary volume (REV) ────────────────────────────────────
# 1 dm³ = 0.001 m³ — working scale for ChemistryLab (sensible mole numbers)
const V_REV = 1.0e-3   # [m³]

# ── Model (mutable: φ and n_CH vary in space and time) ───────────────────────

"""
Parameters of the Chloricem model — Phase 2b.

The vector fields (`phi`, `n_ch`, `n_csh`) hold one entry per grid node and are
updated after every chemical step (operator splitting).
"""
mutable struct ChloriceModel2b <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64

    # ── Spatially varying parameters (one per node) ───────────────────────────
    phi::Vector{Float64}      # porosity  [-]
    n_ch::Vector{Float64}     # portlandite  [mol/m³_concrete]
    n_csh::Vector{Float64}    # C-S-H (constant in Phase 2b)  [mol/m³_concrete]

    # ── Diffusion ionique en eau libre [m²/s] ─────────────────────────────────
    D_Cl::Float64
    D_Na::Float64
    D_K::Float64
    D_OH::Float64
    D_Ca::Float64

    # ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────
    phi_c::Float64
    n_OJ::Float64
    ds_OJ::Float64
    tau_agg::Float64

    # ── Langmuir adsorption (Cl⁻ / C-S-H) ─────────────────────────────────────
    alpha::Float64
    beta::Float64

    # ── Physical constants ────────────────────────────────────────────────────
    Faraday::Float64
    R_gas::Float64
    T_K::Float64

    # ── Boundary conditions ───────────────────────────────────────────────────
    c_cl_BC::Float64   # [mol/m³]  Cl⁻ imposed at x = 0  (external NaCl solution)
    c_na_BC::Float64   # [mol/m³]  Na⁺ imposed at x = 0  (NaCl → c_Na = c_Cl)
    c_k_BC::Float64    # [mol/m³]  K⁺  imposed at x = 0  (very low in NaCl)
    c_oh_BC::Float64   # [mol/m³]  OH⁻ imposed at x = 0  (EN: = c_Na - c_Cl + c_K = 1)
    c_ca_BC::Float64   # [mol/m³]  Ca²⁺ imposed at x = 0 (no portlandite in solution)
    psi_BC::Float64    # [V]       reference potential

    # ── Initial conditions (concrete pore solution) ───────────────────────────
    c_cl_init::Float64
    c_na_init::Float64
    c_k_init::Float64
    c_oh_init::Float64
    c_ca_init::Float64
    psi_init::Float64
end

"""
Default constructor of `ChloriceModel2b` for N nodes.

Parameters are tuned on the reference case.
"""
function ChloriceModel2b(N_nodes::Int)
    phi0    = 0.121
    n_ch0   = 1640.0    # mol/m³  (= 1.64 mol/dm³ in the reference case)
    n_csh0  =  635.0    # mol/m³

    # Portlandite equilibrium at pH 12.65 (c_OH = 0.45 mol/dm³ = 450 mol/m³):
    # Ksp(Ca(OH)₂) = 6.46e-6 mol³/dm⁶ → c_Ca = Ksp / c_OH² = 0.032 mol/m³
    c_ca0   = 0.032
    c_cl0   = 0.01
    c_na0   = 150.0
    c_k0    = 300.0
    # Electroneutrality: -c_Cl + c_Na + c_K - c_OH + 2*c_Ca = 0
    c_oh0   = c_na0 + c_k0 - c_cl0 + 2 * c_ca0   # = 450.054 mol/m³

    return ChloriceModel2b(
        0.05,                          # L
        fill(phi0, N_nodes),           # phi
        fill(n_ch0, N_nodes),          # n_ch
        fill(n_csh0, N_nodes),         # n_csh
        2.032e-9,                      # D_Cl
        1.334e-9,                      # D_Na
        1.957e-9,                      # D_K
        5.273e-9,                      # D_OH
        0.792e-9,                      # D_Ca  (Robinson & Stokes 1959)
        0.18,                          # phi_c
        2.7,                           # n_OJ
        2.0e-4,                        # ds_OJ
        0.27,                          # tau_agg
        3.192,                         # alpha
        0.0266,                        # beta  [m³/mol]
        96485.0,                       # Faraday
        8.314,                         # R_gas
        293.15,                        # T_K   (20°C)
        523.0,                         # c_cl_BC  [mol/m³] solution NaCl ≈ 0.523 mol/dm³
        523.0,                         # c_na_BC  [mol/m³] Na⁺ en solution NaCl = Cl⁻
        1.0,                           # c_k_BC   [mol/m³] K⁺ quasiment absent (solution NaCl)
        1.0,                           # c_oh_BC  [mol/m³] EN : c_Na + c_K - c_Cl = 523+1-523 = 1
        0.0,                           # c_ca_BC  [mol/m³] no portlandite in the external solution
        0.0,                           # psi_BC
        c_cl0, c_na0, c_k0, c_oh0, c_ca0, 0.0,
    )
end

PoroMechanics.nspecies(::ChloriceModel2b) = 6
PoroMechanics.species_names(::ChloriceModel2b) = [:c_cl, :c_na, :c_k, :c_oh, :c_ca, :psi]

# ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────────

function tortuosity_OhJang_2b(phi::T, m::ChloriceModel2b) where {T<:Real}
    phi_cap = phi > 0 ? T(0.5) * phi : zero(T)
    phi_c   = T(m.phi_c);  n = T(m.n_OJ);  ds = T(m.ds_OJ)
    dsn     = ds^(1 / n)
    m_p     = T(0.5) * ((phi_cap - phi_c) + dsn * (1 - phi_c - phi_cap)) / (1 - phi_c)
    tau_paste = (m_p + sqrt(m_p^2 + dsn * phi_c / (1 - phi_c)))^n
    return tau_paste * T(m.tau_agg)    # saturated: s_l = 1
end

# ── Langmuir adsorption ───────────────────────────────────────────────────────

n_ads_2b(c::T, n_csh::Float64, m::ChloriceModel2b) where {T<:Real} =
    T(n_csh) * T(m.alpha) * c / (T(1000.0) * (1 + T(m.beta) * c))

# ── Bernoulli (Scharfetter-Gummel, ForwardDiff-compatible) ───────────────────

function bernoulli_2b(x::T) where {T<:Real}
    abs(x) < 1.0e-7 && return one(T) - x / 2
    return x / (exp(x) - one(T))
end

# ── Node index lookup from a coordinate ───────────────────────────────────────

@inline function _node_index(x::Float64, m::ChloriceModel2b)
    N = length(m.phi) - 1
    i = round(Int, x / (m.L / N)) + 1
    return clamp(i, 1, N + 1)
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

function PoroMechanics.storage!(f, u, node, m::ChloriceModel2b, ::Any)
    i     = _node_index(node.coord[1], m)
    phi_i = m.phi[i]
    ncsh_i = m.n_csh[i]

    f[ICL] = phi_i * u[ICL] + n_ads_2b(u[ICL], ncsh_i, m)
    f[INA] = phi_i * u[INA]
    f[IK]  = phi_i * u[IK]
    f[IOH] = phi_i * u[IOH]
    f[ICA] = phi_i * u[ICA]
    f[IPS] = zero(eltype(u))
end

function PoroMechanics.reaction!(f, u, ::Any, ::ChloriceModel2b, ::Any)
    # Algebraic EN constraint: ψ is the potential that maintains electroneutrality.
    # After the partial chemical step (only Ca²⁺ and OH⁻ updated), dissolution of
    # Ca(OH)₂ is EN-neutral (Δcharge = +2 - 2×1 = 0), so the residual here stays
    # ≈ the Newton tolerance — non-zero → non-singular Jacobian, no NaN.
    f[IPS] = -u[ICL] + u[INA] + u[IK] - u[IOH] + 2 * u[ICA]
end

function PoroMechanics.flux!(f, u, edge, m::ChloriceModel2b, ::Any)
    x_mid = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    i     = _node_index(x_mid, m)
    # Uses the local porosity for the tortuosity
    phi_i = (i <= length(m.phi) - 1) ? (m.phi[i] + m.phi[i+1]) / 2 : m.phi[i]
    tau   = tortuosity_OhJang_2b(phi_i, m)

    FoRT  = m.Faraday / (m.R_gas * m.T_K)
    Δψ    = u[IPS, 1] - u[IPS, 2]
    D_free = (m.D_Cl, m.D_Na, m.D_K, m.D_OH, m.D_Ca)

    for (idx, (zi, Df)) in enumerate(zip(Z_IONS_2B, D_free))
        Di   = Df * tau
        dV   = zi * FoRT * Δψ
        f[idx] = Di * (bernoulli_2b(dV) * u[idx, 1] - bernoulli_2b(-dV) * u[idx, 2])
    end
    f[IPS] = zero(eltype(u))
end

function PoroMechanics.bcondition!(f, u, bnode, m::ChloriceModel2b, ::Any)
    # At x = 0: external NaCl solution — 6 Dirichlet values (Cl, Na, K, OH, Ca, ψ).
    # With ψ imposed (Dirichlet), VoronoiFVM disables the algebraic EN constraint at
    # the boundary node: without Dirichlet on OH and Ca, those species drift toward
    # the interior concrete values through the fast D_OH diffusion.
    # We impose OH = c_na + c_k - c_cl = 1 mol/m³ (electroneutral NaCl solution)
    # and Ca = 0 (no portlandite in the external solution), as in the C++ version.
    boundary_dirichlet!(f, u, bnode; species=ICL, region=1, value=m.c_cl_BC)
    boundary_dirichlet!(f, u, bnode; species=INA, region=1, value=m.c_na_BC)
    boundary_dirichlet!(f, u, bnode; species=IK,  region=1, value=m.c_k_BC)
    boundary_dirichlet!(f, u, bnode; species=IOH, region=1, value=m.c_oh_BC)
    boundary_dirichlet!(f, u, bnode; species=ICA, region=1, value=m.c_ca_BC)
    boundary_dirichlet!(f, u, bnode; species=IPS, region=1, value=m.psi_BC)
end

# ── Initialisation ChemistryLab ───────────────────────────────────────────────

"""
    init_chemistry(m, N_nodes) -> (cs, chem_states)

Loads the cemdata18 thermodynamic database, selects the species relevant to
portlandite dissolution, and creates one initial `ChemicalState` per node
(concrete pore solution + portlandite at equilibrium).

The `chem_states` are `ChemicalState`s in moles per REV (V_REV = 1 dm³).
"""
function init_chemistry(m::ChloriceModel2b, N_nodes::Int)
    # Path to the thermodynamic database (cemdata18)
    data_path = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    if !isfile(data_path)
        error("Base cemdata18 introuvable : $data_path\n" *
              "Check the ChemistryLab.jl path.")
    end

    @info "Loading cemdata18…" path=data_path
    substances = build_species(data_path)

    # Species relevant to Ca(OH)₂ dissolution
    # "Portlandite" = crystalline Ca(OH)₂, the others are aqueous
    seeds = split("Portlandite H2O@ Ca+2 OH- Cl- Na+ K+ H+")
    species = speciation(substances, seeds; aggregate_state = [AS_AQUEOUS])
    @info "Selected species" n_species=length(species)

    cs = ChemicalSystem(species, CEMDATA_PRIMARIES)

    # Create one initial state per node
    @info "Initialising the per-node chemical states…" N=N_nodes
    chem_states = Vector{Any}(undef, N_nodes)

    for i in 1:N_nodes
        phi_i = m.phi[i]
        state = ChemicalState(cs; T = m.T_K * u"K")

        # Amounts in moles per REV (V_REV = 1 dm³ = 0.001 m³)
        set_quantity!(state, "H2O@",        phi_i * V_REV * 55500.0 * u"mol")
        set_quantity!(state, "Cl-",         m.c_cl_init * phi_i * V_REV * u"mol")
        set_quantity!(state, "Na+",         m.c_na_init * phi_i * V_REV * u"mol")
        set_quantity!(state, "K+",          m.c_k_init  * phi_i * V_REV * u"mol")
        set_quantity!(state, "OH-",         m.c_oh_init * phi_i * V_REV * u"mol")
        set_quantity!(state, "Ca+2",        m.c_ca_init * phi_i * V_REV * u"mol")
        set_quantity!(state, "Portlandite", m.n_ch[i]   * V_REV         * u"mol")

        # Initial equilibrium (gives the exact equilibrium Ca²⁺ and OH⁻)
        chem_states[i] = equilibrate(state)
    end
    @info "Initial chemical states created"
    return cs, chem_states
end

# ── Chemical step (operator splitting) ────────────────────────────────────────

"""
    chemistry_step!(m, u, chem_states)

Applies the chemical step of the operator splitting at time tₙ₊₁: at every node
we start from the transported concentrations, equilibrate portlandite (Ca(OH)₂),
and update φ, n_CH and the concentrations.

Units:
  - `u[species, node]` — concentration [mol/m³_pore]
  - `m.phi[node]`       — porosity [-]
  - `m.n_ch[node]`      — portlandite [mol/m³_concrete]
"""
function chemistry_step!(m::ChloriceModel2b, u::Matrix, cs)
    N = size(u, 2)   # number of nodes

    # Node 1 (x = 0) is the BC node: it is the external NaCl solution, not the
    # concrete paste. Portlandite chemistry does not apply there.
    for i in 2:N
        phi_i = m.phi[i]

        # Current concentrations [mol/m³_pore], clamped ≥ 0
        c_cl = max(u[ICL, i], 0.0)
        c_na = max(u[INA, i], 0.0)
        c_k  = max(u[IK,  i], 0.0)
        c_oh = max(u[IOH, i], 0.0)
        c_ca = max(u[ICA, i], 0.0)

        # Conversion to moles per REV: n_i = c_i [mol/m³] × φ × V_REV [m³]
        n_water = phi_i * V_REV * 55500.0
        n_cl    = c_cl * phi_i * V_REV
        n_na    = c_na * phi_i * V_REV
        n_k     = c_k  * phi_i * V_REV
        n_oh    = c_oh * phi_i * V_REV
        n_ca    = c_ca * phi_i * V_REV
        n_ch    = m.n_ch[i] * V_REV           # moles of portlandite per REV

        # Fresh state at every node: avoids carrying over residual state from the
        # previous equilibration (the cause of the observed doubling of n_CH).
        state = ChemicalState(cs; T = m.T_K * u"K")
        set_quantity!(state, "H2O@",        n_water * u"mol")
        set_quantity!(state, "Cl-",         n_cl    * u"mol")
        set_quantity!(state, "Na+",         n_na    * u"mol")
        set_quantity!(state, "K+",          n_k     * u"mol")
        set_quantity!(state, "OH-",         n_oh    * u"mol")
        set_quantity!(state, "Ca+2",        n_ca    * u"mol")
        set_quantity!(state, "Portlandite", n_ch    * u"mol")

        local state_eq
        try
            state_eq = equilibrate(state)
        catch e
            @warn "equilibrate failed at node $i — concentrations left unchanged" exception=e c_cl c_na c_k c_oh c_ca n_ch=m.n_ch[i]
            continue   # keep u[:,i] and m.n_ch[i] unchanged
        end

        # Portlandite after equilibrium [mol/m³_concrete]
        n_ch_new_raw = ustrip(moles(state_eq, "Portlandite")) / V_REV

        # Physical guard: n_ch_new cannot exceed the total available Ca.
        # Ca total = Ca_aq + Ca_portlandite = n_ca/V_REV + m.n_ch[i]
        n_ca_total = n_ca / V_REV + m.n_ch[i]   # mol/m³_concrete
        n_ch_new = clamp(n_ch_new_raw, 0.0, n_ca_total)
        if n_ch_new != n_ch_new_raw
            @warn "unphysical n_ch_new at node $i — clamped to the bound" n_ch_new_raw n_ch_new c_cl c_oh c_ca n_ch_in=m.n_ch[i]
        end

        # Porosity update from the dissolution of Ca(OH)₂:
        #   Δφ = −ΔnCH × Vm,CH   (Vm,CH ≈ 33.06e-6 m³/mol)
        # NOTE: state_eq.porosity[] is NOT used, because ChemistryLab only knows
        # about Portlandite + aqueous species — V_total would be underestimated.
        Vm_CH = 33.06e-6   # m³/mol
        Δn_ch = n_ch_new - m.n_ch[i]   # < 0 when dissolving
        phi_new = phi_i - Δn_ch * Vm_CH
        m.phi[i]  = clamp(phi_new, 1e-4, 0.999)
        m.n_ch[i] = n_ch_new

        # Selective update from ChemistryLab:
        # - Ca²⁺ : taken directly from the equilibrium (Ca(OH)₂ ⇌ Ca²⁺ + 2OH⁻).
        # - OH⁻  : EN-NEUTRAL (stoichiometric) update, not ChemistryLab's raw value.
        #          Dissolution is EN-neutral: Δcharge = +2 - 2×1 = 0, so
        #          Δc_OH = 2 × Δc_Ca (stoichiometry) preserves EN exactly.
        #          Using moles(state_eq,"OH-")/V_liq brings in thermodynamic
        #          corrections (CaOH⁺ complexes, ion pairs) that violate the
        #          transport EN and cause O(1e4 mol/m³) oscillations at node i+1.
        # - Cl⁻, Na⁺, K⁺ : NOT read back. Those ions are handled correctly by the
        #          Nernst-Planck transport (ChemistryLab spreads them over complexes).
        V_liq = ustrip(state_eq.V_phases[].liquid)   # [m³]
        if !isnan(V_liq) && V_liq > 1e-12
            c_ca_new = max(ustrip(moles(state_eq, "Ca+2")) / V_liq, 0.0)
            Δc_ca    = c_ca_new - c_ca          # > 0 when dissolving (Ca(OH)₂ → Ca²⁺ + 2OH⁻)
            u[ICA, i] = c_ca_new
            u[IOH, i] = max(c_oh + 2.0 * Δc_ca, 0.0)  # EN-neutre : Δc_OH = +2 Δc_Ca
        end
    end
end

# ── Solve (operator splitting) ────────────────────────────────────────────────

"""
    run_Chloricem2b(; N, t_end, n_save, verbose) -> (results, model)

Simulates chloride penetration with portlandite dissolution.

# Strategy (SNIA)
1. NP transport (VoronoiFVM) over [tₙ, tₙ₊₁]
2. Chemistry (ChemistryLab) at tₙ₊₁ — node by node

# Returns
- `results` : `[(t, u_matrix, phi_vec, n_ch_vec), …]` at every output time
- `model`   : `ChloriceModel2b` with the final values (φ, n_CH)
"""
function run_Chloricem2b(;
    N       = 100,
    t_end   = 3.1536e7,
    n_save  = 12,
    verbose = false,
)
    m = ChloriceModel2b(N + 1)

    # ── Grid ──────────────────────────────────────────────────────────────────
    grid = simplexgrid(range(0.0, m.L; length = N + 1))

    # ── ChemistryLab: initialisation (once) ───────────────────────────────────
    cs, _ = init_chemistry(m, N + 1)

    # ── Adaptateurs VoronoiFVM ────────────────────────────────────────────────
    _storage!(f, u, node, data)  = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data)     = PoroMechanics.flux!(f, u, edge, m, data)
    _reaction!(f, u, node, data) = PoroMechanics.reaction!(f, u, node, m, data)
    _bcond!(f, u, node, data)    = PoroMechanics.bcondition!(f, u, node, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage    = _storage!,
        flux       = _flux!,
        reaction   = _reaction!,
        bcondition = _bcond!,
        species    = [ICL, INA, IK, IOH, ICA, IPS],
    )

    # ── Condition initiale ────────────────────────────────────────────────────
    inival = unknowns(sys)
    inival[ICL, :] .= m.c_cl_init
    inival[INA, :] .= m.c_na_init
    inival[IK,  :] .= m.c_k_init
    inival[IOH, :] .= m.c_oh_init
    inival[ICA, :] .= m.c_ca_init
    inival[IPS, :] .= m.psi_init
    # IC/BC consistency at node x = 0 (electroneutral external NaCl solution):
    # EN : -c_Cl + c_Na + c_K - c_OH + 2*c_Ca = 0
    # with c_Cl=523, c_Na=523, c_K=1, c_Ca≈0 → c_OH = 1 mol/m³
    inival[ICL, 1] = m.c_cl_BC
    inival[INA, 1] = m.c_na_BC
    inival[IK,  1] = m.c_k_BC
    inival[IOH, 1] = m.c_na_BC + m.c_k_BC - m.c_cl_BC   # = 1 mol/m³ (EN with c_Ca=0)
    inival[ICA, 1] = 0.0    # no portlandite in the external solution

    # ── Time step controller ──────────────────────────────────────────────────
    # Δt_max = t_end/(4*n_save): transport segments are subdivided to stay away
    # from Δt_min. A larger Δu_opt (50 % of the BC) allows wider steps without
    # triggering excessive subdivision.
    ctrl = VoronoiFVM.SolverControl(;
        Δt               = 1.0,
        Δt_max           = t_end / (4 * n_save),
        Δu_opt           = 0.5 * m.c_cl_BC,
        handle_exceptions = true,
        verbose          = verbose,
    )

    # ── Operator splitting loop ───────────────────────────────────────────────
    tsave   = range(0.0, t_end; length = n_save + 1)
    results = Tuple{Float64, Matrix{Float64}, Vector{Float64}, Vector{Float64}}[]
    u_cur   = copy(inival)

    for k in 2:lastindex(tsave)
        t0 = tsave[k-1]
        t1 = tsave[k]

        # Diagnostic before transport
        @info "pre-transport seg $k" cl_min=minimum(u_cur[ICL,:]) cl_max=maximum(u_cur[ICL,:]) na_min=minimum(u_cur[INA,:]) oh_min=minimum(u_cur[IOH,:]) ca_min=minimum(u_cur[ICA,:]) psi_min=minimum(u_cur[IPS,:]) psi_max=maximum(u_cur[IPS,:])

        # 1. Transport step (VoronoiFVM)
        seg   = solve(sys; inival = u_cur, times = [t0, t1], control = ctrl)
        u_cur = Matrix(seg[:, :, end])

        # 2. Chemical step (ChemistryLab)
        chemistry_step!(m, u_cur, cs)

        # Restore the Dirichlet conditions at the boundary node (x = 0):
        # the chemistry already skips node 1, but the BCs are re-imposed for safety.
        u_cur[ICL, 1] = m.c_cl_BC
        u_cur[INA, 1] = m.c_na_BC
        u_cur[IK,  1] = m.c_k_BC
        u_cur[IOH, 1] = m.c_oh_BC
        u_cur[ICA, 1] = m.c_ca_BC
        u_cur[IPS, 1] = m.psi_BC

        # NaN check after the chemistry
        if any(isnan, u_cur)
            nan_idx = findall(isnan, u_cur)
            @warn "NaN in u_cur after chemistry, seg $k" nan_idx=nan_idx[1:min(5,end)]
        end
        if any(isnan, m.phi)
            @warn "NaN in phi after chemistry, seg $k"
        end

        push!(results, (t1, copy(u_cur), copy(m.phi), copy(m.n_ch)))

        φ_mean = mean(m.phi)
        @info "SNIA" seg=k-1 t=round(t1/3.1536e7; digits=2) φ_mean=round(φ_mean; sigdigits=4) n_CH0=round(m.n_ch[1]; sigdigits=4)
    end

    return results, m
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_reference_2b(results, grid, m)

Prints the chloride profile at t_final and compares it with the C++ reference.
"""
function compare_reference_2b(results, grid)
    t, u, phi, n_ch = results[end]
    x_m  = grid[Coordinates][1, :]
    x_dm = x_m .* 10.0

    c_cl = u[ICL, :] ./ 1000.0
    c_oh = u[IOH, :] ./ 1000.0
    c_ca = u[ICA, :] ./ 1000.0

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    println("\nProfile at t = $(round(t/3.1536e7; digits=2)) year(s) — Phase 2b (NP + CH dissolution)")
    println("=" ^ 80)
    @printf("%-8s  %-14s  %-14s  %-10s  %-8s  %-8s\n",
            "x [dm]", "c_Cl [mol/dm³]", "Ref C++", "c_OH [mol/dm³]", "c_Ca", "φ [-]")
    println("-" ^ 80)
    for (xr, cr) in zip(x_ref, c_ref)
        idx = argmin(abs.(x_dm .- xr))
        @printf("%-8.4f  %-14.4f  %-14.4f  %-10.4f  %-8.4f  %-8.4f\n",
                xr, c_cl[idx], cr, c_oh[idx], c_ca[idx], phi[idx])
    end
    println("=" ^ 80)

    idx_front = findlast(c_cl .> 1e-3)
    x_front   = idx_front !== nothing ? x_dm[idx_front] : NaN
    @printf("\nDiagnostics Phase 2b :\n")
    @printf("  c_Cl(x=0) = %.4f mol/dm³  (ref: 0.523)\n", c_cl[1])
    @printf("  φ(x=0)    = %.4f  (initial = 0.121)\n", phi[1])
    @printf("  n_CH(x=0) = %.1f mol/m³  (initial = 1640)\n", n_ch[1])
    @printf("  Front Cl⁻ (c > 1e-3 mol/dm³) ≈ %.1f mm\n", x_front * 100.0)
    @printf("  C++ reference              ≈ 7.5 mm\n")
end

# ── Visualisation ─────────────────────────────────────────────────────────────

"""
    plot_Chloricem2b(results, grid; n_curves=4, save_path=nothing)

Plots three panels:
  - Left   : c_Cl profiles at n_curves times + C++ reference
  - Middle : porosity profiles φ(x) at the same times
  - Right  : ψ and c_OH profiles at the final time

Requires Plots.jl loaded in the session (`using Plots`).
"""
function plot_Chloricem2b(results, grid;
    n_curves  = 4,
    save_path = nothing,
)
    x_dm = grid[Coordinates][1, :] .* 10.0
    t_yr = 3.1536e7

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    n_t  = length(results)
    idxs = unique(clamp.(round.(Int, range(1, n_t; length = n_curves)), 1, n_t))

    palette = [:steelblue, :darkorange, :crimson, :forestgreen,
               :purple, :teal, :goldenrod, :indianred]

    # Panneau 1 : c_Cl
    p1 = plot(; xlabel="x [dm]", ylabel="c_Cl [mol/dm³]",
               title="Phase 2b — Cl⁻", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k, u_k, _, _ = results[ti]
        yrs = round(t_k / t_yr; digits = 2)
        plot!(p1, x_dm, u_k[ICL, :] ./ 1000;
            lw=2, color=palette[mod1(k, length(palette))], label="t = $yrs yr")
    end
        scatter!(p1, x_ref, c_ref; ms=5, color=:black, label="C++ ref (1 yr)")

    # Panel 2: porosity φ
    p2 = plot(; xlabel="x [dm]", ylabel="φ [-]",
               title="Phase 2b — porosity φ(x)", legend=:topright)
    for (k, ti) in enumerate(idxs)
        t_k, _, phi_k, _ = results[ti]
        yrs = round(t_k / t_yr; digits = 2)
        plot!(p2, x_dm, phi_k;
            lw=2, color=palette[mod1(k, length(palette))], label="t = $yrs yr")
    end
    hline!(p2, [0.121]; lw=1, ls=:dot, color=:gray50, label="φ₀ = 0.121")

    # Panel 3: c_OH and ψ at the final time
    t_fin, u_fin, _, _ = results[end]
    p3 = plot(x_dm, u_fin[IOH, :] ./ 1000;
        xlabel="x [dm]", ylabel="c_OH [mol/dm³]",
        title="Phase 2b — OH⁻ & ψ (t = $(round(t_fin/t_yr; digits=1)) yr)",
        lw=2, color=:darkorange, label="c_OH")
    plot!(twinx(p3), x_dm, u_fin[IPS, :] .* 1e3;
        ylabel="ψ [mV]", lw=2, ls=:dash, color=:steelblue, label="ψ")

    fig = plot(p1, p2, p3; layout=(1, 3), size=(1300, 430))
    save_path !== nothing && savefig(fig, save_path)
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    # Import Statistics for mean() in the loop
    using Statistics

    @info "Chloricem — Phase 2b : NP multi-ions + dissolution Ca(OH)₂ (ChemistryLab)"
    results, m_fin = run_Chloricem2b(; verbose=false)

    grid_ref = simplexgrid(range(0.0, 0.05; length=101))
    compare_reference_2b(results, grid_ref)

    try
        using Plots
        p = plot_Chloricem2b(results, grid_ref)
        display(p)
    catch e
        @warn "Plots.jl not available — plot skipped" exception=e
    end
end
