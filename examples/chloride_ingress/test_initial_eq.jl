# =============================================================================
# test_initial_eq.jl
#
# Initial thermodynamic equilibrium of an ordinary Portland cement (OPC) paste:
# identification of the stable phases, of the pore solution composition and of
# the porosity.
#
# C-S-H model chosen: CSHQ (Lothenbach & Nonat 2015) — 4 end-members
#   CSHQ-TobD, CSHQ-TobH, CSHQ-JenH, CSHQ-JenD
# The end-members of the other C-S-H models (CSH3T, ECSH-1, ECSH-2, Tob-I/II,
# discrete Jennite, CNASH) are excluded so that they do not interfere.
#
# Database: cemdata18 (Lothenbach et al. 2019, CCR 115:472–506)
#
# Usage :
#   julia --project examples/chloride_ingress/test_initial_eq.jl
#
# Dependencies (in the active environment):
#   ChemistryLab, OptimaSolver (or OptimizationIpopt), DynamicQuantities
# =============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))   # PoroMechanics.jl root

using ChemistryLab
using DynamicQuantities
using Printf

# ── Path to the cemdata18 database ────────────────────────────────────────────
# pkgdir returns the root directory of ChemistryLab (works in dev mode)
const CEMDATA18 = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")

# ── Solver: OptimaSolver (preferred) or Ipopt (fallback) ──────────────────────
# Uncomment the line matching the solver available in your environment.
using OptimaSolver
const OPTIMIZER = OptimaOptimizer(tol = 1.0e-10, verbose = false)
# using Optimization, OptimizationIpopt
# const OPTIMIZER = IpoptOptimizer(tol = 1.0e-10, print_level = 0)

# =============================================================================
# 1. Loading the database and selecting the species
# =============================================================================

println("── Loading cemdata18 ────────────────────────────────────────────────")
substances = build_species(CEMDATA18)
println("  $(length(substances)) species loaded")

# ── End-members to exclude: everything but CSHQ ───────────────────────────────
# For each alternative C-S-H model in cemdata18 we list its end-members so that
# they appear neither as discrete phases nor in the equilibrium residual.
# The LDH end-members (M4A-OH-LDH, M6A-OH-LDH) are excluded too, because the
# discrete "hydrotalcite" (Mg₆Al₂(OH)₁₄·4H₂O ~ M4AH10) is preferred.

const EXCLUDE_NONCSHQ = [
    # CSH3T (ternaire, C/S variable)
    "CSH3T-TobH", "CSH3T-T5C", "CSH3T-T2C",
    # ECSH-1 (with alkalis Na, K, Sr)
    "ECSH1-TobCa", "ECSH1-SH", "ECSH1-NaSH", "ECSH1-KSH", "ECSH1-SrSH",
    # ECSH-2 (with alkalis, OPC pore water version)
    "ECSH2-TobCa", "ECSH2-JenCa", "ECSH2-NaSH", "ECSH2-KSH",
    "ECSH2-SrSH", "ECSH2-SrSH(ACW)",
    # Tob-I / Tob-II (CSH-I / CSH-II model, 1 end-member per model)
    "Tob-I", "Tob-II",
    # Discrete Jennite (pure phase — duplicate of CSHQ-JenH/JenD)
    "Jennite",
    # CNASH (Ca-Na-Si ternary)
    "TobH-CNASHss", "T5C-CNASHss", "T2C-CNASHss",
    # LDH (OH⁻ solid solution) → the discrete "hydrotalcite" is kept instead
    "M4A-OH-LDH", "M6A-OH-LDH",
]

# ── Species selection strategy ────────────────────────────────────────────────
# Selecting the aqueous solutes (automatically, by elemental space) is kept
# deliberately separate from selecting the solid phases (an explicit list of the
# hydration products expected for an OPC).
# This avoids pulling in the dozens of polymorphic/metastable cemdata18 phases
# that satisfy the elemental space but never form in this context.

# Elemental space for the solutes: Ca, Si, Al, Fe, S, Mg, Na, K, O, H, C
# Na+ and K+ are added to model the cement alkalis (Na₂O, K₂O)
const SEEDS = split(
    "C3S C2S C3A C4AF Gp Portlandite H2O@ hydrotalcite ettringite Monocarbonate Na+ K+"
)

# Aqueous solutes — automatic speciation by atoms
aq_species = speciation(
    substances, SEEDS;
    aggregate_state  = [AS_AQUEOUS],
    exclude_species  = EXCLUDE_NONCSHQ,
)

# Phases solides — liste explicite : clinker + produits d'hydratation OPC
# Anhydrous clinker (reactants that may not be fully consumed)
const CLINKER_SOLIDS = ["C3S", "C2S", "C3A", "C4AF", "Gp"]
# C-S-H through CSHQ (solid solution end-members)
const CSHQ_END_MEMBERS = ["CSHQ-TobD", "CSHQ-TobH", "CSHQ-JenH", "CSHQ-JenD"]
# Other usual hydration products of an OPC
const OPC_HYDRATE_SOLIDS = [
    "Portlandite",     # Ca(OH)₂
    "ettringite",      # Ca₆Al₂(SO₄)₃(OH)₁₂·26H₂O  (early phase)
    "monosulphate12",  # C₄AS̄H₁₂  (replaces ettringite once SO₄ is exhausted)
    "monocarbonate05", # C₄AĈ₀.₅H₁₂  (traces of CO₂, formerly Hemicarbonate)
    "monocarbonate",   # C₄AĈH₁₁     (mild carbonation)
    "C3AH6",           # hydrogrenat Al  (end-member solution solide)
    "C3FH6",           # hydrogrenat Fe  (end-member solution solide)
    "C4AH13",          # tetracalcium aluminate hydrate
    "C4FH13",          # analogue fer
    "Fe-ettringite",   # ettringite Fe
    "Fe-monosulphate", # monosulfate Fe
    "hydrotalcite",    # Mg₆Al₂(OH)₁₄·4H₂O
]

solid_names = vcat(CLINKER_SOLIDS, CSHQ_END_MEMBERS, OPC_HYDRATE_SOLIDS)

# dict_species built here (reused in section 2)
dict_species = Dict(symbol(s) => s for s in substances)

solid_species = [dict_species[n] for n in solid_names if haskey(dict_species, n)]
missing_solids = filter(n -> !haskey(dict_species, n), solid_names)
if !isempty(missing_solids)
    missing_str2 = join(missing_solids, ", ")
    @warn "Phases not found in cemdata18 (ignored): $missing_str2"
end

# Final assembly: aqueous solutes + explicit solids
species = vcat(collect(aq_species), solid_species)

println("── Speciation ───────────────────────────────────────────────────────")
println("  $(length(aq_species)) aqueous solutes  +  $(length(solid_species)) explicit solid phases")
println("  = $(length(species)) species in total")

# =============================================================================
# 2. Solutions solides
# =============================================================================

# dict_species and CSHQ_END_MEMBERS are defined in section 1 — reused here.

# CSHQ — ideal solid solution with 4 end-members (Lothenbach & Nonat 2015)
ss_cshq = SolidSolutionPhase(
    "CSHQ",
    [dict_species[em] for em in CSHQ_END_MEMBERS];
    model = IdealSolidSolutionModel(),
)

# Al/Fe hydrogarnet — ideal binary solid solution (Lothenbach 2019)
ss_hydrogarnet = SolidSolutionPhase(
    "Hydrogarnet",
    [dict_species["C3AH6"], dict_species["C3FH6"]];
    model = IdealSolidSolutionModel(),
)

solid_solutions = [ss_cshq, ss_hydrogarnet]
println("── Solutions solides actives ────────────────────────────────────────")
for ss in solid_solutions
    ems = join(symbol.(end_members(ss)), ", ")
    println("  $(ChemistryLab.name(ss)) : $ems")
end

# =============================================================================
# 3. Chemical system
# =============================================================================

cs = ChemicalSystem(collect(species), CEMDATA_PRIMARIES; solid_solutions = solid_solutions)
println("── Chemical system ──────────────────────────────────────────────────")
println("  $(length(cs)) species, $(length(solutes(cs))) solutes, $(length(crystal(cs))) solid phases")
println("  $(length(solid_solutions)) solutions solides")

# =============================================================================
# 4. Initial composition: fresh OPC paste, w/c ratio = 0.50
# =============================================================================
# Typical OPC clinker (mass fractions normalised to 1 kg of clinker)
#   C3S  ~65 %, C2S ~15 %, C3A ~8 %, C4AF ~8 %, Gp ~4 %
# Added water: w/c = 0.50 relative to the clinker mass
# Everything is then normalised to 1 kg of total paste by rescale!.

const CLINKER_COMPO = [   # (symbol => mass fraction in the clinker)
    "C3S"  => 0.650,
    "C2S"  => 0.150,
    "C3A"  => 0.080,
    "C4AF" => 0.080,
    "Gp"   => 0.040,   # gypsum CaSO₄·2H₂O (set regulator)
]
const WC_RATIO = 0.50   # water / cement ratio (by mass)

# Soluble alkalis (Na₂O_eq ≈ 0.6 %, K₂O_eq ≈ 1.2 % of the clinker — typical OPC)
# Injected directly as Na⁺ and K⁺ in the pore solution
const NA2O_FRAC = 0.003   # Na₂O mass fraction in the clinker
const K2O_FRAC  = 0.006   # K₂O  mass fraction in the clinker

state = ChemicalState(cs)
m_clinker = sum(last.(CLINKER_COMPO))     # = 1.0 by construction
m_water   = WC_RATIO * m_clinker

for (sp, frac) in CLINKER_COMPO
    set_quantity!(state, sp, frac * u"kg")
end
set_quantity!(state, "H2O@", m_water * u"kg")

# Alkalis: Na₂O → 2 Na⁺ + O²⁻ (O absorbed into the water), K₂O → 2 K⁺ + O²⁻
# M(Na₂O) = 62 g/mol → NA2O_FRAC × m_clinker kg → ×1000/62 mol/kg × 2 Na+ per mol
n_Na = NA2O_FRAC * m_clinker * 1000.0 / 61.98 * 2.0   # mol
n_K  = K2O_FRAC  * m_clinker * 1000.0 / 94.20 * 2.0   # mol  (M(K₂O) = 94.2)
set_quantity!(state, "Na+", n_Na * u"mol")
set_quantity!(state, "K+",  n_K  * u"mol")

# Normalisation to 1 kg of total paste
rescale!(state, 1.0u"kg")

println("── Initial state ────────────────────────────────────────────────────")
println("  e/c = $WC_RATIO   T = 25 °C   P = 1 bar")
println("  Na₂O = $(round(NA2O_FRAC*100; digits=2)) %  K₂O = $(round(K2O_FRAC*100; digits=2)) %")

# =============================================================================
# 5. Equilibrium computation
# =============================================================================

println("\n── Equilibrium computation (Gibbs minimisation) ─────────────────────")
state_eq = equilibrate(state, OPTIMIZER; variable_space = Val(:linear))
println("  Done.")

# =============================================================================
# 6. Results
# =============================================================================

println("\n══════════════════════════════════════════════════════════════════════")
println("  RESULTS: stable phase assemblage")
println("══════════════════════════════════════════════════════════════════════")

# ── Solid phases with moles > threshold ───────────────────────────────────────
println("\n── Phases solides ───────────────────────────────────────────────────")
println("  $(rpad("Phase", 30)) $(rpad("n (mol)", 12)) V (cm³)")
println("  " * "─"^55)

# ChemistryLab stores its amounts in SymbolicDimensions → use us"..." throughout
threshold = 1e-6us"mol"
n = state_eq.n
sys_species = cs.species

for i in cs.idx_crystal
    sp = sys_species[i]
    ni = n[i]
    ni < threshold && continue
    Vi = if haskey(sp, :V⁰)
        sp[:V⁰](T = state_eq.T[], P = state_eq.P[]; unit = true)
    else
        nothing
    end
    n_mol  = ustrip(uconvert(us"mol", ni))
    Vi_str = if isnothing(Vi)
        "     n/a"
    else
        @sprintf("%8.3f", ustrip(uconvert(us"cm^3/mol", Vi)) * n_mol)
    end
    println("  $(rpad(symbol(sp), 30)) $(rpad(round(n_mol, digits=4), 12)) $Vi_str")
end

# ── Solution interstitielle ───────────────────────────────────────────────────
println("\n── Solution interstitielle ──────────────────────────────────────────")
println("  pH = $(round(state_eq.pH[], digits=2))")
println()
println("  $(rpad("Species", 20)) c (mol/L)")
println("  " * "─"^35)

# Liquid volume, needed to compute the concentrations
V_liq = state_eq.V_phases[].liquid

if !iszero(ustrip(V_liq))
    V_liq_L = uconvert(us"L", V_liq)
    for i in cs.idx_solutes
        sp = sys_species[i]
        ni = n[i]
        ci = ni / V_liq_L               # mol/L
        ustrip(ci) < 1e-7 && continue
        println("  $(rpad(symbol(sp), 20)) $(round(ustrip(ci), sigdigits=4))")
    end
else
    println("  (zero liquid volume — V⁰ missing for the solutes)")
end

# ── Volume summary ────────────────────────────────────────────────────────────
println("\n── Volume balance ───────────────────────────────────────────────────")
Vp = state_eq.V_phases[]
println("  V_liquide = $(round(ustrip(uconvert(us"cm^3", Vp.liquid)), digits=2)) cm³")
println("  V_solide  = $(round(ustrip(uconvert(us"cm^3", Vp.solid)),  digits=2)) cm³")
println("  V_total   = $(round(ustrip(uconvert(us"cm^3", Vp.total)),  digits=2)) cm³")
println("  Porosity  = $(round(state_eq.porosity[], digits=4))")

# =============================================================================
# 7. Initial values for run_3.jl
# =============================================================================
# Converts the results to mol/m³_concrete and mol/m³_water for the chloride_ingress
# Phase 3 transport model (V_REV = 1e-3 m³, phi = porosity).
# Concentrations are in mol/m³_water (= mol/L × 1000).

println("\n══════════════════════════════════════════════════════════════════════")
println("  INITIAL VALUES FOR run_3.jl")
println("══════════════════════════════════════════════════════════════════════")

phi_ic   = state_eq.porosity[]
V_paste  = ustrip(uconvert(us"cm^3", Vp.total))  * 1e-6   # m³ of paste normalised to 1 kg
V_liq_ic = ustrip(uconvert(us"cm^3", Vp.liquid)) * 1e-6   # m³ eau

function get_n_solid(name)
    haskey(dict_species, name) || return 0.0
    idx = findfirst(s -> symbol(s) == name, sys_species)
    idx === nothing && return 0.0
    ustrip(uconvert(us"mol", n[idx]))   # moles in 1 kg of normalised paste
end

function get_c_aq(name)
    haskey(dict_species, name) || return 0.0
    V_liq_ic == 0 && return 0.0
    idx = findfirst(s -> symbol(s) == name, sys_species)
    idx === nothing && return 0.0
    ni = ustrip(uconvert(us"mol", n[idx]))
    ni / V_liq_ic   # mol/m³_eau
end

# Solids (mol/m³_concrete): moles in the paste / V_total [m³]
n_ch_ic  = get_n_solid("Portlandite")    / V_paste
n_ett_ic = get_n_solid("ettringite")     / V_paste
n_ms_ic  = get_n_solid("monosulphate12") / V_paste

# Solution interstitielle (mol/m³_eau)
c_ca_ic  = get_c_aq("Ca+2")
c_na_ic  = get_c_aq("Na+")
c_k_ic   = get_c_aq("K+")
c_oh_ic  = get_c_aq("OH-")
c_so4_ic = get_c_aq("SO4-2")

@printf("\n  # To copy into compute_opc_ic() in run_3.jl:\n")
@printf("  phi0         = %.4f\n", phi_ic)
@printf("  n_ch0        = %.1f   # mol/m³_concrete — portlandite\n", n_ch_ic)
@printf("  n_ett_init   = %.1f   # mol/m³_concrete — ettringite\n",  n_ett_ic)
@printf("  n_ms_init    = %.1f   # mol/m³_concrete — monosulphate12\n", n_ms_ic)
@printf("  c_na0        = %.1f   # mol/m³_eau\n", c_na_ic)
@printf("  c_k0         = %.1f   # mol/m³_eau\n", c_k_ic)
@printf("  c_oh0        = %.2f  # mol/m³_eau  (pH = %.2f)\n", c_oh_ic, 14+log10(c_oh_ic*1e-3))
@printf("  c_ca0        = %.4f # mol/m³_eau\n", c_ca_ic)
@printf("  c_so4_0      = %.4f # mol/m³_eau\n", c_so4_ic)
