# M100 concrete exposed to seawater (≡SiOCaCl ternary complex, Yoshida 2021)
#
# This script defines the M100 specific case:
#   - CEM I 52.5N composition (Bogue-Rietveld mass fractions)
#   - mix design: 359 kg/m³ binder, w/c = 0.435
#   - exposure conditions: Atlantic seawater at 15 °C (Thermoddem 2023)
#   - initial hydration and entry point
#
# The generic infrastructure (ternary DLM, kinetics, SNIA, Fick transport,
# visualisation) is defined in chloride_ternary.jl.
#
# Usage:
#   julia --project examples/chloride_ingress/m100_ternary.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf
using Statistics
using ChemistryLab
using DynamicQuantities
using OptimaSolver
using Plots

include("physdata.jl")

# ════════════════════════════════════════════════════════════════════════════════
# Material data — M100 CEM I 52.5N concrete
# ════════════════════════════════════════════════════════════════════════════════

"Clinker mineralogical composition (Bogue mass fractions) and alkali content."
Base.@kwdef struct ClinkerComposition
    na2o_frac::Float64 = 0.0018   # Na₂O mass fraction  (CEM I 52.5N)
    k2o_frac::Float64  = 0.0028   # K₂O mass fraction
    f_C3S::Float64     = 0.6942   # alite
    f_C2S::Float64     = 0.0515   # belite
    f_C3A::Float64     = 0.1047   # tricalcium aluminate
    f_C4AF::Float64    = 0.0642   # ferroaluminate
    f_Gp::Float64      = 0.0709   # gypsum
end

"Concrete mix design: binder content and water-to-cement ratio."
Base.@kwdef struct MixDesign
    m_clinker::Float64 = 359.0   # [kg/m³_concrete]
    wc::Float64        = 0.435   # water/cement mass ratio
end

"""
Oh-Jang 2004 tortuosity parameters for M100 concrete (CEM I 52.5N, w/c = 0.435).
Calibrated on D_app(Cl⁻) = 1.45×10⁻¹² m²/s.

  `phi0`    : transport-accessible porosity (measured).
  `phi_c`   : capillary percolation threshold (Oh-Jang 2004).
  `n_OJ`    : Oh-Jang exponent.
  `ds_OJ`   : normalised gel/capillary diffusivity.
  `tau_agg` : aggregate obstruction factor (calibrated on D_app).
  `n_csh0`  : fixed C-S-H content [mol/m³_concrete] for DLM; 0 = dynamic.
"""
Base.@kwdef struct TransportModel
    phi0::Float64    = 0.182   # measured M100 porosity
    phi_c::Float64   = 0.10    # capillary percolation threshold
    n_OJ::Float64    = 2.0     # Oh-Jang exponent
    ds_OJ::Float64   = 0.015   # normalised gel/capillary diffusivity
    tau_agg::Float64 = 0.15    # aggregate tortuosity (calibrated M100, D_app=1.45e-12)
    n_csh0::Float64  = 0.0     # [mol_CSH/m³_concrete] — 0 = dynamic from CSHQ
end

"All data specific to the studied cementitious material."
Base.@kwdef struct CementMaterial
    clinker::ClinkerComposition = ClinkerComposition()
    mix::MixDesign              = MixDesign()
    transport::TransportModel   = TransportModel()
end

"""
Exposure conditions: temperature and aggressive solution composition.
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
    c_Si::Float64  = 0.005    # total Si Atlantic surface (~5 µmol/L)
    c_Al::Float64  = 1e-5     # Al(OH)₄⁻ seawater [mol/m³] (negligible)
end

include("chloride_ternary.jl")

# ── Initial hydration M100 ────────────────────────────────────────────────────

function _hydrate_m100_t(cs, mat::CementMaterial, env::ExposureConditions)
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

    phases_bogue = (("C3S", f_C3S, 0.22831), ("C2S", f_C2S, 0.17224),
        ("C3A", f_C3A, 0.27019), ("C4AF", f_C4AF, 0.48596), ("Gp", f_Gp, 0.17217))
    m_cem = m_clinker_kgm3 * V_REV_T
    m_eau = wc * m_cem
    n_eau = m_eau / 0.018015
    n_na = 2.0 * (m_cem * na2o_frac / 61.98e-3)
    n_k = 2.0 * (m_cem * k2o_frac / 94.20e-3)
    T_q = T_K * us"K"
    state = ChemicalState(cs; T=T_q)
    for (name, frac, mw) in phases_bogue
        set_quantity!(state, name, m_cem * frac / mw * us"mol")
    end
    set_quantity!(state, "H2O@", n_eau * us"mol")
    set_quantity!(state, "Na+", n_na * us"mol")
    set_quantity!(state, "K+", n_k * us"mol")
    set_quantity!(state, "OH-", (n_na + n_k) * us"mol")
    set_quantity!(state, "Cl-", 1e-16 * us"mol")
    set_quantity!(state, "Mg+2", 1e-16 * us"mol")

    @info "_hydrate_m100_t: M100 hydration equilibrium…"
    state_eq = equilibrate(state, OptimaOptimizer(tol=1e-10, verbose=false))

    V_liq = ustrip(uconvert(us"m^3", state_eq.V_phases[].liquid))
    V_liq < 1e-15 && error("Zero liquid volume after hydration")

    get_n(name) = max(ustrip(moles(state_eq, name)) / V_REV_T, 0.0)
    get_c(name) = ustrip(moles(state_eq, name)) / V_liq
    get_n_safe(name) = try; max(ustrip(moles(state_eq, name)) / V_REV_T, 0.0); catch; 0.0; end
    get_c_safe(name) = try; ustrip(moles(state_eq, name)) / V_liq; catch; 0.0; end

    n_ch = get_n("Portlandite")
    n_ett = get_n("ettringite")
    n_ms = get_n("monosulphate12")
    n_csh_tobh = get_n("CSHQ-TobH")
    n_csh_tobd = get_n("CSHQ-TobD")
    n_csh_jenh = get_n("CSHQ-JenH")
    n_csh_jend = get_n("CSHQ-JenD")
    n_cash_tobh = get_n_safe("CSH3T-TobH")
    n_cash_t5c  = get_n_safe("CSH3T-T5C")
    n_cash_t2c  = get_n_safe("CSH3T-T2C")
    c_ca = get_c("Ca+2")
    c_cl = max(get_c("Cl-"), 0.0)
    c_na = get_c("Na+")
    c_k = get_c("K+")
    c_oh = get_c("OH-")
    c_so4 = get_c("SO4-2")
    c_mg = get_c("Mg+2")
    c_si = sum(n * ustrip(moles(state_eq, sp)) for (sp, n) in SI_AQ_SPECIES_T
               if ustrip(moles(state_eq, sp)) > 0.0) / V_liq
    c_al = sum(n * max(get_c_safe(sp), 0.0) for (sp, n) in AL_AQ_SPECIES_T)

    pH = 14.0 + log10(max(c_oh, 1e-20) / 1000.0)
    n_csh_total = n_csh_tobh + n_csh_tobd + n_csh_jenh + n_csh_jend
    n_cash_total = n_cash_tobh + n_cash_t5c + n_cash_t2c
    @info "M100 hydration (ternary):" pH = round(pH; digits=2) n_CH = round(n_ch; digits=0) n_CSH = round(n_csh_total; digits=0) n_CASH = round(n_cash_total; digits=0)

    return (;
        phi=phi0, n_ch, n_ett, n_ms, n_fs=0.0, n_brc=0.0,
        n_csh_tobh, n_csh_tobd, n_csh_jenh, n_csh_jend,
        n_cash_tobh, n_cash_t5c, n_cash_t2c,
        n_msh_08=0.0, n_msh_13=0.0, n_ldh_m4=0.0, n_ldh_m6=0.0, n_ldh_m8=0.0, n_gyp=0.0,
        c_ca, c_cl, c_na, c_k, c_oh, c_so4, c_mg, c_si, c_al,
    )
end

# ── Constructor CementTernaryModel ──────────────────────────────────────────────

function CementTernaryModel(
    N_nodes::Int, cs_hyd;
    dlm::DLMTernaryParams    = DLMTernaryParams(),
    kin::KineticParams       = KineticParams(),
    mat::CementMaterial      = CementMaterial(),
    env::ExposureConditions  = ExposureConditions(),
    diff::IonicDiffusivities = IonicDiffusivities(),
)
    ic = _hydrate_m100_t(cs_hyd, mat, env)
    N  = N_nodes

    n_csh_init = ic.n_csh_tobh + ic.n_csh_tobd + ic.n_csh_jenh + ic.n_csh_jend
    n_cash_init = ic.n_cash_tobh + ic.n_cash_t5c + ic.n_cash_t2c
    n_csh_total_init = n_csh_init + n_cash_init
    x_cas_cshq0 = n_csh_init > 1e-6 ? (
        (5 / 6 * ic.n_csh_tobh + 5 / 6 * ic.n_csh_tobd + 9 / 6 * ic.n_csh_jenh + 10 / 6 * ic.n_csh_jend) / n_csh_init
    ) : 1.5
    x_cas_cash0 = n_cash_init > 1e-6 ? (
        (XCAS_CASH_TOBH * ic.n_cash_tobh + XCAS_CASH_T5C * ic.n_cash_t5c + XCAS_CASH_T2C * ic.n_cash_t2c) / n_cash_init
    ) : 1.5
    x_cas_init = n_csh_total_init > 1e-6 ? (
        (x_cas_cshq0 * n_csh_init + x_cas_cash0 * n_cash_init) / n_csh_total_init
    ) : 1.5
    n_csh_dlm_init = dlm.n_csh0 > 0.0 ? dlm.n_csh0 : n_csh_total_init
    _, S_Cl0, S_Na0, S_K0, S_Ca0, S_Mg0 = solve_dlm_ternary(
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca, ic.c_mg, ic.c_oh,
        n_csh_dlm_init, x_cas_init; dlm, T_K=env.T_K,
    )
    S_Cl0 = max(S_Cl0, 0.0)
    S_Na0 = max(S_Na0, 0.0)
    S_K0  = max(S_K0,  0.0)
    S_Ca0 = max(S_Ca0, 0.0)
    S_Mg0 = max(S_Mg0, 0.0)
    n_csh_dlm_init > 0.0 && @info "Initial ternary DLM:" n_csh = round(n_csh_dlm_init; digits=0) x_cas = round(x_cas_init; digits=2) S_Cl = round(S_Cl0; sigdigits=3) S_Na = round(S_Na0; sigdigits=3)

    # TODO: expose L as a keyword argument so specimen thickness can be varied
    # without editing this file. Currently hardcoded to 0.10 m (10 cm).
    return CementTernaryModel(
        0.10,
        fill(ic.phi, N), fill(ic.n_ch, N), fill(ic.n_ett, N), fill(ic.n_ms, N),
        fill(ic.n_fs, N), fill(ic.n_brc, N),
        fill(ic.n_csh_tobh, N), fill(ic.n_csh_tobd, N),
        fill(ic.n_csh_jenh, N), fill(ic.n_csh_jend, N),
        fill(ic.n_cash_tobh, N), fill(ic.n_cash_t5c, N), fill(ic.n_cash_t2c, N),
        fill(ic.n_msh_08, N), fill(ic.n_msh_13, N),
        fill(ic.n_ldh_m4, N), fill(ic.n_ldh_m6, N), fill(ic.n_ldh_m8, N),
        fill(ic.n_gyp, N),
        fill(ic.c_oh, N),
        fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N), fill(0.0, N),
        fill(S_Cl0, N), fill(S_Na0, N), fill(S_K0, N), fill(S_Ca0, N), fill(S_Mg0, N),
        dlm, kin, diff, mat, env,
        ic.c_cl, ic.c_na, ic.c_k, ic.c_ca, ic.c_mg, ic.c_so4, ic.c_si, ic.c_al,
    )
end

# ── SNIA driver ───────────────────────────────────────────────────────────────

"""
    run_fickian_diffusion00_ternary(; N, t_end, n_save, dlm, kwargs...) -> (results, model)

SNIA for M100 concrete — ternary complex ≡SiOCaCl + Yoshida 2021.
"""
function run_fickian_diffusion00_ternary(;
    N=400,
    t_end=3.1536e7,
    n_save=12,
    verbose=false,
    dlm::DLMTernaryParams    = DLMTernaryParams(),
    kin::KineticParams       = KineticParams(),
    mat::CementMaterial      = CementMaterial(),
    env::ExposureConditions  = ExposureConditions(),
    diff::IonicDiffusivities = IonicDiffusivities(),
)
    @info "Chemistry parallelism (ternary)" n_threads = Threads.nthreads()

    cs_hyd, cs_tr, has_afm_ss, has_friedels, has_brucite, has_msh, has_ldh, has_gyp, has_cash = _init_chemistry_ternary()

    let tr = mat.transport
        oj = OhJang(; phi_c = tr.phi_c, n = tr.n_OJ, ds = tr.ds_OJ, tau_agg = tr.tau_agg)
        tau = tortuosity(oj, tr.phi0, 1)
        D_eff = diff.D_Cl * tau
        D_app = D_eff * tr.phi0 / (tr.phi0 + 1.0)
        @info "M100 transport (ternary, phi=$(tr.phi0))" tau = round(tau; sigdigits=3) D_eff_Cl = round(D_eff; sigdigits=3) D_app_Cl = round(D_app; sigdigits=3)
    end

    m = CementTernaryModel(N + 1, cs_hyd; dlm, kin, mat, env, diff)
    grid = simplexgrid(range(0.0, m.L; length=N + 1))

    _storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data) = PoroMechanics.flux!(f, u, edge, m, data)
    _bcond!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage=_storage!, flux=_flux!, bcondition=_bcond!,
        species=[ICL_T, INA_T, IK_T, ICA_T, IMG_T, ISO4_T, ISI_T, IAL_T],
    )

    inival = unknowns(sys)
    inival[ICL_T, :]  .= m.c_cl_init
    inival[ICL_T,  1]  = m.env.c_Cl
    inival[INA_T, :]  .= m.c_na_init
    inival[INA_T,  1]  = m.env.c_Na
    inival[IK_T,  :]  .= m.c_k_init
    inival[IK_T,   1]  = m.env.c_K
    inival[ICA_T, :]  .= m.c_ca_init
    inival[ICA_T,  1]  = m.env.c_Ca
    inival[IMG_T, :]  .= m.c_mg_init
    inival[IMG_T,  1]  = m.env.c_Mg
    inival[ISO4_T, :] .= m.c_so4_init
    inival[ISO4_T, 1]  = m.env.c_SO4
    inival[ISI_T, :]  .= m.c_si_init
    inival[ISI_T,  1]  = m.env.c_Si
    inival[IAL_T, :]  .= m.c_al_init
    inival[IAL_T,  1]  = m.env.c_Al

    ctrl = VoronoiFVM.SolverControl(;
        Δt=1.0, Δt_max=t_end / (4 * n_save),
        Δu_opt=0.5 * m.env.c_Cl, handle_exceptions=true, verbose,
    )
    tsave = range(0.0, t_end; length=n_save + 1)
    results = Tuple{Float64,Matrix{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},
        Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64}}[]
    u_cur = copy(inival)

    for k in 2:lastindex(tsave)
        t0, t1 = tsave[k-1], tsave[k]
        @info "Transport segment $k (ternary)" t_yr = round(t1 / 3.1536e7; digits=2)

        seg = solve(sys; inival=u_cur, times=[t0, t1], control=ctrl)
        u_cur = Matrix(seg[:, :, end])

        Δt_seg = t1 - t0
        t_chem = @elapsed chemistry_step_ternary!(m, u_cur, cs_tr, has_afm_ss, has_friedels, has_brucite, has_msh, has_ldh, has_gyp, has_cash, Δt_seg)
        @info "Chemistry (ternary)" seg = k - 1 t_chem_s = round(t_chem; digits=1)

        u_cur[ICL_T,  1] = m.env.c_Cl
        u_cur[INA_T,  1] = m.env.c_Na
        u_cur[IK_T,   1] = m.env.c_K
        u_cur[ICA_T,  1] = m.env.c_Ca
        u_cur[IMG_T,  1] = m.env.c_Mg
        u_cur[ISO4_T, 1] = m.env.c_SO4
        u_cur[ISI_T,  1] = m.env.c_Si
        u_cur[IAL_T,  1] = m.env.c_Al
        m.c_oh_frozen[1] = 1.0
        m.Kd_Cl[1] = 0.0
        m.Kd_Mg[1] = 0.0
        m.Kd_SO4[1] = 0.0
        m.S_Cl_dlm[1] = 0.0
        m.S_Na_dlm[1] = 0.0
        m.S_K_dlm[1] = 0.0
        m.S_Ca_dlm[1] = 0.0
        m.S_Mg_dlm[1] = 0.0
        any(isnan, u_cur) && @warn "NaN after chemistry segment $k"

        push!(results, (
            t1, copy(u_cur), copy(m.phi),
            copy(m.n_ch), copy(m.n_ett), copy(m.n_ms), copy(m.n_fs), copy(m.n_brc),
            copy(m.c_oh_frozen),
            copy(m.n_csh_tobh), copy(m.n_csh_tobd), copy(m.n_csh_jenh), copy(m.n_csh_jend),
            copy(m.n_msh_08), copy(m.n_msh_13),
            copy(m.n_ldh_m4), copy(m.n_ldh_m6), copy(m.n_ldh_m8),
            copy(m.S_Cl_dlm), copy(m.S_Na_dlm), copy(m.S_K_dlm), copy(m.S_Ca_dlm),
            copy(m.n_gyp),
            copy(m.n_cash_tobh), copy(m.n_cash_t5c), copy(m.n_cash_t2c),
        ))
        @info "SNIA ternary" seg = k - 1 φ_mean = round(mean(m.phi); sigdigits=4) n_FS0 = round(m.n_fs[2]; sigdigits=4) n_BRC0 = round(m.n_brc[2]; sigdigits=4)
    end
    return results, m
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "M100 — ternary ≡SiOCaCl + Yoshida 2021 (SNIA: Fick + kinetics)"

    results, m_fin = run_fickian_diffusion00_ternary(;
        N=100, t_end=3.1536e7, n_save=12,
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
        kin=KineticParams(),
    )

    grid_ref = simplexgrid(range(0.0, m_fin.L; length=101))

    # Chloride balance at t = 1 year
    u_f = results[end][2]
    phi_f = results[end][3]
    n_fs_f = results[end][7]
    S_Cl_f = results[end][19]
    M_Cl = 35.453
    fac = M_Cl * 100.0 / 359_000.0
    x_dm = collect(range(0.0, 1.0; length=101))
    println("\n── Cl⁻ balance at t=1 year (ternary, g/100g cement) ──")
    @printf("%-8s  %-12s  %-12s  %-12s  %-12s\n", "x [dm]", "free Cl", "DLM Cl", "Friedel Cl", "Total")
    for xr in [0.005, 0.010, 0.020, 0.030, 0.050]
        idx = argmin(abs.(x_dm .- xr))
        cl_l = u_f[ICL_T, idx] * phi_f[idx] * fac
        cl_d = S_Cl_f[idx] * fac
        cl_f = 2.0 * n_fs_f[idx] * fac
        @printf("%-8.3f  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n", xr, cl_l, cl_d, cl_f, cl_l + cl_d + cl_f)
    end

    try
        figs = plot_M100_ternary(results, grid_ref;
            save_path="./examples/chloride_ingress/fig_M100_ternary_1yr.png")
        for f in figs
            display(f)
        end
    catch e
        @warn "Plots.jl not available" exception = e
    end
end
