# M6_drying/run.jl — 1D non-isothermal drying (VoronoiFVM.jl)
#
# Validation case: engineered barrier under a temperature rise.
# Test case: engineered barrier subjected to a temperature rise
#            (2 materials: compacted clay + host rock).
#
# Coupled PDEs (3 unknowns: p_l, p_a, T):
#   1. Conservation of total water (liquid + vapour)
#   2. Conservation of dry air
#   3. Bilan d'entropie ≡ bilan thermique
#
# Usage :
#   julia --project examples/M6_drying/run.jl
#
# Dependencies:  pkg> add VoronoiFVM ExtendableGrids

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using Printf

# ============================================================
# 1. Data structures
# ============================================================

"""
Parameters of one material for model M6 (non-isothermal drying).
Two instances are embedded in `M6Model`: clay (mat1) and rock (mat2).
"""
struct M6Mat
    phi::Float64       # porosity [-]
    k_int::Float64     # intrinsic permeability [m²]
    lam_s::Float64     # solid thermal conductivity [W/(m·K)]
    C_s::Float64       # volumetric heat capacity [J/(m³·K)]
    p_c3::Float64      # regularisation capillary pressure [Pa]
    mat_type::Int      # 1 = clay, 2 = rock (selects the retention curves)
end

"""
    M6Model

Model M6 — thermo-hydric drying of an unsaturated porous medium, 2 materials.

**Primary unknowns:** liquid pressure `p_l` [Pa], dry air pressure `p_a` [Pa],
temperature `T` [K].

**Geometry:** 1D domain `[0, L]` with an interface at `x_int` separating clay and rock.

**Boundary conditions:**
- `x = 0` : Neumann heat flux `Q(t)` [W/m²] (radioactive canister)
- `x = L` : Dirichlet `p_l`, `p_a`, `T` (initial values of the rock)
"""
Base.@kwdef struct M6Model <: AbstractPoroModel
    # ── Parameters common to both materials ───────────────────────────────────
    rho_l::Float64   = 1000.0       # liquid density [kg/m³]
    mu_l::Float64    = 1.0e-3       # liquid viscosity [Pa·s]
    mu_g::Float64    = 1.8e-5       # gas viscosity [Pa·s]
    M_vsR::Float64   = 0.00216      # M_vapeur/R [kg/J]
    M_asR::Float64   = 0.00346      # M_air/R [kg/J]
    p_l0::Float64    = 1.0e5        # reference liquid pressure [Pa]
    p_v0::Float64    = 2460.0       # saturated vapour pressure at T₀ [Pa]
    p_a0::Float64    = 97540.0      # reference air pressure [Pa]
    T_0::Float64     = 293.0        # reference temperature [K]
    D_av0::Float64   = 0.00248      # air-vapour diffusion at T₀ [m²/s]
    lam_l::Float64   = 0.6          # liquid thermal conductivity [W/(m·K)]
    lam_g::Float64   = 0.026        # gas thermal conductivity [W/(m·K)]
    C_pl::Float64    = 4180.0       # liquid specific heat [J/(kg·K)]
    C_pv::Float64    = 1800.0       # vapour specific heat [J/(kg·K)]
    C_pa::Float64    = 1000.0       # dry air specific heat [J/(kg·K)]
    L_0::Float64     = 2.45e6       # latent heat of vaporisation [J/kg]
    alpha_T::Float64 = 0.003        # thermal variation coeff. of S_l [1/K]

    # ── Geometry ───────────────────────────────────────────────────────────────
    x_int::Float64 = 0.425          # interface position [m]
    L::Float64     = 1.225          # total length [m]

    # ── Materials ──────────────────────────────────────────────────────────────
    mat1::M6Mat = M6Mat(0.30, 1.0e-20, 1.12, 2.3e6, 1.0e6, 1)   # compacted clay
    mat2::M6Mat = M6Mat(0.05, 1.0e-19, 1.62, 2.0e6, 2.0e5, 2)   # host rock

    # ── Initial conditions (reference case, fields 1-5) ───────────────────────
    p_l_ini1::Float64 = -7.611655e7  # p_l in the clay zone [Pa]
    p_a_ini1::Float64 =  9.225595e4  # p_a in the clay zone [Pa]
    p_l_ini2::Float64 =  4.905e6     # p_l in the rock zone [Pa]
    p_a_ini2::Float64 =  4.891671e6  # p_a in the rock zone [Pa]
    T_ini::Float64    =  323.0       # initial temperature (zones 1 and 2) [K]
end

PoroMechanics.nspecies(::M6Model)      = 3
PoroMechanics.species_names(::M6Model) = [:p_l, :p_a, :T]

# Indices of the unknowns (VoronoiFVM species)
const U_PL  = 1
const U_PA  = 2
const U_TEM = 3

# ============================================================
# 2. Retention curves (analytical)
# ============================================================

# ── Material 1 — compacted clay ───────────────────────────────────────────────

_Sl_raw1(pc::Real) = pc <= 0 ? 1.0 : (1.0 + (pc / 1.5e6)^1.06383)^(-0.06)

_dSl_raw1(pc::Real) = pc <= 0 ? 0.0 :
    let u = (pc / 1.5e6)^1.06383
        -0.06 * (1.0 + u)^(-1.06) * 1.06383 * u / pc
    end

_krl1(pc::Real) = pc <= 0 ? 1.0 : (1.0 + (pc / 3.0e6)^2)^(-0.5)

# ── Material 2 — host rock ────────────────────────────────────────────────────

_Sl_raw2(pc::Real) = pc <= 0 ? 1.0 : (1.0 + (pc / 10.0e6)^1.7)^(-0.4117)

_dSl_raw2(pc::Real) = pc <= 0 ? 0.0 :
    let u = (pc / 10.0e6)^1.7
        -0.4117 * (1.0 + u)^(-1.4117) * 1.7 * u / pc
    end

_krl2(pc::Real) = pc <= 0 ? 1.0 : (1.0 + (pc / 10.0e6)^2)^(-1.0)

# ── Gas relative permeability (common) ────────────────────────────────────────

_krg(sl::Real) = sl >= 1.0 ? 0.0 :
    let s = clamp(sl, 0.0, 1.0 - 1e-12)
        (1.0 - s)^2 * (1.0 - s^(5.0 / 3.0))
    end

# ── Regularised saturation (exponential branch for p_c < p_c3) ────────────────
# Exponential branch joined at p_c3, so the derivative stays bounded.

function _Sl_reg(pc::Real, p_c3::Real, Sl_raw::Function)
    pc <= 0    && return 1.0
    pc >= p_c3 && return Sl_raw(pc)
    sl3 = Sl_raw(p_c3)
    return 1.0 - (1.0 - sl3) * exp((pc - p_c3) / p_c3)
end

function _dSl_reg(pc::Real, p_c3::Real, Sl_raw::Function, dSl_raw::Function)
    pc <= 0    && return 0.0
    pc >= p_c3 && return dSl_raw(pc)
    sl3 = Sl_raw(p_c3)
    return -(1.0 - sl3) * exp((pc - p_c3) / p_c3) / p_c3
end

# ── Dispatch per material ─────────────────────────────────────────────────────

function _Sl(mat::M6Mat, pc::Real)
    mat.mat_type == 1 ? _Sl_reg(pc, mat.p_c3, _Sl_raw1) :
                        _Sl_reg(pc, mat.p_c3, _Sl_raw2)
end

function _dSl(mat::M6Mat, pc::Real)
    mat.mat_type == 1 ? _dSl_reg(pc, mat.p_c3, _Sl_raw1, _dSl_raw1) :
                        _dSl_reg(pc, mat.p_c3, _Sl_raw2, _dSl_raw2)
end

function _krl(mat::M6Mat, pc::Real)
    mat.mat_type == 1 ? _krl1(pc) : _krl2(pc)
end

function _get_mat(m::M6Model, x::Real)
    x < m.x_int ? m.mat1 : m.mat2
end

# ============================================================
# 3. Kelvin equation (vapour pressure)
# ============================================================

"""
Vapour pressure p_v(p_l, T) — modified Kelvin equation.
"""
function _p_vapor(m::M6Model, pl::Real, T::Real)
    θ = T - m.T_0
    return m.p_v0 * exp(m.M_vsR / T * (
        (pl - m.p_l0) / m.rho_l
        + m.L_0 * θ / m.T_0
        + (m.C_pl - m.C_pv) * (θ - T * log(T / m.T_0))
    ))
end

# ============================================================
# 4. Capillary entropy term (3-point Gauss quadrature)
#    Integral of dS_l/dT over the capillary pressure range.
# ============================================================

const _gauss_a = (0.93246951420, 0.66120938646, 0.23861918608)
const _gauss_w = (0.17132449237, 0.36076157304, 0.46791393457)

function _dSsdT(m::M6Model, mat::M6Mat, pc::Real, θ::Real)
    at = 1.0 - m.alpha_T * θ
    at <= 0 && return 0.0
    pc0 = pc / at
    return _dSl(mat, pc0) * pc0 * m.alpha_T / at
end

function _compute_dUsdT(m::M6Model, mat::M6Mat, pc::Real, θ::Real)
    pc <= 0 && return 0.0
    h  = pc / 2.0
    dU = 0.0
    for j in 1:3
        dU += _gauss_w[j] * (
            _dSsdT(m, mat, h * (1.0 + _gauss_a[j]), θ) +
            _dSsdT(m, mat, h * (1.0 - _gauss_a[j]), θ)
        )
    end
    return dU * h
end

# ============================================================
# 5. Heat flux imposed at x = 0 (radioactive waste)
#    Cumulative heat table F(t) [J/m²] of the reference case.
#    Instantaneous flux Q(t) = dF/dt.
# ============================================================

const _t_F  = [0.0, 3.1536e8, 6.3072e8, 9.4608e8, 1.26144e9, 1.5768e9,
               2.20752e9, 2.83824e9, 3.78432e9, 4.7304e9, 6.3072e9, 9.4608e9]
const _F_val = [0.0, 1.1006064e11, 1.978884e11, 2.6852904e11, 3.2734368e11,
                3.7504188e11, 4.4410572e11, 4.90779e11, 5.3902908e11,
                5.71526928e11, 6.09764328e11, 6.61641048e11]

"""Heat flux Q(t) [W/m²] injected at x=0 (piecewise-linear interpolation)."""
function _heat_flux(t::Real)
    t <= 0          && return 0.0
    t >= _t_F[end]  && return (_F_val[end] - _F_val[end-1]) /
                               (_t_F[end]  - _t_F[end-1])
    for i in 2:lastindex(_t_F)
        t <= _t_F[i] && return (_F_val[i] - _F_val[i-1]) / (_t_F[i] - _t_F[i-1])
    end
    return 0.0
end

# ── Time data (must live at module level, not inside a function) ──────────────
mutable struct M6Data
    t::Float64
end

# ============================================================
# 6. Interface PoroMechanics — callbacks VoronoiFVM
# ============================================================

"""
Storage term at a node.

- `f[U_PL]`  = M_l + M_v  — total water mass (liquid + vapour)
- `f[U_PA]`  = M_a         — dry air mass
- `f[U_TEM]` = S           — volumetric entropy
"""
function PoroMechanics.storage!(f, u, node, m::M6Model, ::Any)
    mat = _get_mat(m, node.coord[1])
    φ   = mat.phi
    Cs  = mat.C_s

    pl  = u[U_PL];  pa = u[U_PA];  T = u[U_TEM]
    θ   = T - m.T_0
    pv  = _p_vapor(m, pl, T)
    pg  = pv + pa
    pc  = pg - pl
    at  = 1.0 - m.alpha_T * θ
    pc0 = pc / at
    sl  = _Sl(mat, pc0);  sg = 1.0 - sl

    ρv  = pv * m.M_vsR / T
    ρa  = pa * m.M_asR / T
    Ml  = m.rho_l * φ * sl
    Mv  = ρv * φ * sg
    Ma  = ρa * φ * sg

    s_l = m.C_pl * log(T / m.T_0)
    s_v = m.C_pv * log(T / m.T_0) - log(pv / m.p_v0) / m.M_vsR + m.L_0 / m.T_0
    s_a = m.C_pa * log(T / m.T_0) - log(pa / m.p_a0) / m.M_asR
    dU  = _compute_dUsdT(m, mat, pc, θ)

    f[U_PL]  = Ml + Mv
    f[U_PA]  = Ma
    f[U_TEM] = Cs * log(T / m.T_0) - φ * dU + Ml * s_l + Mv * s_v + Ma * s_a
end

"""
Numerical flux on an edge.

Convention VoronoiFVM : `f[s] = K·(u₁ - u₂)` ↔ `W = f[s]/(x₂ - x₁)`.

- `f[U_PL]`  = W_l + W_v   — flux d'eau total (Darcy + Fick)
- `f[U_PA]`  = W_a          — flux d'air sec
- `f[U_TEM]` = J_s          — flux d'entropie = Q/T + s_l·W_l + s_v·W_v + s_a·W_a
"""
function PoroMechanics.flux!(f, u, edge, m::M6Model, ::Any)
    x_m = (edge.coord[1, 1] + edge.coord[1, 2]) / 2.0
    mat = _get_mat(m, x_m)
    φ   = mat.phi
    ki  = mat.k_int
    λs  = mat.lam_s

    pl1, pl2 = u[U_PL,  1], u[U_PL,  2]
    pa1, pa2 = u[U_PA,  1], u[U_PA,  2]
    T1,  T2  = u[U_TEM, 1], u[U_TEM, 2]
    plm = (pl1 + pl2) / 2.0
    pam = (pa1 + pa2) / 2.0
    Tm  = (T1  + T2 ) / 2.0

    θm   = Tm - m.T_0
    at   = 1.0 - m.alpha_T * θm
    pvm  = _p_vapor(m, plm, Tm)
    pgm  = pvm + pam
    pcm  = pgm - plm
    pc0m = pcm / at
    slm  = _Sl(mat, pc0m);  sgm = 1.0 - slm

    ρvm = pvm * m.M_vsR / Tm
    ρam = pam * m.M_asR / Tm
    ρgm = ρvm + ρam
    cvm = ρvm / ρgm;  cam = 1.0 - cvm

    # Darcy conductivities
    Kl  = m.rho_l * ki / m.mu_l * _krl(mat, pc0m)
    krg = _krg(slm)
    KDv = ρvm * ki / m.mu_g * krg
    KDa = ρam * ki / m.mu_g * krg

    # Fick diffusion — Millington-Quirk tortuosity
    τ    = φ^(1.0 / 3.0) * max(sgm, 0.0)^(7.0 / 3.0)
    Dav  = m.D_av0 * (m.p_v0 + m.p_a0) / pgm * (Tm / m.T_0)^1.88
    Def  = φ * sgm * τ * Dav
    KFv  = ρgm * Def;  KFa = KFv
    bar  = Def * cvm * cam / Tm
    KDv += bar * (m.M_asR - m.M_vsR)
    KDa += bar * (m.M_vsR - m.M_asR)

    # Thermal conductivity — Johansen geometric mean
    KTH = λs^(1.0 - φ) * m.lam_l^(φ * slm) * m.lam_g^(φ * sgm)

    # Pressures and mass fractions at the nodes
    pv1 = _p_vapor(m, pl1, T1);  pg1 = pv1 + pa1
    pv2 = _p_vapor(m, pl2, T2);  pg2 = pv2 + pa2
    ρg1 = pv1 * m.M_vsR / T1 + pa1 * m.M_asR / T1
    ρg2 = pv2 * m.M_vsR / T2 + pa2 * m.M_asR / T2
    cv1 = pv1 * m.M_vsR / T1 / ρg1;  ca1 = 1.0 - cv1
    cv2 = pv2 * m.M_vsR / T2 / ρg2;  ca2 = 1.0 - cv2

    # Elementary fluxes
    Wl = Kl  * (pl1 - pl2)
    Wv = KDv * (pg1 - pg2) + KFv * (cv1 - cv2)
    Wa = KDa * (pg1 - pg2) + KFa * (ca1 - ca2)

    # Flux d'entropie
    s_lm = m.C_pl * log(Tm / m.T_0)
    s_vm = m.C_pv * log(Tm / m.T_0) - log(pvm / m.p_v0) / m.M_vsR + m.L_0 / m.T_0
    s_am = m.C_pa * log(Tm / m.T_0) - log(pam / m.p_a0) / m.M_asR
    Js   = KTH / Tm * (T1 - T2) + s_lm * Wl + s_vm * Wv + s_am * Wa

    f[U_PL]  = Wl + Wv
    f[U_PA]  = Wa
    f[U_TEM] = Js
end

"""
Boundary conditions:
- Region 1 (x = 0)  : Neumann heat flux `Q(t) / T_node`
- Region 2 (x = L)  : Dirichlet `p_l`, `p_a`, `T` (initial rock values)

`data.t` must hold the current time (updated before each segment).
"""
function PoroMechanics.bcondition!(f, u, node, m::M6Model, data)
    # Right (x = L): Dirichlet, initial values of zone 2
    boundary_dirichlet!(f, u, node; species = U_PL,  region = 2, value = m.p_l_ini2)
    boundary_dirichlet!(f, u, node; species = U_PA,  region = 2, value = m.p_a_ini2)
    boundary_dirichlet!(f, u, node; species = U_TEM, region = 2, value = m.T_ini)

    # Left (x = 0): incoming entropy flux = Q / T
    # VoronoiFVM convention: boundary_neumann!(value = v) does f[s] -= v,
    # which adds +v to the residual → v > 0 for an incoming entropy source.
    if node.region == 1
        Q = _heat_flux(data.t)
        boundary_neumann!(f, u, node; species = U_TEM, region = 1,
                          value = Q / max(u[U_TEM], 200.0))
    end
end

# ============================================================
# 7. Simulation
# ============================================================

"""
    run_M6(; Nx1=43, Nx2=80, n_years=100, verbose=false)

Solve the M6 non-isothermal drying problem and return
`(model, x_all, results)` where `results` is a vector of tuples
`(t [s], u [n_species × n_nodes])`.

# Arguments
- `Nx1` : number of nodes in the clay zone `[0, x_int]`
- `Nx2` : number of nodes in the rock zone `[x_int, L]`
- `n_years` : maximum simulated duration [years]
- `verbose` : print per-segment diagnostics

# Returns
- `model`   : an `M6Model` instance
- `x_all`   : vector of nodal positions [m]
- `results` : `[(t₁, u₁), …, (tₙ, uₙ)]`
"""
function run_M6(; Nx1=43, Nx2=80, n_years=100, verbose=false)

    m = M6Model()

    # ── Two-material grid ─────────────────────────────────────────────────────
    x1    = collect(range(0.0, m.x_int; length = Nx1))
    x2    = collect(range(m.x_int, m.L; length = Nx2))
    x_all = vcat(x1, x2[2:end])
    grid  = simplexgrid(x_all)
    grid[CellRegions][1:Nx1-1] .= 1    # clay zone
    grid[CellRegions][Nx1:end] .= 2    # rock zone

    # ── Adapters (closures over the model) ────────────────────────────────────
    data = M6Data(0.0)

    _storage!(f, u, node, d)  = PoroMechanics.storage!(f, u, node, m, d)
    _flux!(f, u, edge, d)     = PoroMechanics.flux!(f, u, edge, m, d)
    _bcond!(f, u, node, d)    = PoroMechanics.bcondition!(f, u, node, m, d)

    sys = VoronoiFVM.System(
        grid;
        storage    = _storage!,
        flux       = _flux!,
        bcondition = _bcond!,
        species    = [U_PL, U_PA, U_TEM],
        data       = data,
    )

    # ── Condition initiale ────────────────────────────────────────────────────
    inival = unknowns(sys)
    for i in eachindex(x_all)
        if x_all[i] < m.x_int
            inival[U_PL, i] = m.p_l_ini1
            inival[U_PA, i] = m.p_a_ini1
        else
            inival[U_PL, i] = m.p_l_ini2
            inival[U_PA, i] = m.p_a_ini2
        end
        inival[U_TEM, i] = m.T_ini
    end
    # IC/BC consistency on the right
    inival[U_PL,  end] = m.p_l_ini2
    inival[U_PA,  end] = m.p_a_ini2
    inival[U_TEM, end] = m.T_ini

    # ── Output times ──────────────────────────────────────────────────────────
    yr    = 3.1536e7   # 1 year [s]
    tsave = [0.0, yr, 2yr, 4yr, 6yr, 8yr, 10yr, 20yr, 40yr, 50yr, n_years*yr]

    # ── Solver parameters (tuned on the reference case) ──────────────────────
    ctrl = VoronoiFVM.SolverControl(;
        Δt      = 1.0e2,        # Dtini [s]
        Δt_max  = 3.1536e7,     # Dtmax [s] = 1 year
        Δt_min  = 1.0e-2,
        Δu_opt  = 1.0e4,        # OBJE p_l = p_a = 1e4 Pa, tem = 1 K
        reltol  = 1.0e-6,
        abstol  = 1.0e-10,
        verbose = false,
    )

    # ── Time loop — data.t updated before each segment ────────────────────────
    results = Tuple{Float64, Matrix{Float64}}[]
    u_cur   = copy(inival)

    for k in 2:lastindex(tsave)
        t0 = tsave[k-1]
        t1 = tsave[k]
        data.t = t1

        seg   = solve(sys; inival = u_cur, times = [t0, t1], control = ctrl)
        u_new = seg[:, :, end]

        if verbose
            Q_val   = _heat_flux(t1)
            dT_max  = maximum(u_new[U_TEM, :]) - m.T_ini
            du_max  = maximum(abs.(u_new .- u_cur))
            @printf("  seg %2d [%.2ea → %.2ea s] : Q=%5.0f W/m²  ΔT_max=%+7.3f K  Δu=%g\n",
                    k - 1, t0, t1, Q_val, dT_max, du_max)
        end

        u_cur = u_new
        push!(results, (t1, copy(u_cur)))
    end

    return m, x_all, results
end

# ============================================================
# 8. Post-traitement
# ============================================================

"""
Print a summary table and a physical check of the M6 result.
"""
function print_summary(m::M6Model, x_all::Vector, results)
    yr     = 3.1536e7
    Nx1_idx = argmin(abs.(x_all .- m.x_int))      # node ≈ interface
    i_mid2  = argmin(abs.(x_all .- 0.825))         # node ≈ 0.825 m

    println("\nM6 — 1D non-isothermal drying (VoronoiFVM.jl)")
    println("Grid: $(length(x_all)) nodes  |  interface at x = $(m.x_int) m\n")

    println("t (yr)  | p_l[int] (Pa)    | T[int] (K)  | S_l[int] (-) | T[0.825m] (K)")
    println("-"^72)

    sl_vec = Float64[]
    for (t, u) in results
        pl = u[U_PL, Nx1_idx];  pa = u[U_PA, Nx1_idx];  T = u[U_TEM, Nx1_idx]
        pv  = _p_vapor(m, pl, T)
        pc0 = (pv + pa - pl) / (1.0 - m.alpha_T * (T - m.T_0))
        sl  = _Sl(_get_mat(m, x_all[Nx1_idx]), pc0)
        push!(sl_vec, sl)
        Tm2 = u[U_TEM, i_mid2]
        @printf("%6.0f  | %+14.4e  | %10.3f  | %11.6f  | %10.3f\n",
                t / yr, pl, T, sl, Tm2)
    end

    # Physical check
    T_ini_int = results[1][2][U_TEM, Nx1_idx]
    T_fin_int = results[end][2][U_TEM, Nx1_idx]
    println("\nPhysical check at the interface (x ≈ $(round(x_all[Nx1_idx]; digits=3)) m):")
    @printf("  T   : %.3f K  →  %.3f K\n", T_ini_int, T_fin_int)
    @printf("  S_l : %.6f  →  %.6f\n",    sl_vec[1], sl_vec[end])
    T_fin_int > T_ini_int ?
        println("  OK: temperature increased (heating confirmed)") :
        println("  WARNING: temperature did not increase")
    sl_vec[end] < sl_vec[1] ?
        println("  OK: saturation decreased (thermal drying confirmed)") :
        println("  WARNING: saturation did not decrease")
end

# ============================================================
# 9. Entry point
# ============================================================

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Starting the M6 simulation (1D non-isothermal drying)…"
    m, x_all, results = run_M6(; verbose=true)
    print_summary(m, x_all, results)

    # ── Plot (uncomment if Plots.jl is available) ───────────────────────────
    # using Plots
    # yr = 3.1536e7
    # colors = [:steelblue, :darkorange, :crimson, :green]
    # times_plot = [1yr, 10yr, 40yr, 100yr]
    # p_T  = plot(; xlabel="x [m]", ylabel="T [K]",   title="M6 — temperature profiles",  legend=:topright)
    # p_sl = plot(; xlabel="x [m]", ylabel="Sₗ [-]",  title="M6 — saturation profiles",   legend=:topright)
    # for (k, t_tgt) in enumerate(times_plot)
    #     idx = argmin(abs.([r[1] for r in results] .- t_tgt))
    #     u   = results[idx][2]
    #     T_prof  = [u[U_TEM, i] for i in 1:length(x_all)]
    #     sl_prof = [begin
    #         pl = u[U_PL,i]; pa = u[U_PA,i]; T = u[U_TEM,i]
    #         pv  = _p_vapor(m, pl, T)
    #         pc0 = (pv + pa - pl) / (1 - m.alpha_T*(T - m.T_0))
    #         _Sl(_get_mat(m, x_all[i]), pc0)
    #     end for i in 1:length(x_all)]
    #     yrs = round(Int, t_tgt / yr)
    #     plot!(p_T,  x_all, T_prof;  lw=2, color=colors[k], label="t = $yrs yr")
    #     plot!(p_sl, x_all, sl_prof; lw=2, color=colors[k], label="t = $yrs yr")
    # end
    # vline!(p_T,  [m.x_int]; lw=1, ls=:dash, color=:black, label="interface")
    # vline!(p_sl, [m.x_int]; lw=1, ls=:dash, color=:black, label="interface")
    # display(plot(p_T, p_sl; layout=(1,2), size=(900, 380)))
end
