# chloride_ingress example — Phase 2a: multi-species Nernst-Planck transport + Langmuir
#
# Validation: chloride profile against a published C++ reference solution.
# Phase 2a: saturated medium (s_l = 1), 4 ions (Cl⁻, Na⁺, K⁺, OH⁻) + potential ψ,
#           electroneutrality constraint, no dissolution of the solids.
#
# System solved (5 unknowns per node):
#   ∂/∂t [φ·c_i + δ_{i,Cl}·n_ads(c_Cl)] = -∇·J_i  for i ∈ {Cl⁻, Na⁺, K⁺, OH⁻}
#   0 = -c_Cl + c_Na + c_K - c_OH                  (electroneutrality → determines ψ)
#
# Nernst-Planck flux (Scharfetter-Gummel discretisation):
#   J_i = -D_i_eff · (∇c_i + z_i · F/(RT) · c_i · ∇ψ)
#   D_i_eff = D_i_free · τ(φ)    (Oh-Jang tortuosity)
#
# Boundary conditions:
#   x = 0 : Dirichlet c_Cl = c_BC, Dirichlet ψ = 0  (potential reference)
#            Na⁺, K⁺, OH⁻ → zero Neumann (free to evolve, no fixed external reservoir)
#   x = L : zero Neumann for every species
#
# Initial condition (concrete pore solution, electroneutral):
#   c_Cl = 0.01, c_Na = 150, c_K = 300, c_OH = 449.99  [mol/m³]
#   ψ = 0 [V]
#
# Usage :
#   julia --project examples/chloride_ingress/run_2a.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf

# ── Species indices ───────────────────────────────────────────────────────────

const ICL = 1   # Cl⁻   z = -1
const INA = 2   # Na⁺   z = +1
const IK  = 3   # K⁺    z = +1
const IOH = 4   # OH⁻   z = -1
const IPS = 5   # ψ     algebraic equation (electroneutrality)

# Ionic charges (same indices as the species)
const Z_IONS = (-1.0, +1.0, +1.0, -1.0)   # z_Cl, z_Na, z_K, z_OH

# ── Physical parameters ───────────────────────────────────────────────────────

"""
Parameters of the chloride_ingress model — Phase 2a (multi-species Nernst-Planck).

All quantities are SI (m, s, mol, mol/m³, V).
"""
Base.@kwdef struct ChlorideModel2a <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64       = 0.05           # [m] (= 0.5 dm)

    # ── Porosity and C-S-H (constant in Phase 2a) ─────────────────────────────
    phi::Float64     = 0.121
    n_csh::Float64   = 635.0          # [mol/m³]
    alpha::Float64   = 3.192
    beta::Float64    = 0.0266         # [m³/mol]

    # ── Ionic diffusion coefficients in free water [m²/s] ─────────────────────
    # Values at 25 °C after Robinson & Stokes (1959) and the CRC Handbook.
    D_Cl::Float64    = 2.032e-9
    D_Na::Float64    = 1.334e-9
    D_K::Float64     = 1.957e-9
    D_OH::Float64    = 5.273e-9

    # ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────
    phi_c::Float64   = 0.18
    n_OJ::Float64    = 2.7
    ds_OJ::Float64   = 2.0e-4
    tau_agg::Float64 = 0.27

    # ── Constantes physiques ──────────────────────────────────────────────────
    Faraday::Float64 = 96485.0        # [C/mol]
    R_gas::Float64   = 8.314          # [J/(mol·K)]
    T_K::Float64     = 293.15         # [K]  (≈ 20°C)

    # ── Boundary conditions at x = 0 ──────────────────────────────────────────
    c_cl_BC::Float64 = 523.0          # [mol/m³]  imposed Cl⁻ (seawater)
    psi_BC::Float64  = 0.0            # [V]       reference potential

    # ── Initial conditions (concrete pore solution) ───────────────────────────
    # Electroneutrality: -c_Cl + c_Na + c_K - c_OH = 0
    # → c_OH = c_Na + c_K - c_Cl = 150 + 300 - 0.01 = 449.99
    c_cl_init::Float64 =   0.01
    c_na_init::Float64 = 150.0
    c_k_init::Float64  = 300.0
    c_oh_init::Float64 = 449.99       # exactly electroneutral
    psi_init::Float64  =   0.0
end

PoroMechanics.nspecies(::ChlorideModel2a) = 5
PoroMechanics.species_names(::ChlorideModel2a) = [:c_cl, :c_na, :c_k, :c_oh, :psi]

# ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────────

"""
    tortuosity_OhJang(phi, s_l, m::ChlorideModel2a)

Oh-Jang tortuosity of the cement paste, from the package constitutive layer.
"""
function tortuosity_OhJang(phi, s_l, m::ChlorideModel2a)
    oj = OhJang(; phi_c = m.phi_c, n = m.n_OJ, ds = m.ds_OJ, tau_agg = m.tau_agg)
    return tortuosity(oj, phi, s_l)
end

# Effective diffusion coefficient of ion i [m²/s], saturation = 1
D_eff_ion(D_free::Float64, m::ChlorideModel2a) =
    D_free * tortuosity_OhJang(m.phi, 1.0, m)

# ── Langmuir adsorption (Cl⁻ only) ────────────────────────────────────────────

"""
    n_ads(c_cl, m)

Amount of adsorbed Cl⁻ [mol/m³] — Langmuir isotherm (curve Adscl).
Facteur 1000 : normalisation implicite c_ref = 1 mol/dm³ = 1000 mol/m³.
"""
n_ads(c::T, m::ChlorideModel2a) where {T<:Real} =
    T(m.n_csh) * T(m.alpha) * c / (T(1000.0) * (1 + T(m.beta) * c))

# ── Bernoulli function (Scharfetter-Gummel) ───────────────────────────────────

"""
    bernoulli(x)

Bernoulli function B(x) = x / (exp(x) - 1).
Numerically stable through a Taylor expansion for |x| < 1e-7.
ForwardDiff-compatible: the comparison is made on the primal value.
"""
function bernoulli(x::T) where {T<:Real}
    if abs(x) < 1.0e-7
        return one(T) - x / 2
    end
    return x / (exp(x) - one(T))
end

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

"""
Storage term:
  Cl⁻ : φ·c_Cl + n_ads(c_Cl)   (in solution + Langmuir-adsorbed)
  Na⁺, K⁺, OH⁻ : φ·c_i
  ψ : 0                          (algebraic equation, no time derivative)
"""
function PoroMechanics.storage!(f, u, ::Any, m::ChlorideModel2a, ::Any)
    f[ICL] = m.phi * u[ICL] + n_ads(u[ICL], m)
    f[INA] = m.phi * u[INA]
    f[IK]  = m.phi * u[IK]
    f[IOH] = m.phi * u[IOH]
    f[IPS] = zero(eltype(u))
end

"""
Electroneutrality constraint (algebraic equation for ψ):
    -c_Cl + c_Na + c_K - c_OH = 0
"""
function PoroMechanics.reaction!(f, u, ::Any, ::ChlorideModel2a, ::Any)
    f[IPS] = -u[ICL] + u[INA] + u[IK] - u[IOH]
end

"""
Nernst-Planck flux, discretised with Scharfetter-Gummel.

For ion i between nodes 1 and 2:
  dV   = z_i · F/(RT) · (ψ₁ - ψ₂)
  f[i] = D_i_eff · [B(dV)·c_i,1 - B(-dV)·c_i,2]

VoronoiFVM convention: f = K·(u₁ - u₂); the divergence is taken by the solver.
For ψ: no spatial flux (f[IPS] = 0).
"""
function PoroMechanics.flux!(f, u, ::Any, m::ChlorideModel2a, ::Any)
    FoRT = m.Faraday / (m.R_gas * m.T_K)   # F/(RT)  [V⁻¹]  ≈ 38.9 V⁻¹ at 20 °C
    Δψ   = u[IPS, 1] - u[IPS, 2]

    D_ions = (D_eff_ion(m.D_Cl, m),
              D_eff_ion(m.D_Na, m),
              D_eff_ion(m.D_K,  m),
              D_eff_ion(m.D_OH, m))

    for (idx, (zi, Di)) in enumerate(zip(Z_IONS, D_ions))
        dV   = zi * FoRT * Δψ
        f[idx] = Di * (bernoulli(dV) * u[idx, 1] - bernoulli(-dV) * u[idx, 2])
    end

    f[IPS] = zero(eltype(u))
end

"""
Boundary conditions at x = 0 (region 1):
  - c_Cl = c_BC  (Dirichlet — seawater concentration)
  - ψ = 0        (Dirichlet — reference potential)
  - Na⁺, K⁺, OH⁻ : zero Neumann (their value follows from electroneutrality + diffusion)

x = L (region 2): zero Neumann by default for every species.
"""
function PoroMechanics.bcondition!(f, u, bnode, m::ChlorideModel2a, ::Any)
    boundary_dirichlet!(f, u, bnode; species=ICL, region=1, value=m.c_cl_BC)
    boundary_dirichlet!(f, u, bnode; species=IPS, region=1, value=m.psi_BC)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

"""
    run_chloride_ingress2a(; N, t_end, n_save, verbose) -> (tsol, grid, model)

Simulates multi-ionic penetration (Nernst-Planck) into a saturated concrete.

# Arguments
- `N`       : number of elements (default 100)
- `t_end`   : duration [s]  (default 3.1536e7 ≈ 1 year)
- `n_save`  : number of saved profiles
- `verbose` : VoronoiFVM solver diagnostics

# Returns
- `tsol`  : solution transiente (VoronoiFVM.TransientSolution)
- `grid`  : 1D grid
- `model` : the ChlorideModel2a instance
"""
function run_chloride_ingress2a(;
    N       = 100,
    t_end   = 3.1536e7,
    n_save  = 12,
    verbose = false,
)
    m = ChlorideModel2a()

    # ── Uniform 1D grid ───────────────────────────────────────────────────────
    grid = simplexgrid(range(0.0, m.L; length = N + 1))

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
        species    = [ICL, INA, IK, IOH, IPS],
    )

    # ── Condition initiale ────────────────────────────────────────────────────
    # Concrete pore solution, electroneutral:
    # -c_Cl + c_Na + c_K - c_OH = -0.01 + 150 + 300 - 449.99 = 0  ✓
    inival = unknowns(sys)
    inival[ICL, :] .= m.c_cl_init
    inival[INA, :] .= m.c_na_init
    inival[IK,  :] .= m.c_k_init
    inival[IOH, :] .= m.c_oh_init
    inival[IPS, :] .= m.psi_init

    # IC/BC consistency at node x = 0: c_Cl is set to c_BC from t = 0, and the other
    # ions are initialised electroneutrally with that value of c_BC.
    # Na⁺ + K⁺ − OH⁻ = c_BC → keep the initial Na⁺ and K⁺ and adjust OH⁻:
    inival[ICL, 1] = m.c_cl_BC
    inival[IOH, 1] = m.c_na_init + m.c_k_init - m.c_cl_BC  # = 150+300-523 = -73 < 0 !
    # → negative: clamp OH⁻ ≥ 0 and let the solver adjust Na⁺/K⁺
    inival[IOH, 1] = max(inival[IOH, 1], 1.0e-6)
    # The Newton solve of the first time step restores electroneutrality by
    # freely changing c_Na and c_K (no Dirichlet on those species at x=0).

    # ── Output times ──────────────────────────────────────────────────────────
    times = range(0.0, t_end; length = n_save + 1)

    # ── Time step controller ──────────────────────────────────────────────────
    ctrl = VoronoiFVM.SolverControl(;
        Δt               = 0.1,
        Δt_max           = t_end / 10,
        Δu_opt           = 0.1 * m.c_cl_BC,
        handle_exceptions = true,
        verbose          = verbose,
    )

    tsol = solve(sys; inival, times, control = ctrl)

    τ = tortuosity_OhJang(m.phi, 1.0, m)
    @info "chloride_ingress Phase 2a done" steps=length(tsol.t) τ=round(τ; sigdigits=3) D_eff_Cl=round(D_eff_ion(m.D_Cl, m); sigdigits=3) c_cl_max=round(maximum(tsol[ICL, :, end]); sigdigits=4)

    return tsol, grid, m
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_reference_2a(tsol, grid, m)

Prints the chloride profile and the potential field at t_final.
Compares c_Cl with the C++ reference solution.
"""
function compare_reference_2a(tsol, grid, ::ChlorideModel2a)
    x_m  = grid[Coordinates][1, :]
    x_dm = x_m .* 10.0

    c_cl = tsol[ICL, :, end] ./ 1000.0   # mol/dm³
    c_na = tsol[INA, :, end] ./ 1000.0
    c_k  = tsol[IK,  :, end] ./ 1000.0
    c_oh = tsol[IOH, :, end] ./ 1000.0
    psi  = tsol[IPS, :, end]              # V

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    println("\nProfile at t = $(round(tsol.t[end]/3.1536e7; digits=2)) year(s) — Phase 2a (Nernst-Planck)")
    println("=" ^ 80)
    @printf("%-8s  %-14s  %-14s  %-14s  %-8s\n",
            "x [dm]", "c_Cl [mol/dm³]", "Ref C++", "c_OH [mol/dm³]", "ψ [mV]")
    println("-" ^ 80)
    for (xr, cr) in zip(x_ref, c_ref)
        idx = argmin(abs.(x_dm .- xr))
        @printf("%-8.4f  %-14.4f  %-14.4f  %-14.4f  %-8.3f\n",
                xr, c_cl[idx], cr, c_oh[idx], psi[idx] * 1e3)
    end
    println("=" ^ 80)

    # Diagnostics
    idx_front = findlast(c_cl .> 1e-3)
    x_front   = idx_front !== nothing ? x_dm[idx_front] : NaN
    en_max    = maximum(abs.(-tsol[ICL, :, end] .+ tsol[INA, :, end] .+
                              tsol[IK, :, end]  .- tsol[IOH, :, end]))

    @printf("\nDiagnostics Phase 2a :\n")
    @printf("  c_Cl(x=0) = %.4f mol/dm³  (ref: 0.523)\n",  c_cl[1])
    @printf("  c_Na(x=0) = %.4f mol/dm³\n",                  c_na[1])
    @printf("  c_K(x=0)  = %.4f mol/dm³\n",                  c_k[1])
    @printf("  c_OH(x=0) = %.4f mol/dm³\n",                  c_oh[1])
    @printf("  ψ(x=L)    = %.3f mV        (potential at the sealed face)\n", psi[end] * 1e3)
    @printf("  |EN|_max  = %.2e mol/m³    (electroneutrality residual)\n", en_max)
    @printf("  Front Cl⁻ (c > 1e-3 mol/dm³) ≈ %.1f mm\n",   x_front * 100.0)
    @printf("  C++ reference              ≈ 75 mm\n")
end

# ── Visualisation ─────────────────────────────────────────────────────────────

"""
    plot_chloride_ingress2a(tsol, grid, m; n_curves=4, save_path=nothing)

Plots two panels:
- **Left**  : c_Cl profiles at `n_curves` times + C++ reference + final c_OH profile
- **Right** : electric potential profile ψ at t_final

Requires Plots.jl loaded in the session (`using Plots`).

# Exemple
```julia
tsol, grid, m = run_chloride_ingress2a()
using Plots
p = plot_chloride_ingress2a(tsol, grid, m)
display(p)
savefig(p, "chloricem_phase2a.png")
```
"""
function plot_chloride_ingress2a(tsol, grid, m::ChlorideModel2a;
    n_curves  = 4,
    save_path = nothing,
)
    x_dm  = grid[Coordinates][1, :] .* 10.0
    t_yr  = 3.1536e7

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    n_t  = length(tsol.t)
    idxs = unique(clamp.(round.(Int, range(2, n_t; length = n_curves)), 2, n_t))

    palette = [:steelblue, :darkorange, :crimson, :forestgreen,
               :purple, :teal, :goldenrod, :indianred]

    # ── Left panel: concentrations ────────────────────────────────────────────
    p1 = plot(;
        xlabel = "x [dm]",
        ylabel = "c [mol/dm³]",
        title  = "Phase 2a — Nernst-Planck 4 ions",
        legend = :topright,
        xlims  = (0.0, maximum(x_dm)),
        ylims  = (0.0, m.c_cl_BC / 1000.0 * 1.08),
    )

    for (k, ti) in enumerate(idxs)
        c_dm = tsol[ICL, :, ti] ./ 1000.0
        yrs  = round(tsol.t[ti] / t_yr; digits = 2)
        plot!(p1, x_dm, c_dm;
            lw    = 2,
            color = palette[mod1(k, length(palette))],
            label = "c_Cl  t = $yrs yr",
        )
    end

    # OH⁻ at the final time
    c_oh_fin = tsol[IOH, :, end] ./ 1000.0
    plot!(p1, x_dm, c_oh_fin;
        lw        = 2,
        ls        = :dash,
        color     = :gray40,
        label     = "c_OH  t = $(round(tsol.t[end]/t_yr; digits=1)) yr",
    )

    # C++ reference
    scatter!(p1, x_ref, c_ref;
        ms    = 6,
        shape = :circle,
        color = :black,
        label = "C++ reference c_Cl (t = 1 year)",
    )

    # ── Right panel: electric potential ───────────────────────────────────────
    psi_mV = tsol[IPS, :, end] .* 1e3
    p2 = plot(x_dm, psi_mV;
        xlabel = "x [dm]",
        ylabel = "ψ [mV]",
        title  = "Electric potential (t = $(round(tsol.t[end]/t_yr; digits=1)) yr)",
        lw     = 2,
        color  = :steelblue,
        legend = false,
    )
    hline!(p2, [0.0]; lw=1, ls=:dot, color=:gray60)

    fig = plot(p1, p2; layout = (1, 2), size = (1000, 430))
    save_path !== nothing && savefig(fig, save_path)
    return fig
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "chloride_ingress — Phase 2a: Nernst-Planck transport, 4 ions + electroneutrality"
    tsol, grid, m = run_chloride_ingress2a(; verbose=false)
    compare_reference_2a(tsol, grid, m)

    try
        using Plots
        p = plot_chloride_ingress2a(tsol, grid, m)
        display(p)
    catch e
        @warn "Plots.jl not available — plot skipped" exception=e
    end
end
