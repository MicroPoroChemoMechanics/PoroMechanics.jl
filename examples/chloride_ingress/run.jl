# chloride_ingress example — Phase 1: 1D chloride transport + Langmuir adsorption
#
# Validation: chloride profile against a published C++ reference solution.
# Phase 1: saturated medium (s_l = 1), single species Cl⁻, no dissolution of the
# solids (portlandite and C-S-H held constant).
#
# Equation solved:
#   ∂/∂t [φ · c + n_ads(c)] = ∇·(D_eff(φ) · ∇c)     over [0, L]
#
# Storage term — total Cl⁻ per unit volume of material:
#   N_cl = φ · c + n_csh · α · c / (1 + β · c)   [mol/m³]
#   ↑ Cl⁻ in solution    ↑ Langmuir adsorption (curve Adscl)
#
# Fick flux (effective diffusion, Oh-Jang 2004):
#   D_eff = D_Cl · τ(φ)
#   τ = τ_paste(φ) · τ_agg
#   τ_paste as defined by Oh & Jang (2004)
#
# Boundary conditions:
#   c(0, t) = c_BC  [Dirichlet] — external solution (~ seawater)
#   ∂c/∂x(L, t) = 0 [zero Neumann]  — sealed face
#
# Initial condition:
#   c(x, 0) = c_init  (chloride-free concrete)
#
# Parameters tuned on the reference case:
#   φ₀ = 0.121 · n_csh = 635 mol/m³ · α = 3.192 · β = 26.6 dm³/mol
#   D_Cl (free water, 25 °C) = 2.032e-9 m²/s
#
# Validation: compare the profile c(x, t=1 year) with the reference solution
#   Reference result: front at ~7-8 mm, c(0) ≈ 0.523 mol/dm³ = 523 mol/m³
#
# Usage :
#   julia --project examples/chloride_ingress/run.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve
using Printf

# ── Physical parameters ───────────────────────────────────────────────────────

"""
Parameters of the chloride_ingress model — Phase 1.

All quantities are SI (m, s, mol, mol/m³).
Values come from the reference case.
"""
Base.@kwdef struct ChlorideModel <: AbstractPoroModel
    # ── Geometry ──────────────────────────────────────────────────────────────
    L::Float64       = 0.05          # domain length [m]  (= 0.5 dm)

    # ── Porosity and initial state of the solids ──────────────────────────────
    phi::Float64     = 0.121         # initial porosity [-]
    n_csh::Float64   = 635.0         # initial C-S-H content [mol/m³] (= 0.635 mol/dm³)

    # ── Langmuir adsorption (curve Adscl — constant for c/c_OH between 0.5 and 1.5) ──
    alpha::Float64   = 3.192         # Langmuir coefficient α [-]   (mol_Cl / mol_CSH / (mol_Cl/dm³))
    beta::Float64    = 0.0266        # Langmuir coefficient β [m³/mol]  (= 26.6 dm³/mol)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    D_Cl::Float64    = 2.032e-9      # Cl⁻ diffusion in free solution [m²/s]

    # Parameters of the Oh-Jang (2004) tortuosity model
    phi_c::Float64   = 0.18          # critical porosity [-]
    n_OJ::Float64    = 2.7           # exponent (OPC)
    ds_OJ::Float64   = 2.0e-4        # ratio D_gel/D_water (OPC)
    tau_agg::Float64 = 0.27          # aggregate factor (= 1.0 for pure paste)

    # ── Boundary and initial conditions ───────────────────────────────────────
    c_BC::Float64    = 523.0         # [mol/m³]  Cl⁻ imposed at x = 0  (≈ seawater, 0.523 mol/dm³)
    c_init::Float64  = 1.0e-2        # [mol/m³]  initial concentration (fresh concrete, ≈ 0)
end

PoroMechanics.nspecies(::ChlorideModel) = 1
PoroMechanics.species_names(::ChlorideModel) = [:c_cl]

# ── Oh-Jang (2004) tortuosity ─────────────────────────────────────────────────

"""
    tortuosity_OhJang(phi, s_l, m)

Effective liquid tortuosity after Oh & Jang (2004),
after Oh & Jang (2004).

  τ = τ_paste(φ) · τ_agg · s_l^4.5

For the saturated medium (s_l = 1): τ = τ_paste · τ_agg.
"""
"""
    tortuosity_OhJang(phi, s_l, m::ChlorideModel)

Oh-Jang tortuosity of the cement paste, from the package constitutive layer.
"""
function tortuosity_OhJang(phi, s_l, m::ChlorideModel)
    oj = OhJang(; phi_c = m.phi_c, n = m.n_OJ, ds = m.ds_OJ, tau_agg = m.tau_agg)
    return tortuosity(oj, phi, s_l)
end

"""
    D_eff(phi, m)

Effective Cl⁻ diffusion coefficient [m²/s] for a saturation s_l = 1.

    D_eff = D_Cl · τ(φ, 1)
"""
D_eff(phi::T, m::ChlorideModel) where {T<:Real} =
    T(m.D_Cl) * tortuosity_OhJang(phi, one(T), m)

# ── Langmuir adsorption ───────────────────────────────────────────────────────

"""
    n_ads(c, m)

Amount of Cl⁻ adsorbed per unit volume [mol/m³] through the Langmuir isotherm
(curve `Adscl` of chloride_ingress):

    n_ads = n_csh · α · c / (c_ref · (1 + β · c))

with `c_ref = 1000 mol/m³ = 1 mol/dm³` — the implicit normalisation of the "1" in
the Langmuir denominator, which is 1 mol/dm³ in the original units.

Check: at c = 523 mol/m³, n_csh = 635, α = 3.192, β = 0.0266 m³/mol
→ n_ads = 635 × 3.192 × 523 / (1000 × 14.91) ≈ 71.1 mol/m³ = 0.0711 mol/dm³  ✓
"""
n_ads(c::T, m::ChlorideModel) where {T<:Real} =
    T(m.n_csh) * T(m.alpha) * c / (T(1000.0) * (1 + T(m.beta) * c))

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

"""
Storage term: N_cl = φ · c + n_ads(c)

Unit: [mol/m³]
"""
function PoroMechanics.storage!(f, u, ::Any, m::ChlorideModel, ::Any)
    c    = u[1]
    f[1] = m.phi * c + n_ads(c, m)
end

"""
Effective Fick flux: J_cl = D_eff(φ) · (c₁ − c₂)

VoronoiFVM convention: f = K·(u₁ − u₂) with K = D_eff.
"""
function PoroMechanics.flux!(f, u, ::Any, m::ChlorideModel, ::Any)
    f[1] = D_eff(m.phi, m) * (u[1, 1] - u[1, 2])
end

"""
Dirichlet condition at x = 0 (region 1): c = c_BC.
The right face (region 2) stays at zero Neumann (VoronoiFVM default).
"""
function PoroMechanics.bcondition!(f, u, bnode, m::ChlorideModel, ::Any)
    boundary_dirichlet!(f, u, bnode; species=1, region=1, value=m.c_BC)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

"""
    run_chloride_ingress(; N, t_end, n_save, verbose) -> (tsol, grid, model)

Simulates chloride penetration into a saturated concrete (Phase 1).

# Arguments
- `N`       : number of elements (default 100, as in the reference case)
- `t_end`   : total duration [s]  (default 3.1536e7 ≈ 1 year)
- `n_save`  : number of saved profiles
- `verbose` : print solver diagnostics

# Returns
- `tsol`  : transient solution (VoronoiFVM.TransientSolution)
- `grid`  : 1D grid
- `model` : the ChlorideModel instance
"""
function run_chloride_ingress(;
    N       = 100,
    t_end   = 3.1536e7,   # 1 year [s]
    n_save  = 12,
    verbose = false,
)
    m = ChlorideModel()

    # ── Uniform 1D grid ───────────────────────────────────────────────────────
    grid = simplexgrid(range(0.0, m.L; length = N + 1))

    # ── Adaptateurs VoronoiFVM ────────────────────────────────────────────────
    _storage!(f, u, node, data)  = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data)     = PoroMechanics.flux!(f, u, edge, m, data)
    _bcondition!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage    = _storage!,
        flux       = _flux!,
        bcondition = _bcondition!,
        species    = [1],
    )

    # ── Condition initiale ────────────────────────────────────────────────────
    # IC/BC consistency: the Dirichlet node (x=0) is set to c_BC from t=0 so that
    # the adaptive controller does not see an infinite jump.
    inival = unknowns(sys; inival = m.c_init)
    inival[1, 1] = m.c_BC

    # ── Output times ──────────────────────────────────────────────────────────
    # Tuned on the 12 output steps of the reference case (0 to 3.1536e7 s)
    times = range(0.0, t_end; length = n_save + 1)

    # ── Time step controller ──────────────────────────────────────────────────
    # Δt_ini tuned on Dtini = 0.1 s of the reference case.
    # Δu_opt in mol/m³: optimal variation ~ 10 % of c_BC.
    ctrl = VoronoiFVM.SolverControl(;
        Δt              = 0.1,
        Δt_max          = t_end / 10,
        Δu_opt          = 0.1 * m.c_BC,
        handle_exceptions = true,
        verbose         = verbose,
    )

    tsol = solve(sys; inival, times, control = ctrl)

    τ_ref = tortuosity_OhJang(m.phi, 1.0, m)
    @info "chloride_ingress Phase 1 done" steps=length(tsol.t) τ=round(τ_ref; sigdigits=3) D_eff=round(D_eff(m.phi, m); sigdigits=3) c_max=round(maximum(tsol[1, :, end]); sigdigits=4)

    return tsol, grid, m
end

# ── Post-traitement ───────────────────────────────────────────────────────────

"""
    compare_reference(tsol, grid, m)

Prints the chloride profile at t_final and compares it with the C++ reference.

Reference solution (t = 3.1536e7 s = 1 year):
  x [dm]   c_cl_l [mol/dm³]
  0.00     0.523
  0.005    0.406
  0.010    0.283
  0.015    0.146
  0.020    0.131
  ...
  0.065    0.005
  0.075    0.0005
"""
function compare_reference(tsol, grid, ::ChlorideModel)
    x_m    = grid[Coordinates][1, :]         # coordinates [m]
    x_dm   = x_m .* 10.0                     # in [dm] for comparison
    c_SI   = tsol[1, :, end]                 # [mol/m³] at t_final
    c_dm   = c_SI ./ 1000.0                  # [mol/dm³]

    # C++ reference at t = 1 year
    x_ref  = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref  = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    println("\nChloride profile at t = $(round(tsol.t[end]/3.1536e7; digits=2)) year(s)")
    println("=" ^ 60)
    @printf("%-10s  %-18s  %-18s\n", "x [dm]", "Julia [mol/dm³]", "Ref C++ [mol/dm³]")
    println("-" ^ 60)

    for (xr, cr) in zip(x_ref, c_ref)
        # Find the closest node in the Julia grid
        idx = argmin(abs.(x_dm .- xr))
        @printf("%-10.4f  %-18.4f  %-18.4f\n", xr, c_dm[idx], cr)
    end
    println("=" ^ 60)

    # Diagnostics globaux
    # Global diagnostics
    idx_front = findlast(c_dm .> 1e-3)   # seuil = 0.001 mol/dm³
    idx_front = findlast(c_dm .> 1e-3)   # threshold = 0.001 mol/dm³

    @printf("\nDiagnostics :\n")
    @printf("  c(x=0) = %.4f mol/dm³  (ref: 0.523)\n", c0)
    @printf("  Penetration front (c > 1e-3 mol/dm³) ≈ %.1f mm\n", x_front * 100.0)
    @printf("  C++ reference: ~75 mm\n")
end

# ── Visualisation ─────────────────────────────────────────────────────────────

"""
    plot_chloride_ingress(tsol, grid, m; n_curves=4, save_path=nothing)

Plots the Cl⁻ profiles at `n_curves` times and compares them with the C++ reference.

Requires Plots.jl loaded in the session (`using Plots`).

# Exemple
```julia
tsol, grid, m = run_chloride_ingress()
using Plots
p = plot_chloride_ingress(tsol, grid, m)
display(p)
savefig(p, "chloricem_phase1.png")
```
"""
function plot_chloride_ingress(tsol, grid, m::ChlorideModel;
    n_curves  = 4,
    save_path = nothing,
)
    x_dm  = grid[Coordinates][1, :] .* 10.0
    t_yr  = 3.1536e7   # 1 year [s]

    x_ref = [0.000, 0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.065, 0.075]
    c_ref = [0.523, 0.406, 0.283, 0.146, 0.131, 0.098, 0.066, 0.037, 0.005, 0.001]

    n_t   = length(tsol.t)
    idxs  = unique(clamp.(round.(Int, range(2, n_t; length = n_curves)), 2, n_t))

    palette = [:steelblue, :darkorange, :crimson, :forestgreen,
               :purple, :teal, :goldenrod, :indianred]

    p = plot(;
        xlabel  = "x [dm]",
        ylabel  = "c_Cl [mol/dm³]",
        title   = "chloride_ingress Phase 1 — Cl⁻ penetration (Fick + Langmuir)",
        legend  = :topright,
        xlims   = (0.0, maximum(x_dm)),
        ylims   = (0.0, m.c_BC / 1000.0 * 1.08),
        size    = (700, 450),
    )

    for (k, ti) in enumerate(idxs)
        c_dm = tsol[1, :, ti] ./ 1000.0
        yrs  = round(tsol.t[ti] / t_yr; digits = 2)
        plot!(p, x_dm, c_dm;
            lw    = 2,
            color = palette[mod1(k, length(palette))],
            label = "Julia  t = $yrs yr",
        )
    end

    scatter!(p, x_ref, c_ref;
        ms     = 6,
        shape  = :circle,
        color  = :black,
        label  = "C++ reference (t = 1 year)",
    )

    save_path !== nothing && savefig(p, save_path)
    return p
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    @info "chloride_ingress — Phase 1 : Transport Cl⁻ + adsorption Langmuir"
    tsol, grid, m = run_chloride_ingress(; verbose = false)
    compare_reference(tsol, grid, m)

    try
        using Plots
        p = plot_chloride_ingress(tsol, grid, m)
        display(p)
    catch e
        @warn "Plots.jl not available — plot skipped" exception=e
    end
end
