# chloride_ingress — Phase 5: the balances of `CLAUDE/gia_formulation.md`, assembled
#
# `run_4.jl` is the SNIA reference: four ions, four independent Fick laws, a retardation
# coefficient frozen between chemistry passes, and electroneutrality restored after the
# fact by letting OH⁻ absorb whatever the transport unbalanced. This is the same problem
# written on conservative balances instead.
#
# | | `run_4.jl` | here |
# |---|---|---|
# | storage | `(φ + K_d) c`, `K_d` frozen | `φ c + S(c, β) + ν n` — the inventory |
# | flux | four independent Fick laws | Nernst-Planck, Scharfetter-Gummel |
# | charge | repaired by OH⁻ each pass | `Σ zᵢcᵢ = q_bg`, a constraint |
# | surface | root-find between passes, diagonal `K_d` | β a nodal unknown, full `∂S/∂c` |
# | AFm | equilibrated between passes | box complementarity, a nodal unknown |
#
# Nothing here calls `equilibrate` inside the time loop: the transport, the double layer
# and the AFm exchange are one implicit system. The rest of the assemblage — portlandite,
# ettringite, the porosity update — is **not modelled yet**, which is the honest gap
# between this and `run_4.jl` and the reason the two are not yet comparable on profiles.
#
# ## What does not converge yet, and what the diagnosis is
#
# The transport and the double layer are solid: measured on the real OPC composition they
# run at 9, 41 and 81 nodes in the same 67 steps, and neutrality holds to 1e-13.
#
# The **AFm exchange does not converge at the monosulphate content `run_4.jl` uses**
# (`n_ms0 = 3000`). The failure is not a physical limit — the released sulfate does not
# accumulate, because the coexistence branch pins `c_SO4` near `K c_Cl²` and the conversion
# is diffusion-limited; `c_SO4` peaks at 1.59 whatever `n_afm` is. It is **not monotone**
# either, in `n_afm` (100 converges, 300 does not, 1000 does, 3000 does not) or in the
# smoothing (`ε = 1e-2` converges at 300 and not at 1000, `ε = 1e-4` the reverse). A
# non-monotone failure is Newton chattering across the branch switch, which is exactly the
# risk the plan's table names.
#
# A **fixed** `ε` is therefore not the parade, and this file does not pretend otherwise.
# The parade is a continuation *within each step* — solve smooth, tighten, reuse the
# previous iterate — which needs a homotopy loop around `solve` rather than a parameter, or
# an active-set method instead of a smoothed reformulation. Neither is written.
#
# The entry point runs `n_ms0 = 100`, which converges over a full year in 350 steps. That
# is a tenth of the monosulphate `run_4.jl` carries, so the two are **not** comparable on
# profiles yet — for the AFm reason above and for the missing assemblage.
#
# Usage:
#   julia --project=examples examples/chloride_ingress/run_5.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using ChemistryLab
using DynamicQuantities
using OptimaSolver
using Printf

## `run_4.jl` is included for its OPC initialisation — `_init_chemistry4` and
## `_compute_opc_ic4` — so that both phases start from the *same* certified state and any
## difference between them is the scheme, not the initial condition. Its entry point is
## guarded, so including it defines without running.
include("run_4.jl")
include("nernst_planck.jl")
include("sorbing_transport.jl")
include("afm_exchange.jl")

## The transported set. Six ions, then the potential, then the two algebraic unknowns.
const I5 = (Cl = 1, Na = 2, K = 3, Ca = 4, SO4 = 5, OH = 6, Ψ = 7, β = 8, x_FS = 9)
const Z5 = (-1, 1, 1, 2, -2, -1)
const V_REV_5 = 1.0e-3

# ── Reading the exchange constant, rather than retyping it ────────────────────

"""
    exchange_logK(cs; T_K, P_Pa) -> Float64

`log₁₀ K` of `MS + 2Cl⁻ ⇌ FS + SO₄²⁻ + 2H₂O`, from the standard Gibbs energies the
database carries. The two AFm phases differ by their anion **and** by two waters —
monosulphate is `·6H₂O`, Friedel's salt `·4H₂O` — so the water term is not optional.

Activities of the solids are one; the aqueous ones are referred to `c_ref` in
[`AFmExchange`](@ref), which is where the units live.
"""
function exchange_logK(cs; T_K = 293.15, P_Pa = 1.0e5)
    T, P = T_K * us"K", P_Pa * us"Pa"
    g(name) = let i = findfirst(s -> string(symbol(s)) == name, cs.species)
        i === nothing && error("exchange_logK: `$name` is not in the chemical system")
        ustrip(us"J/mol", cs.species[i][:ΔₐG⁰](T = T, P = P; unit = true))
    end
    ΔG = g("C4AClH10") + g("SO4-2") + 2 * g("H2O@") - g("monosulphate12") - 2 * g("Cl-")
    RT = 8.31446261815324 * T_K
    return -ΔG / (RT * log(10))
end

# ── The case ──────────────────────────────────────────────────────────────────

"""
    build_case(; N, n_ms0, phi0, ...) -> (model, grid, inival, meta)

The OPC initial state from ChemistryLab, a NaCl boundary, and the assembled model.

Two things the reduction forces, both made explicit rather than absorbed:

  * **the background charge.** Six ions out of some thirty: the rest carry a net
    `q_bg = Σ zᵢcᵢ` which is 0.63 mol/m³ against `Σ|zᵢcᵢ| = 771` on this state. It goes
    into the constraint, not into OH⁻.
  * **consistent initial values for the algebraic unknowns.** β and `x̂` have no storage,
    so they must start on the constraint manifold or the first step has to move the whole
    surface and mineral inventory at once — the step controller then collapses to `Δt_min`
    and reports what looks like a physics failure.
"""
function build_case(;
        N = 40, L = 0.05, n_ms0 = 3000.0, n_csh0 = 635.0, x_cas = 1.5, T_K = 293.15,
        c_cl_bc = 523.0, c_na_bc = 523.0, c_k_bc = 1.0, c_ca_bc = 1.0e-3,
        c_so4_bc = 1.0e-3, c_cl_floor = 1.0e-6,
    )
    cs, has_friedels = _init_chemistry4()
    ic = _compute_opc_ic4(cs, has_friedels; n_ms0 = n_ms0, T_K = T_K)

    ## The certified OPC equilibrium returns `c_Cl ≈ 8e-301` — a denormal, because the
    ## state was seeded with 1e-16 mol of chloride and nothing produced any. `run_4.jl`
    ## never notices: its transport is linear in `c`. Here the affinity takes a logarithm
    ## of it, and the mass-action row starts at `𝒜 ≈ −10³`. Floor it at a value that is
    ## still nothing physically — 1e-6 mol/m³ is 1e-9 mol/L — and say so.
    c_ic = (
        max(ic.c_cl, c_cl_floor), ic.c_na, ic.c_k, ic.c_ca, ic.c_so4, ic.c_oh,
    )
    q_bg = sum(Z5[i] * c_ic[i] for i in 1:6)

    ## The boundary composition closes on the same background, so nothing has to be
    ## injected or destroyed at `x = 0` to keep the constraint satisfiable.
    ## `Σ zᵢcᵢ = q_bg` with `z_OH = −1` gives `c_OH = Σ_{i≤5} zᵢcᵢ − q_bg`.
    c_oh_bc = c_cl_bc * Z5[1] + c_na_bc * Z5[2] + c_k_bc * Z5[3] +
        c_ca_bc * Z5[4] + c_so4_bc * Z5[5] - q_bg
    c_oh_bc > 0 || error("the boundary composition needs a negative OH⁻: $(c_oh_bc)")
    c_bc = (c_cl_bc, c_na_bc, c_k_bc, c_ca_bc, c_so4_bc, c_oh_bc)

    dlm = DLM_TRAN2018(n_csh0 = n_csh0)
    tort = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27)
    ions = IonIndex(; Cl = I5.Cl, Na = I5.Na, K = I5.K, Ca = I5.Ca, OH = I5.OH, SO4 = I5.SO4)

    transport = NernstPlanck(;
        phi = ic.phi,
        D = (2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9, 1.065e-9, 5.273e-9),
        z = Z5, tortuosity = tort, q_background = q_bg,
        dirichlet = ntuple(i -> i <= 6 ? ((1, c_bc[i]),) : ((1, 0.0),), 7),
    )
    surface = SurfaceResolvedTransport(;
        transport = transport, dlm = dlm, ions = ions, n_csh = n_csh0, x_cas = x_cas,
    )
    n_afm = ic.n_ms + ic.n_fs
    model = AFmExchange(;
        inner = surface, ions = ions, n_afm = n_afm,
        logK = exchange_logK(cs; T_K = T_K), c_ref = 1000.0,
        eps_mcp = 1.0e-4, lambda = 0.1,
    )

    grid = simplexgrid(range(0.0, L; length = N + 1))
    inival = _initial_vector(model, grid, c_ic, c_bc, dlm, n_csh0, x_cas)
    meta = (; ic, q_bg, c_bc, n_afm, logK = model.logK, cs, has_friedels)
    return model, grid, inival, meta
end

function _initial_vector(model, grid, c_ic, c_bc, dlm, n_csh0, x_cas)
    sys = fvm_system(model, grid; reaction = true)
    u = unknowns(sys)
    for i in 1:6
        u[i, :] .= c_ic[i]
        u[i, 1] = c_bc[i]
    end
    u[I5.Ψ, :] .= 0.0
    ## β on the manifold, from the root-find that `SurfaceResolvedTransport` no longer does.
    β(c) = solve_dlm(c[1], c[2], c[3], c[4], 0.0, c[6], n_csh0, x_cas; dlm = dlm)[1]
    u[I5.β, :] .= β(c_ic)
    u[I5.β, 1] = β(c_bc)
    ## `x̂` at the branch its affinity selects: fully converted where Friedel's salt is
    ## favoured, absent where it is not.
    x̂(c) = affinity(model, vcat(collect(c), zeros(3))) > 0 ? 1.0 : 0.0
    u[I5.x_FS, :] .= x̂(c_ic)
    u[I5.x_FS, 1] = x̂(c_bc)
    return u
end

"""
    run_chloride_ingress5(; t_end, n_save, kwargs...) -> (tsol, model, grid, meta)

One implicit solve over the whole interval — no operator splitting, no chemistry pass
between segments.
"""
function run_chloride_ingress5(;
        t_end = 3.1536e7, n_save = 12, verbose = false, kwargs...
    )
    model, grid, inival, meta = build_case(; kwargs...)
    sys = fvm_system(model, grid; reaction = true)
    control = VoronoiFVM.SolverControl(;
        Δt = 1.0e2, Δt_min = 1.0e-3, Δt_max = 1.0e5,
        ## `damp_initial = 0.2` and not less. Damping harder does not help a stalled
        ## Newton, it exhausts its iteration budget: at 0.1 the residual sits at 1e-3 and
        ## oscillates for a hundred iterations, which reads as non-smoothness and is
        ## over-damping.
        Δu_opt = 50.0, damp_initial = 0.2, damp_growth = 1.2, verbose = verbose,
    )
    tsol = solve(sys; inival, times = [0.0, t_end], control)
    return tsol, model, grid, meta
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    ## `n_ms0 = 100`, not the 3000 of `run_4.jl`: see the note at the top of this file.
    tsol, model, grid, meta = run_chloride_ingress5(;
        N = 40, t_end = 3.1536e7, n_save = 12, n_ms0 = 100.0,
    )
    u = tsol.u[end]
    x = grid[Coordinates][1, :]
    @printf("\nlog₁₀K de l'échange AFm, lu dans cemdata18 : %.4f\n", meta.logK)
    @printf("charge de fond des espèces non transportées : %.4f mol/m³\n", meta.q_bg)
    @printf("AFm total : %.1f mol/m³_béton\n\n", meta.n_afm)
    @printf(
        "%-8s %-10s %-10s %-10s %-9s %-9s %-9s\n",
        "x[mm]", "c_Cl", "c_SO4", "c_OH", "Ψ", "β", "x_FS"
    )
    for k in 1:4:length(x)
        @printf(
            "%-8.2f %-10.4g %-10.4g %-10.4g %-9.4f %-9.4f %-9.5f\n",
            x[k] * 1000, u[I5.Cl, k], u[I5.SO4, k], u[I5.OH, k],
            u[I5.Ψ, k], u[I5.β, k], u[I5.x_FS, k]
        )
    end
end
