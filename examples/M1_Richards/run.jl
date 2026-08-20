# examples/M1_Richards/run.jl
#
# Richards equation 1D — imbibition of a containment barrier.
# Van Genuchten / Mualem formulation on VoronoiFVM.jl.
# Validation case: horizontal imbibition of a containment barrier.
#
# PDE :  ρ_l φ ∂S_l(p_c)/∂t  +  div(W_l) = 0
#   W_l  = −K_l ∇p_l  +  K_l ρ_l g
#   K_l  = ρ_l k_int k_rl(p_c) / μ_l
#   p_c  = p_g − p_l
#
# Test case: material "bo", horizontal imbibition.
#   x = 0 (left)  : zero Neumann (impermeable)
#   x = L (right) : Dirichlet p_l = p_g (imposed saturation)
#
# Van Genuchten retention curves:
#   S_l(p_c) = (1 + (p_c/a_Sl)^n)^(−m_Sl),  n = 1/(1−m)
#   k_rl(p_c) — Mualem

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model — M1RichardsModel <: AbstractPoroModel
# ─────────────────────────────────────────────────────────────────────────────

"""
    M1RichardsModel

Richards model 1D — single-phase unsaturated flow.

Unknown: liquid pore pressure p_l [Pa].
Van Genuchten / Mualem retention curves.
"""
Base.@kwdef struct M1RichardsModel <: AbstractPoroModel
    # Material parameters (material "bo")
    phi   :: Float64 = 0.30      # porosity [-]
    rho_l :: Float64 = 1.0e3    # liquid density [kg/m³]
    k_int :: Float64 = 1.0e-20  # intrinsic permeability [m²]
    mu_l  :: Float64 = 1.0e-3   # dynamic viscosity [Pa·s]
    p_g   :: Float64 = 1.0e5    # gas pressure [Pa]
    gravite :: Float64 = 0.0    # gravity [m/s²] (horizontal → 0)

    # Van Genuchten — S_l curve
    a_Sl  :: Float64 = 1.5e6
    m_Sl  :: Float64 = 0.06

    # Van Genuchten / Mualem — k_rl curve
    a_krl :: Float64 = 3.0e6
    m_krl :: Float64 = 0.5
end

PoroMechanics.nspecies(::M1RichardsModel)      = 1
PoroMechanics.species_names(::M1RichardsModel) = [:p_l]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Constitutive laws (type-stable for ForwardDiff)
# ─────────────────────────────────────────────────────────────────────────────

"""
S_l(p_c) — Van Genuchten.
`zero(pc)` preserves the type (Float64 or Dual) for automatic differentiation.
"""
function _Sl(pc, a, m)
    pc ≤ 0 && return 1.0 + zero(pc)
    n = 1.0 / (1.0 - m)
    return (1.0 + (pc / a)^n)^(-m)
end

"""
k_rl(p_c) — Mualem / Van Genuchten, type-stable.

ForwardDiff guard: d/dSe[√Se] = 1/(2√Se) → ∞ as Se → 0.
Below the threshold k_rl ≈ 0 physically, so we return 0 directly
(zero(pc) preserves the Dual type for automatic differentiation).
"""
function _krl(pc, a, m)
    pc ≤ 0 && return 1.0 + zero(pc)
    n   = 1.0 / (1.0 - m)
    Se  = clamp((1.0 + (pc / a)^n)^(-m), 0.0, 1.0)
    Se < 1e-14 && return zero(pc)          # dry zone: k_rl = 0, gradient = 0
    arg = clamp(1.0 - Se^(1.0 / m), 0.0, 1.0)
    return sqrt(Se) * (1.0 - arg^m)^2
end

Sl(m::M1RichardsModel, pc)  = _Sl(pc,  m.a_Sl,  m.m_Sl)
krl(m::M1RichardsModel, pc) = _krl(pc, m.a_krl, m.m_krl)
Kl(m::M1RichardsModel, pc)  = m.rho_l * m.k_int / m.mu_l * krl(m, pc)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Interface AbstractPoroModel
# ─────────────────────────────────────────────────────────────────────────────

"""
Richards flux: W_l = −K_l ∇p_l + K_l ρ_l g · dx.
VoronoiFVM convention: f[1] is divided by (x₂ − x₁).
"""
function PoroMechanics.flux!(f, u, edge, m::M1RichardsModel, ::Any)
    pl1, pl2 = u[1, 1], u[1, 2]
    pc_avg   = m.p_g - (pl1 + pl2) / 2
    Kl_avg   = Kl(m, pc_avg)
    dx       = edge.coord[1, 2] - edge.coord[1, 1]
    f[1]     = Kl_avg * (pl1 - pl2) + Kl_avg * m.rho_l * m.gravite * dx
end

"""Storage term: ρ_l φ S_l(p_c)."""
function PoroMechanics.storage!(f, u, ::Any, m::M1RichardsModel, ::Any)
    f[1] = m.rho_l * m.phi * Sl(m, m.p_g - u[1])
end

"""
Boundary conditions:
  Region 1 (x = 0) : zero Neumann (impermeable — the default behaviour)
  Region 2 (x = L) : Dirichlet p_l = p_g (full saturation)
"""
function PoroMechanics.bcondition!(f, u, bnode, m::M1RichardsModel, ::Any)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 2, value = m.p_g)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_M1Richards(; L, N, t_max_ans, verbose)

- `t_max_ans` : simulated duration in years (10 for a quick test, 100 for the full run)
- `verbose`   : print Newton iterations and time steps
"""
function run_M1Richards(; L = 0.2, N = 101, t_max_ans = 10, verbose = true)
    m = M1RichardsModel()

    grid = simplexgrid(range(0.0, L; length = N))

    _flux!(f, u, edge, data)       = PoroMechanics.flux!(f, u, edge, m, data)
    _storage!(f, u, node, data)    = PoroMechanics.storage!(f, u, node, m, data)
    _bcondition!(f, u, node, data) = PoroMechanics.bcondition!(f, u, node, m, data)

    sys = VoronoiFVM.System(
        grid;
        flux       = _flux!,
        storage    = _storage!,
        bcondition = _bcondition!,
        species    = [1],
    )

    # Initial condition: p_l = −7.611930e7 Pa (dry state)
    inival = unknowns(sys)
    inival[1, :] .= -7.611930e7
    inival[1, end] = m.p_g    # pre-apply the right-hand BC

    an     = 3.1536e7   # one year in seconds
    t_max  = t_max_ans * an

    # Output times up to t_max
    all_saves = [0.0, 1an, 2an, 4an, 6an, 8an, 10an, 20an, 40an, 50an, 100an]
    tsave = filter(t -> t ≤ t_max + 1.0, all_saves)
    tsave[end] != t_max && push!(tsave, t_max)

    @printf("M1 Richards simulation — %d years (%d output times)\n",
            t_max_ans, length(tsave) - 1)
    @printf("Press Ctrl+C to interrupt if it runs too slowly.\n\n")

    ctrl = VoronoiFVM.SolverControl(;
        Δt      = 1.0e6,
        Δt_max  = an,
        Δt_min  = 1.0,
        # The total range of p_l is ~7.7e7 Pa.
        # Δu_opt = 1e5 Pa means 0.13 % variation per step → far too restrictive
        # (27 000+ steps for 10 years). 1e6 Pa ≈ 1.3 % stays accurate and cuts
        # the step count by a factor of ~10.
        Δu_opt  = 1.0e6,
        reltol  = 1.0e-4,
        abstol  = 1.0e-8,
        verbose = verbose,
    )

    tsol = solve(sys; inival, times = tsave, control = ctrl)

    # ── Post-processing ───────────────────────────────────────────────────────
    xcoords = grid[Coordinates][1, :]
    nn      = length(xcoords)

    println("\nM1 Richards 1D — barrier imbibition")
    println("Grid: $nn nodes over [0, $L] m")
    println("Time steps taken: $(length(tsol.t) - 1)\n")

    println("t [years]      | p_l[x=0] [Pa]   | p_l[mid] [Pa]   | S_l[mid] [-]")
    println("-"^72)
    for t_s in tsave
        it   = argmin(abs.(tsol.t .- t_s))
        pl0  = tsol[1, 1,    it]
        pl_m = tsol[1, nn÷2, it]
        sl_m = Sl(m, m.p_g - pl_m)
        @printf("%-14.4f | %+14.4e  | %+14.4e  | %.6f\n",
                tsol.t[it] / an, pl0, pl_m, sl_m)
    end
    println()

    # Physical check: imbibition → p_l[mid] must increase
    pl_ini = tsol[1, nn÷2, 1]
    pl_fin = tsol[1, nn÷2, end]
    sl_ini = Sl(m, m.p_g - pl_ini)
    sl_fin = Sl(m, m.p_g - pl_fin)

    println()
    @printf("Physical check (x = middle):\n")
    @printf("  p_l : %+.4e → %+.4e Pa\n", pl_ini, pl_fin)
    @printf("  S_l : %.6f → %.6f\n",       sl_ini, sl_fin)
    if pl_fin > pl_ini
        println("  ✓ p_l increases (imbibition confirmed)")
    else
        println("  ✗ WARNING: p_l does not increase")
    end

    return tsol, grid, m
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    run_M1Richards()
end
