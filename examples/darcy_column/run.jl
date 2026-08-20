# examples/darcy_column/run.jl
#
# Transient single-phase Darcy flow, 1D.
#
# PDE : S ∂p/∂t − ∇·(K/μ · ∇p) = 0
#
# Boundary conditions:
#   x = 0 (bottom) : Dirichlet p = 0
#   x = L (top)    : Dirichlet p = p_top (linear ramp over t_ramp)
#
# Analytical steady state: p(x) = p_top · x / L
#
# Dependencies: VoronoiFVM, ExtendableGrids (already in PoroMechanics.jl)
# Optional plotting: Plots.jl

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# 1. Model — DarcyModel <: AbstractPoroModel
# ─────────────────────────────────────────────────────────────────────────────

"""
    DarcyModel

Linear single-phase Darcy model.

Unknown : pore pressure p [Pa].
PDE     : S ∂p/∂t = ∇·(K/μ · ∇p)
"""
Base.@kwdef struct DarcyModel <: AbstractPoroModel
    K     :: Float64 = 1e-12   # intrinsic permeability [m²]
    mu    :: Float64 = 1e-3    # dynamic viscosity [Pa·s]
    S     :: Float64 = 1e-8    # storage coefficient [-/Pa]
    L     :: Float64 = 1.0     # column length [m]
    p_top :: Float64 = 1.0e5   # pressure imposed at the top [Pa]
end

PoroMechanics.nspecies(::DarcyModel)      = 1
PoroMechanics.species_names(::DarcyModel) = [:p]

# ─────────────────────────────────────────────────────────────────────────────
# 2. AbstractPoroModel interface — VoronoiFVM callbacks (5 args)
# ─────────────────────────────────────────────────────────────────────────────

"""Darcy flux: f = (K/μ) · (p₁ − p₂)."""
function PoroMechanics.flux!(f, u, ::Any, m::DarcyModel, ::Any)
    f[1] = (m.K / m.mu) * (u[1, 1] - u[1, 2])
end

"""Storage term: S · p."""
function PoroMechanics.storage!(f, u, ::Any, m::DarcyModel, ::Any)
    f[1] = m.S * u[1]
end

"""
Boundary conditions:
  Region 1 (x = 0) : Dirichlet p = 0
  Region 2 (x = L) : Dirichlet p = p_top · ramp(t)
"""
function PoroMechanics.bcondition!(f, u, bnode, m::DarcyModel, ::Any)
    t_c  = m.S * m.mu * m.L^2 / m.K          # characteristic time
    p_bc = m.p_top * min(1.0, bnode.time / t_c)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 1, value = 0.0)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 2, value = p_bc)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Simulation
# ─────────────────────────────────────────────────────────────────────────────

function run_darcy(; N = 100, verbose = false)
    m   = DarcyModel()
    t_c = m.S * m.mu * m.L^2 / m.K   # = 10 s

    # 1D grid over [0, L]
    grid = simplexgrid(range(0.0, m.L; length = N + 1))

    # VoronoiFVM closures (inject the model in 5th position)
    _flux!(f, u, edge, data)      = PoroMechanics.flux!(f, u, edge, m, data)
    _storage!(f, u, node, data)   = PoroMechanics.storage!(f, u, node, m, data)
    _bcondition!(f, u, node, data)= PoroMechanics.bcondition!(f, u, node, m, data)

    sys = VoronoiFVM.System(
        grid;
        flux       = _flux!,
        storage    = _storage!,
        bcondition = _bcondition!,
        species    = [1],
    )

    inival = unknowns(sys; inival = 0.0)

    t_end = 500.0
    dt    = t_c / 20
    ctrl  = VoronoiFVM.SolverControl(;
        Δt        = dt,
        Δt_min    = dt,
        Δt_max    = t_end / 10,
        Δu_opt    = m.p_top / 10,
        store_all = true,
        reltol    = 1e-6,
        verbose   = verbose,
    )

    tsol = solve(sys; inival, times = (0.0, t_end), control = ctrl)

    # Analytical steady state: p(x) = p_top · x / L
    xcoords  = grid[Coordinates][1, :]
    p_ref    = m.p_top .* xcoords ./ m.L
    p_final  = tsol[1, :, end]
    err_L2   = sqrt(sum((p_final .- p_ref) .^ 2) / N)
    err_Linf = maximum(abs.(p_final .- p_ref))

    @printf("\nDarcy 1D — results\n")
    @printf("  Nodes          : %d\n", N + 1)
    @printf("  t_c            : %.1f s\n", t_c)
    @printf("  Time steps     : %d\n", length(tsol.t) - 1)
    @printf("  L2 error       : %.2e Pa\n", err_L2)
    @printf("  L∞ error       : %.2e Pa\n", err_Linf)

    # Check: error below 1 % of p_top
    tol = 0.01 * m.p_top
    if err_Linf < tol
        println("  ✓ Converged to the analytical solution (err < 1 %)")
    else
        println("  ✗ WARNING: error too large ($(round(err_Linf/m.p_top*100; digits=1)) %)")
    end

    return tsol, grid, m
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    tsol, grid, m = run_darcy()

    # Optional plot (requires Plots.jl)
    # using Plots
    # xcoords = grid[Coordinates][1, :]
    # t_c = m.S * m.mu * m.L^2 / m.K
    # p = plot(; xlabel="p [Pa]", ylabel="x [m]",
    #            title="Darcy 1D — pressure profiles")
    # for frac in [0.1, 0.5, 1.0, 2.0, 5.0]
    #     t_req = frac * t_c
    #     it = argmin(abs.(tsol.t .- t_req))
    #     plot!(p, tsol[1, :, it], xcoords; label="t = $(round(t_req; sigdigits=2)) s")
    # end
    # p_ref = m.p_top .* xcoords ./ m.L
    # plot!(p, p_ref, xcoords; lw=2, ls=:dash, color=:red, label="Analytical")
    # display(p)
end
