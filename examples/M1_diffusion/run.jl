# Example M1 — 1D diffusion (Fick's law)
#
# Diffusion of a solute in a saturated porous medium (Fick's law), on VoronoiFVM.jl.
# Validation: analytical solution (complementary error function).
#
# Equation solved:
#   φ ∂c/∂t = ∇·(D φ ∇c)   over [0, L]
#
# Boundary conditions:
#   c(0, t) = c_in   (Dirichlet)
#   ∂c/∂x(L, t) = 0  (zero Neumann, the default behaviour)
#
# Initial condition:
#   c(x, 0) = 0

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearSolve

# ── Physical parameters ───────────────────────────────────────────────────────

"""Parameters of model M1 (Fick diffusion, saturated porous medium)."""
Base.@kwdef struct M1Model <: AbstractPoroModel
    φ::Float64 = 0.30     # porosity [-]
    D::Float64 = 1e-10    # effective diffusion coefficient [m²/s]
    c_in::Float64 = 1.0   # concentration imposed at x=0 [mol/m³]
end

PoroMechanics.nspecies(::M1Model) = 1
PoroMechanics.species_names(::M1Model) = [:c]

# ── Interface VoronoiFVM ──────────────────────────────────────────────────────

"""Storage term: φ ∂c/∂t"""
function PoroMechanics.storage!(f, u, node, m::M1Model, data)
    f[1] = m.φ * u[1]
end

"""Fick flux: -D φ ∇c (finite difference between neighbours)"""
function PoroMechanics.flux!(f, u, edge, m::M1Model, data)
    f[1] = m.D * m.φ * (u[1, 1] - u[1, 2])
end

"""Dirichlet boundary condition at x = 0 (region 1)"""
function PoroMechanics.bcondition!(f, u, bnode, m::M1Model, data)
    boundary_dirichlet!(f, u, bnode; species=1, region=1, value=m.c_in)
end

# ── Solve ─────────────────────────────────────────────────────────────────────

function run_M1(; L=1.0, N=100, t_end=1e8, Δt0=1e4, n_save=20)
    m = M1Model()

    # Uniform 1D grid
    grid = simplexgrid(range(0, L; length=N + 1))

    # VoronoiFVM adapters (closure over the model)
    _storage!(f, u, node, data) = PoroMechanics.storage!(f, u, node, m, data)
    _flux!(f, u, edge, data) = PoroMechanics.flux!(f, u, edge, m, data)
    _bcondition!(f, u, bnode, data) = PoroMechanics.bcondition!(f, u, bnode, m, data)

    sys = VoronoiFVM.System(
        grid;
        storage=_storage!,
        flux=_flux!,
        bcondition=_bcondition!,
        species=[1],
    )

    # Initial condition: zero concentration except at the Dirichlet node (x=0).
    # Without that consistency the time step controller sees Δu=1 at the first
    # step and shrinks Δt forever (Dirichlet is unconditional, Δu ≠ f(Δt)).
    inival = unknowns(sys; inival=0.0)
    inival[1, 1] = m.c_in

    # Output time steps
    times = range(0, t_end; length=n_save + 1)

    ctrl = VoronoiFVM.SolverControl(;
        Δt=Δt0,
        Δt_max=t_end / 10,
        Δu_opt=0.1,
        handle_exceptions=true,
        verbose=false,
    )

    tsol = solve(sys; inival, times, control=ctrl)

    @info "M1 done: $(length(tsol.t)) time steps, c_max = $(maximum(tsol[1,:,end]))"
    return tsol, grid
end

# ── Entry point ───────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    tsol, grid = run_M1()
    # Uncomment to plot (requires Plots.jl or Makie.jl):
    # using Plots
    # x = grid[Coordinates][1,:]
    # plot(x, tsol[1,:,end]; xlabel="x [m]", ylabel="c [mol/m³]",
    #      title="1D diffusion (M1) — final profile", label="c(x, tₑₙ𝒹)")
end
