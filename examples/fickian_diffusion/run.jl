# # Fickian Diffusion 1D
#
# Diffusion of a solute in a saturated porous medium, on
# [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl). This is the simplest model in
# the package: one species, a linear equation, and a closed-form reference solution — so
# it is the case that validates the whole `AbstractPoroModel` → closures → VoronoiFVM
# chain.
#
# ## Physical problem
#
# A solute diffuses into a saturated soil column. The concentration ``c`` [mol/m³] is the
# only unknown, and it obeys
#
# ```math
# \varphi \frac{\partial c}{\partial t} = \nabla \cdot \left(D \varphi \, \nabla c\right)
# ```
#
# The porosity ``\varphi`` cancels, leaving pure diffusion:
#
# ```math
# \frac{\partial c}{\partial t} = D \, \frac{\partial^2 c}{\partial x^2}
# ```
#
# | Boundary | Condition |
# |---|---|
# | ``x = 0`` (inlet) | Dirichlet ``c = c_\text{in}`` |
# | ``x = L`` (outlet) | Zero Neumann ``\partial c / \partial x = 0`` (the VoronoiFVM default) |
#
# The initial condition is ``c(x, 0) = 0``, with ``c(0, 0) = c_\text{in}`` so that it is
# consistent with the boundary condition — see the note at the end of this page.
#
# ## Reference solution
#
# On a semi-infinite domain the solution is
#
# ```math
# c(x, t) = c_\text{in} \operatorname{erfc}\!\left(\frac{x}{2\sqrt{D t}}\right)
# ```
#
# valid as long as the diffusion front ``2\sqrt{Dt}`` stays small compared with ``L``.
#
# ## Parameters
#
# | Symbol | Value | Unit | Description |
# |---|---|---|---|
# | ``\varphi`` | ``0.30`` | — | Porosity |
# | ``D`` | ``10^{-10}`` | m²/s | Effective diffusion coefficient |
# | ``c_\text{in}`` | ``1.0`` | mol/m³ | Concentration imposed at the inlet |
# | ``L`` | ``1.0`` | m | Column length |
#
# The characteristic diffusion time is ``t_\text{diff} = L^2/D = 10^{10}`` s. The
# simulation covers ``t_\text{end} = 10^8`` s ``= t_\text{diff}/100``, an early transient
# in which the front penetrates only about ``2\sqrt{D t_\text{end}} \approx 0.2`` m.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids

# ## The model
#
# [`FickModel`](@ref) lives in the package, not in this script. Diffusion through a
# saturated medium is the same equation whatever column it is solved on; what belongs here
# is the material data, the geometry, and the concentration imposed at the inlet.
#
# That imposed concentration is given as data — `dirichlet = ((1, c_in),)`, meaning "impose
# `c_in` on boundary region 1" — rather than written into a method, so the same model serves
# a column fed from the other end without editing anything. The sealed face at ``x = L``
# needs no code at all: zero flux is what `VoronoiFVM` does with a boundary nobody claims.

const C_IN = 1.0    # concentration imposed at x = 0 [mol/m³]

fick_material(; c_in = C_IN) = FickModel(;
    phi = 0.30,                  # porosity [-]
    D = 1.0e-10,                 # effective diffusion coefficient [m²/s]
    dirichlet = ((1, c_in),),    # imposed concentration at x = 0
)

# ## Solving

function run_fickian_diffusion(; L = 1.0, N = 100, t_end = 1e8, Δt0 = 1e4, n_save = 20)
    m = fick_material()

    ## Uniform 1D grid
    grid = simplexgrid(range(0, L; length = N + 1))

    sys = fvm_system(m, grid)

    ## Initial condition: zero concentration except at the Dirichlet node (x=0).
    ## Without that consistency the time step controller sees Δu=1 at the first
    ## step and shrinks Δt forever (Dirichlet is unconditional, Δu ≠ f(Δt)).
    inival = unknowns(sys; inival = 0.0)
    inival[1, 1] = C_IN

    ## Output time steps
    times = range(0, t_end; length = n_save + 1)

    ctrl = VoronoiFVM.SolverControl(;
        Δt = Δt0,
        Δt_max = t_end / 10,
        Δu_opt = 0.1,
        handle_exceptions = true,
        verbose = false,
    )

    tsol = solve(sys; inival, times, control = ctrl)

    return tsol, grid, m
end

tsol, grid, model = run_fickian_diffusion()

# ## Results

using Plots
using Printf
using SpecialFunctions

xcoords = grid[Coordinates][1, :]
t_end = tsol.t[end]

# ### Concentration profiles over time

p = plot(;
    xlabel = "Position x [m]",
    ylabel = "Concentration c [mol/m³]",
    title = "Fickian diffusion 1D — transient profiles",
    legend = :topright,
    size = (700, 420),
)

for frac in [0.01, 0.05, 0.1, 0.5, 1.0]
    t_req = frac * t_end
    it = argmin(abs.(tsol.t .- t_req))
    plot!(p, xcoords, tsol[1, :, it]; label = "t = $(round(t_req; sigdigits = 2)) s")
end
p

# ### Comparison with the analytical solution
#
# The numerical profile is compared with the semi-infinite `erfc` solution at the final
# time.

c_num = tsol[1, :, end]
c_ref = C_IN .* erfc.(xcoords ./ (2 * sqrt(model.D * t_end)))

err_L2 = sqrt(sum((c_num .- c_ref) .^ 2) / length(c_num))
err_Linf = maximum(abs.(c_num .- c_ref))

@printf("L2 error   : %.2e mol/m³\n", err_L2)
@printf("L∞ error   : %.2e mol/m³\n", err_Linf)
err_Linf < 0.01 * C_IN ? println("✓ err < 1 %") : println("✗ err > 1 %")

# ## Key points
#
# - **IC/BC consistency** — the initial condition must satisfy the Dirichlet condition at
#   node `x=0` from `t=0` (`inival[1,1] = C_IN`). Without it the adaptive controller sees
#   `Δu = C_IN` regardless of `Δt`, and shrinks the time step until `Δt_min`.
# - **Implicit zero Neumann** — VoronoiFVM applies zero flux by default on any boundary
#   that `bcondition!` does not handle, so `x = L` needs no `boundary_neumann!` call.
# - **`Δu_opt`** — set to `0.1` mol/m³, 10 % of `C_IN`: VoronoiFVM adapts `Δt` so that the
#   largest concentration change per step stays under that threshold.
