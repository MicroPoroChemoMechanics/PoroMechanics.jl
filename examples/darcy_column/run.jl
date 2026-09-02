# # Darcy Column 1D
#
# Transient single-phase Darcy flow in a vertical saturated soil column, on
# [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl). The steady state is known in
# closed form, which makes this a convenient check on the whole chain.
#
# ## Physical problem
#
# The pore pressure ``p`` [Pa] is the only unknown:
#
# ```math
# S \frac{\partial p}{\partial t} - \nabla \cdot \left(\frac{K}{\mu} \nabla p\right) = 0
# ```
#
# | Boundary | Condition |
# |---|---|
# | ``x = 0`` (bottom) | Dirichlet ``p = 0`` |
# | ``x = L`` (top) | Dirichlet ``p = p_\text{top} \cdot r(t)`` (ramp) |
#
# The ramp ``r(t) = \min(1,\, t / t_c)`` removes the jump between the zero initial
# condition and the imposed pressure, which would otherwise force very small time steps
# at startup.
#
# ## Reference solution
#
# At steady state the profile is linear:
#
# ```math
# p(x) = p_\text{top} \cdot \frac{x}{L}
# ```
#
# ## Parameters
#
# | Symbol | Value | Unit | Description |
# |---|---|---|---|
# | ``K`` | ``10^{-12}`` | m² | Intrinsic permeability |
# | ``\mu`` | ``10^{-3}`` | Pa·s | Dynamic viscosity |
# | ``S`` | ``10^{-8}`` | Pa⁻¹ | Storage coefficient |
# | ``L`` | ``1.0`` | m | Column length |
# | ``p_\text{top}`` | ``10^5`` | Pa | Pressure imposed at the top |
#
# The characteristic diffusion time is ``t_c = S \mu L^2 / K = 10`` s.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using Printf

# ## The model
#
# [`DarcyModel`](@ref) comes from the package. Darcy's law does not change from one column
# to the next, so what this script owns is the geometry, the material data, and the two
# pressures imposed at the ends.
#
# Both are given as data — `dirichlet = ((1, 0.0), (2, ramp))`. The second one is a
# *function of time*, which is how the ramp is expressed without a method of its own:
# `PoroMechanics.dirichlet_value` calls anything that is not a number with the current time.
# The ramp is a property of this case, not of Darcy's law.

const L = 1.0          # column length [m]
const P_TOP = 1.0e5    # pressure imposed at the top [Pa]

"""Characteristic diffusion time ``t_c = S \\mu L^2 / k`` [s]."""
characteristic_time(m::DarcyModel, L) = m.storativity * m.mu_l * L^2 / m.k_int

function darcy_material(; len = L, p_top = P_TOP)
    k_int, mu_l, storativity = 1.0e-12, 1.0e-3, 1.0e-8   # [m²], [Pa·s], [Pa⁻¹]
    t_c = storativity * mu_l * len^2 / k_int
    return DarcyModel(;
        k_int, mu_l, storativity,
        dirichlet = (
            (1, 0.0),                              # bottom: p = 0
            (2, t -> p_top * min(1.0, t / t_c)),   # top: ramped to p_top over t_c
        ),
    )
end

# ## Solving

function run_darcy(; N = 100, verbose = false)
    m = darcy_material()
    t_c = characteristic_time(m, L)   # = 10 s

    ## 1D grid over [0, L]
    grid = simplexgrid(range(0.0, L; length = N + 1))

    sys = fvm_system(m, grid)

    inival = unknowns(sys; inival = 0.0)

    t_end = 500.0
    dt = t_c / 20
    ctrl = VoronoiFVM.SolverControl(;
        Δt = dt,
        Δt_min = dt,
        Δt_max = t_end / 10,
        ## Δu_opt = p_top/10: VoronoiFVM adapts Δt so that the largest pressure
        ## change per step stays below this threshold.
        Δu_opt = P_TOP / 10,
        store_all = true,
        reltol = 1e-6,
        verbose = verbose,
    )

    tsol = solve(sys; inival, times = (0.0, t_end), control = ctrl)

    return tsol, grid, m
end

tsol, grid, model = run_darcy()

# ## Results

using Plots

xcoords = grid[Coordinates][1, :]
t_c = characteristic_time(model, L)

# ### Convergence to the analytical solution

p_ref = P_TOP .* xcoords ./ L
p_final = tsol[1, :, end]

err_L2 = sqrt(sum((p_final .- p_ref) .^ 2) / length(p_final))
err_Linf = maximum(abs.(p_final .- p_ref))

@printf("Nodes       : %d\n", length(xcoords))
@printf("t_c         : %.1f s\n", t_c)
@printf("Time steps  : %d\n", length(tsol.t) - 1)
@printf("L2 error    : %.2e Pa\n", err_L2)
@printf("L∞ error    : %.2e Pa\n", err_Linf)
err_Linf < 0.01 * P_TOP ? println("✓ err < 1 %") : println("✗ err > 1 %")

# ### Pressure profiles over time

p = plot(;
    xlabel = "Pressure p [Pa]",
    ylabel = "Height x [m]",
    title = "Darcy 1D — pressure profiles",
    legend = :topleft,
    size = (700, 420),
)

for frac in [0.1, 0.5, 1.0, 2.0, 5.0]
    t_req = frac * t_c
    it = argmin(abs.(tsol.t .- t_req))
    plot!(p, tsol[1, :, it], xcoords; label = "t = $(round(t_req; sigdigits = 2)) s")
end

plot!(
    p, p_ref, xcoords;
    linewidth = 3, color = :red, linestyle = :dash, label = "Analytical (t → ∞)",
)
p

# ## Key points
#
# - **Time ramp** — avoids the IC/BC discontinuity that would otherwise force very small
#   time steps at startup.
# - **A schedule is data** — the ramp is a closure in `dirichlet`, not a branch inside
#   `bcondition!`, so the model stays the equation and the case stays the case.
# - **`store_all = true`** — keeps every internal step in `tsol`, so the profiles at
#   intermediate times can be plotted without re-running.
