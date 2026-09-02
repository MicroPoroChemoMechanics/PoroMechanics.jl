# # Richards' equation — the Bil reference case, and where the 3 % went
#
# `base/Richards-2d` is the drainage of a composite column of glass beads: 0.02 × 0.20 m,
# with a ten-times-less-permeable inclusion in the middle, drained from the base over 360 s
# and then left for another 45 minutes. It is the case
# `examples/richards_2d` already reproduces, and the first one
# the [Bil comparison harness](https://github.com/dangla/bil) was pointed at.
#
# Three things are shared with Bil rather than reconstructed, so that a disagreement has
# nowhere to hide:
#
# * **the mesh** — Bil's own `columncomposite.msh`, read through `read_gmsh_simplexgrid`,
#   so node ``i`` here is node ``i`` there and nothing is interpolated;
# * **the retention curve** — `billes`, the table Bil interpolated, read back rather than
#   refitted to a Van Genuchten;
# * **the initial state** — hydrostatic, agreeing with Bil's `t0` file to 1.4e-7, which is
#   exactly the seven significant digits Bil prints.
#
# And yet the two codes were 3.4 % apart at ``t = 600`` s. This page is about that number.
#
# ## The first answer was wrong
#
# The obvious suspect is the time step, and the obvious test is to refine ours. It does not
# work: our own answer converges cleanly — first order, the error halving as the step halves
# — and the gap to Bil settles at 2.9 % and stays there.
#
# | our ``\Delta t_{max}`` [s] | gap to Bil's shipped output | gap to our own limit |
# |---:|---:|---:|
# | 20 | 3.82e-2 | 9.63e-3 |
# | 10 | 3.35e-2 | 4.88e-3 |
# | 5  | 3.09e-2 | 2.20e-3 |
# | 2  | 2.93e-2 | 5.53e-4 |
# | 1  | 2.88e-2 | — |
#
# Our solution is converged. The 2.9 % is somewhere else.
#
# ## What the source says
#
# `Richards.cpp` computes its flux like this:
#
# ```c
# /* Transfer coefficients at the previous time */
# double k_l = val_n.Permeability_liquid;
#
# T* w_l = val.MassFlow_liquid;
# T* gpl = val.GradPressure_liquid;
#
# for(int i = 0 ; i < 3 ; i++) w_l[i] = - k_l*gpl[i];
# w_l[dim-1] += k_l*rho_l*gravity;
# ```
#
# The permeability multiplying the pressure gradient is `val_n.Permeability_liquid` — the
# value from the **previous** time step. The current one is computed afterwards and stored
# for the next step. Bil lags the mobility explicitly; here it is taken at the current
# iterate, because `VoronoiFVM` differentiates the flux callback and Newton solves for it.
#
# That is the same shape of finding as [the Barcelona Basic Model
# page](bbm_bil.md): not a disagreement about the equations, a disagreement about how they
# are stepped. And it makes a testable prediction — an explicit lag costs first-order
# accuracy in time, so **Bil's answer should move towards ours as Bil's own step is
# refined**, not the other way round.
#
# It also explains the shape of the discrepancy over time, which nothing else did: zero at
# ``t = 0``, largest at ``t = 600`` s where ``k_{rl}`` changes fastest between steps — the
# `billes` curve takes it from 1 to 2.6e-8 across 500 Pa of suction — and decaying to 0.4 %
# by ``t = 3000`` s once the column has stopped moving.
#
# ## The decisive experiment
#
# `run_bil` can rewrite a deck before running it, so Bil's `Dtmax` can be forced down and
# the answer watched. Measured at ``t = 600`` s, against our converged solution:
#
# | Bil `Dtmax` [s] | Bil's runtime | gap to our converged answer |
# |---:|---:|---:|
# | 1000 (the deck) | 6.6 s | 2.83e-2 |
# | 100 | 6.5 s | 2.83e-2 |
# | 20 | 8.4 s | 1.34e-2 |
# | 5 | 22.9 s | 2.98e-3 |
# | 2 | 53.0 s | 1.28e-3 |
# | 1 | 102.2 s | 7.16e-4 |
#
# Bil converges towards us, at first order, over a factor of forty in the gap. The first two
# rows are identical because the deck's output dates are 200 s apart and Bil's own step
# controller was already choosing steps below 100 s; only below about 20 s does forcing
# `Dtmax` change anything.
#
# **The 3.4 % was Bil's time discretisation.** Our answer is the one Bil is converging
# towards, and the budget closes exactly:
#
# ```math
# \underbrace{3.35\times10^{-2}}_{\text{ours}(\Delta t = 10)\ \text{vs shipped Bil}}
# \;=\;
# \underbrace{2.77\times10^{-2}}_{\text{Bil's own time error}}
# \;+\;
# \underbrace{5.6\times10^{-3}}_{\text{ours, at the step the test uses}}
# ```
#
# ## What is left, and what it is worth
#
# At `Dtmax = 1` s the two codes are 7.2e-4 apart, and that residual is still falling with
# the step — there is no plateau yet. Whatever separates a Galerkin finite element
# discretisation (Bil evaluates the gradient and ``k_{rl}`` at quadrature points) from a
# two-point finite volume one on the Voronoi dual (we evaluate ``k_{rl}`` at the mean
# capillary pressure of the edge) is **below 7e-4 on this mesh** — an order of magnitude
# under the time error that had been hiding it.
#
# Two independently written codes, one finite element and one finite volume, agreeing to
# better than 1e-3 on an unsaturated drainage problem with a ten-fold permeability contrast,
# is the result worth recording here.
#
# ## Consequences for the test suite
#
# The frozen reference no longer comes from the file shipped with the deck. Comparing
# against that file meant asserting that we stay 3 % away from a number now known to carry a
# 2.8 % discretisation error — a test that passes for the wrong reason and would keep passing
# through a real regression of the same size. `test/bil/cases.jl` freezes a Bil run with
# `Dtmax = 1` s instead, and the tolerance drops from 5e-2 to 1e-2, most of which is our own
# step at the ``\Delta t_{max} = 10`` s the test runs at.
#
# The general lesson, for the models still to be compared: **a reference case is not
# automatically a converged one.** The decks in `base/` were written to demonstrate a model,
# not to resolve it, and their time steps say so. Before attributing a gap to this package,
# refine Bil.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using Printf

include(joinpath(@__DIR__, "..", "test", "bil", "cases.jl"))

# ## Reproducing the two-sided study
#
# Both halves need Bil: the source tree for the mesh and the curve, and the binary to refine
# the time step. The page reports the numbers measured on 2026-08-30 with Bil 2.11 when it
# cannot run.

if bil_root() === nothing || bil_executable() === nothing
    @info "Bil not available — see the tables above for the measured values."
else
    dir = case_dir("Richards-2d")
    cellnodes = richards_2d_mesh().grid[CellNodes]

    "Relative L2 deviation of `a` from `b`."
    rel(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps())

    "Bil's pressure profile at t = 600 s, with its own `Dtmax` forced to `dt`."
    function bil_profile(dt)
        out = run_bil("Richards-2d", "Richards-2d"; overrides = ("Dtmax" => dt,))
        return nodal_on(read_bil(joinpath(out, "Richards-2d.t3")), cellnodes, "pressure")
    end

    shipped = nodal_on(read_bil(joinpath(dir, "Richards-2d.t3")), cellnodes, "pressure")

    ## Only the Bil half is rerun here. It is the half that carries the result — that Bil
    ## moves and we do not — and it needs no solve of ours to show it.
    println("Bil, as its own time step is refined, against the shipped output")
    @printf("  %-12s %14s\n", "Dtmax [s]", "gap to shipped")
    for dt in (100.0, 20.0, 5.0)
        @printf("  %-12.1f %14.4e\n", dt, rel(bil_profile(dt), shipped))
    end
end
