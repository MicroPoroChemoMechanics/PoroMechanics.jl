# # Identifying a diffusion coefficient
#
# The [Fickian diffusion](fickian_diffusion.md) example solves the *forward* problem: given
# the diffusion coefficient ``D`` and the concentration ``c_\text{in}`` imposed at the inlet,
# compute the concentration profile. A laboratory faces the opposite question. It measures a
# few concentrations at a few depths, for instance on slices ground from a core, and wants the
# ``D`` and ``c_\text{in}`` that explain them. This is the *inverse* problem.
#
# It is solved here by least squares: find the parameters ``\theta`` that minimize
#
# ```math
# \Phi(\theta) = \frac{1}{2} \sum_i \big(c_i^\text{sim}(\theta) - c_i^\text{obs}\big)^2 .
# ```
#
# Gauss–Newton and Levenberg–Marquardt iterations need the Jacobian
# ``J_{ik} = \partial c_i^\text{sim} / \partial \theta_k``, which tells how each simulated
# measurement moves when a parameter moves. Every simulated value comes out of a full
# transient finite volume solve, so this Jacobian is the derivative *of the solve*. It is
# computed by **automatic differentiation**, with `ForwardDiff`, which the next section
# explains.
#
# The page answers four questions in turn:
#
# 1. how the physical experiment becomes a discrete forward model;
# 2. how to differentiate that model and check its sensitivities;
# 3. which parameters the measurements can determine;
# 4. how to calibrate them and assess uncertainty and numerical bias.
#
# Basic derivatives and matrix algebra are enough to follow the derivations below.
# The dual-number examples introduce automatic differentiation from first principles.
# Run the whole script from the repository root with Julia 1.12 or newer:
# `julia --project=examples examples/fickian_identification/run.jl`, after preparing
# the examples environment. It prints sensitivity, calibration, and refinement
# tables and creates a plot of the fitted profiles.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using OrdinaryDiffEq
using ForwardDiff
using SpecialFunctions
using LinearAlgebra
using Random
using Printf
using Plots

# ## 1. Define the experiment and the forward equation
#
# ![Diffusion specimen, sampling depths, and observation ages](../assets/identification_geometry.svg)
#
# *Sampling locations are shown along a homogeneous specimen. The drawing describes
# the experiment; it is not a simulated concentration profile.*
#
# A saturated, homogeneous specimen is initially free of tracer. Its left face
# contacts a reservoir with constant concentration ``c_{\mathrm{in}}``. The right
# face is sealed. Water does not advect: solute moves only by diffusion, with no
# reaction, sorption, or change of porosity. The unknown ``c`` [mol/m³] is solute
# concentration per unit **pore-solution volume**. Stored solute per bulk volume is
# ``\phi c``. This distinction matters when deciding whether porosity is identifiable.
#
# | Location or quantity | Condition or value | Meaning |
# |:--|:--|:--|
# | ``x=0``, region 1 | ``c=c_{\mathrm{in}}`` for ``t>0`` | Prescribed inlet concentration |
# | ``x=L``, region 2 | ``j=0`` | Sealed end; no solute crosses it |
# | Interior, ``t=0`` | ``c=0`` | No initial tracer |
# | ``L`` | 1 m | Finite specimen length |
# | ``\phi`` | 0.30 | Fixed, uniform pore-volume fraction |
# | Reference ``D`` | ``10^{-10}`` m²/s | Diffusivity used to generate synthetic data |
# | Reference ``c_{\mathrm{in}}`` | 1 mol/m³ | Reservoir value used to generate data |
#
# Define solute flux ``j`` [mol/(m²·s)] per unit bulk cross-sectional area, positive
# toward increasing ``x``. For any slice ``[a,b]`` of constant area ``A``, conservation
# and Fick's law give
#
# ```math
# \frac{d}{dt}\int_a^b\phi c\,A\,dx=A j(a,t)-A j(b,t),\qquad
# j=-\phi D\partial_xc.
# ```
#
# Dividing by area and shrinking the slice yields
#
# ```math
# \boxed{\phi\partial_t c-\partial_x(\phi D\partial_xc)=0.}
# ```
#
# For constant ``\phi`` and ``D``, this reduces to ``\partial_t c=D\partial_{xx}c``.
# In this package's convention, the coefficient multiplying the concentration
# gradient in the bulk-area flux is ``\phi D``; a reported experimental diffusion
# coefficient must use the same convention before being compared with ``D``.
# The initially decreasing concentration gives ``j>0``: tracer enters from the left.
#
# ### From a continuous profile to twelve predicted measurements
#
# On a uniform grid with ``N`` intervals, spacing ``h=L/N``, and control-volume
# length ``\ell_i``, integrate the balance around node ``i``:
#
# ```math
# \ell_i\phi\dot c_i+j_{i+1/2}-j_{i-1/2}=0,\qquad
# j_{i+1/2}=-\phi D\frac{c_{i+1}-c_i}{h}.
# ```
#
# Inside the grid ``\ell_i=h``, giving
# ``\dot c_i=D(c_{i-1}-2c_i+c_{i+1})/h^2``; endpoint volumes have length ``h/2``.
# The inlet value is imposed, while the sealed endpoint has zero external flux.
# The `FickModel` storage callback supplies ``\phi c`` and its flux callback supplies
# ``\phi D(c_i-c_j)``. VoronoiFVM supplies geometric factors and assembles the
# spatial system. OrdinaryDiffEq advances that system in time below.
#
# A **measurement operator** then samples the computed profile. Six depths at two
# ages give a vector of 12 predictions, ordered first by age and then by depth.
# Here `simulate` selects the nearest node to each requested depth. All six depths
# coincide with nodes on the 100-, 200-, and 400-interval grids used on this page,
# so this selection introduces no location error in the reported refinement study.
# For arbitrary depths, interpolate explicitly. For concentrations averaged over
# ground slices, average the simulation over the same slices: a point value and a
# slice average are different observables.
#
# ## 2. Understand automatic differentiation
#
# There are three ways to get the derivative of a computed result with respect to a parameter.
#
# - **Finite differences** run the computation twice, at ``\theta`` and at ``\theta + h``, and
#   divide the change by ``h``. The result is an approximation, and its error depends on
#   ``h``: too large and the slope is wrong, too small and rounding errors dominate.
# - **Symbolic differentiation** writes out the formula of the derivative. That is impractical
#   for a result that comes out of a mesh, a Newton loop and a thousand time steps.
# - **Automatic differentiation** applies the rules of differentiation to every elementary
#   operation the computation performs, as it performs them. The result is the exact
#   derivative along the executed differentiable operations, up to rounding, with no
#   finite-difference step to choose. Mesh and time-integration errors still remain.
#
# `ForwardDiff` implements the *forward mode* with **dual numbers**. A dual number carries a
# value and a derivative, written ``a + a'\varepsilon`` with the rule
# ``\varepsilon^2 = 0``. Arithmetic on dual numbers applies the differentiation rules by
# itself:
#
# ```math
# (a + a'\varepsilon)(b + b'\varepsilon) = ab + (a'b + ab')\,\varepsilon,
# \qquad
# \exp(a + a'\varepsilon) = e^{a} + e^{a} a'\,\varepsilon .
# ```
#
# The first is the product rule, the second the chain rule.
#
# Take ``f(x) = x^2 e^x`` at ``x = 2``. The computation starts from its input, so the
# input must already carry a derivative for the rules to propagate. That derivative is the
# **seed**: the derivative of the input with respect to the variable chosen for
# differentiation. To differentiate with respect to ``x`` itself, the seed is
# ``dx/dx = 1``.

a = ForwardDiff.Dual(2.0, 1.0)     # x = 2, seeded with dx/dx = 1: differentiate with respect to x
fa = a^2 * exp(a)

# Each operation passes the derivative along:
#
# | quantity | value | derivative part |
# |---|---|---|
# | ``a`` | ``2`` | ``1`` (the seed) |
# | ``a^2`` | ``4`` | ``2 \cdot 2 \cdot 1 = 4`` |
# | ``e^a`` | ``e^2`` | ``e^2 \cdot 1 = e^2`` |
# | ``a^2 e^a`` | ``4e^2 \approx 29.56`` | ``4 \cdot e^2 + 4 \cdot e^2 = 8e^2 \approx 59.11`` |
#
# The computation never used a formula for ``f'``. It only applied the product rule and the
# chain rule to each operation, with numbers, as the last column shows. The formula
# ``f'(x) = (2x + x^2)\,e^x`` appears here only to **check** the result: at ``x = 2`` it gives
# ``8e^2 \approx 59.11``, the derivative part found above. For the finite volume solve below,
# no such formula exists to check against, and the derivative is compared with finite
# differences instead.
#
# Without the seed of 1, the chain would have no derivative to multiply.
#
# The seed says **with respect to what** the derivative is taken:
#
# | seed | meaning | derivative part |
# |---|---|---|
# | 1 | ``a`` is the variable, ``a = x`` | ``f'(2)`` |
# | 0 | ``a`` is a constant | ``0`` |
# | 3 | ``a = 2 + 3s``, derivative with respect to ``s`` | ``3 f'(2)`` |

for seed in (1.0, 0.0, 3.0)
    z = ForwardDiff.Dual(2.0, seed)
    @printf("seed %.0f  ->  derivative part %.2f\n", seed, ForwardDiff.partials(z^2 * exp(z))[1])
end

# With several parameters, each one gets its own seed. For ``(D, c_\text{in})``, the first
# carries the derivative parts ``(1, 0)`` and the second ``(0, 1)``. Each result then carries
# two derivative parts, which form its row of the Jacobian. `ForwardDiff.derivative` and
# `ForwardDiff.jacobian` set these seeds and read the results for you. Writing `Dual` by hand,
# as above, only serves to show the mechanism:

ForwardDiff.derivative(x -> x^2 * exp(x), 2.0)

# Nothing changes for a longer computation. Every addition, product, `exp`, linear solve and
# time step in the finite volume code passes the ``\varepsilon`` part along. At the end, the
# ``\varepsilon`` part of each simulated concentration is its derivative with respect to the
# seeded parameter. With several parameters, a dual number carries one ``\varepsilon`` part per
# parameter, and all of them travel through a single run. That is what
# `ForwardDiff.jacobian` does.
#
# Two levels of automatic differentiation are nested in this page, and they do not interfere:
#
# 1. VoronoiFVM differentiates `storage!` and `flux!` with respect to the **unknowns**, to
#    build the Jacobian of its Newton iterations.
# 2. This page differentiates the whole solve with respect to the **parameters**
#    ``D`` and ``c_\text{in}``, to build the Jacobian of the least-squares problem.
#
# ForwardDiff labels each level with its own *tag*, so the two kinds of ``\varepsilon`` are
# never mixed up.
#
# Automatic differentiation also has a *reverse mode*, provided by other packages such as
# Zygote or Enzyme.
#
# | | forward mode (`ForwardDiff`) | reverse mode |
# |---|---|---|
# | cost | about one computation per **parameter**, carried together | about one computation per **output** |
# | suited to | few parameters | many parameters and one scalar output, such as a cost function |
# | memory | state plus propagated partial derivatives | intermediates stored or recomputed |
#
# With two parameters, the forward mode is the right choice. The reverse mode would pay off
# if, say, ``D`` were identified cell by cell, with hundreds of parameters for a single cost
# ``\Phi``.

# Adaptive step selection and stopping tests can change the executed path. Therefore,
# differentiating a numerical solve does not by itself certify the sensitivity of
# the continuous PDE. We check derivatives numerically and later refine the spatial
# model; time tolerances also need checking if greater precision is required.
# All intermediate arrays must accept the parameter's numeric type. Explicitly
# converting a dual number to `Float64` would break this propagation; see the
# [ForwardDiff limitations](https://juliadiff.org/ForwardDiff.jl/v1.3/user/limitations/).
#
# ## 3. Generate and interpret the measurements
#
# Six depths are sampled at two ages, about eight months and about three years.
#
# The measurements are synthetic. They come from the analytical solution
#
# ```math
# c(x, t) = c_\text{in} \operatorname{erfc}\!\left(\frac{x}{2\sqrt{D t}}\right)
# ```
#
# with ``D = 10^{-10}`` m²/s and ``c_\text{in} = 1`` mol/m³, and **not** from the finite volume
# model that is fitted to them. Data generated by the fitted model itself would be reproduced
# exactly, which would hide the discretization error. Analytical data keep that error visible,
# and it matters in the last section.
#
# This complementary-error-function solution is exact on a **semi-infinite** domain
# ``x\ge0`` with zero initial concentration and a constant inlet. The fitted model
# uses a finite domain with a sealed end. They agree to high accuracy at the sampled
# depths only while the far boundary has negligible influence. At the latest age,
# ``\sqrt{Dt}=0.10`` m and ``L/(2\sqrt{Dt})=5``; the semi-infinite concentration at
# 1 m is only ``\operatorname{erfc}(5)c_{\mathrm{in}}\approx1.5\times10^{-12}``
# mol/m³. This supports using the analytical profile here, but it is not a general
# identity between the two boundary-value problems. A shorter specimen, later age,
# or much larger diffusivity would require revisiting the reference solution.
#
# Independent Gaussian measurement noise with standard deviation 0.005 mol/m³,
# half a percent of ``c_{\mathrm{in}}``, is added using a fixed random seed. The two
# ages are ``2\times10^7`` s (about 231 days) and ``10^8`` s (about 1,157 days).
# At the deepest, earliest point the expected concentration is smaller than the
# noise, so a noisy observation can be negative. That is possible under this
# additive measurement model even though the underlying concentration is nonnegative.

const D_TRUE = 1.0e-10    # diffusion coefficient used to generate the data [m²/s]
const C_IN_TRUE = 1.0     # inlet concentration used to generate the data [mol/m³]
const PHI = 0.30          # porosity [-]
const L = 1.0             # length of the column [m]
const SIGMA = 0.005       # standard deviation of the measurement noise [mol/m³]

const X_OBS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]    # measurement depths [m]
const T_OBS = [2.0e7, 1.0e8]                          # measurement ages [s]

analytical(D, c_in, x, t) = c_in * erfc(x / (2 * sqrt(D * t)))

## Measurements are ordered by age, then by depth, and every function below uses that order.
Random.seed!(2026)
c_exact = [analytical(D_TRUE, C_IN_TRUE, x, t) for t in T_OBS for x in X_OBS]
c_obs = c_exact .+ SIGMA .* randn(length(c_exact))
nothing #hide

# ## 4. Build a differentiable transient solve
#
# To differentiate with respect to ``D`` and ``c_\text{in}``, those two numbers are replaced by
# dual numbers. Everything computed from them must then be able to hold dual numbers as well:
#
# - the model: [`FickModel`](@ref) takes the *type* of its coefficients as a parameter, so a
#   dual ``D`` is accepted as is;
# - the unknowns: `fvm_system(...; valuetype = T)` builds a system whose values have type `T`.
#
# In the dependency setup documented here, the usual VoronoiFVM call
# `solve(sys; inival, times, control)` does **not** support this parameter-dual path. VoronoiFVM
# chooses each time step from the change ``\Delta u`` of the previous one (see
# [Getting Started](../quickstart.md)). That change is a dual number, so the next step and then
# the time itself become dual numbers too. VoronoiFVM stores the time in a `Float64` field,
# and the solve stops with `MethodError(Float64, Dual(…))`. Fixed time steps do not avoid it.
# [Differentiating a solve](../demos/solver_sensitivity.md) records the details.
#
# The way around it keeps VoronoiFVM for the space discretization and hands the time stepping
# to OrdinaryDiffEq. `ODEProblem(sys, inival, tspan)` turns the finite volume system into a
# system of ordinary differential equations. VoronoiFVM still evaluates `storage!`, `flux!` and
# `bcondition!`, and still differentiates them for the Jacobian. OrdinaryDiffEq integrates in
# time, and its step-size control is written to carry dual numbers.
#
# `Rosenbrock23` is a linearly implicit method, stable on stiff problems such as diffusion.
# `saveat = T_OBS` keeps only the solution at the measurement ages, and `reshape(sol, sys)`
# turns the result into the familiar `tsol[species, node, step]` form.

"""
    solve_profiles(D, c_in; phi = PHI, N = 100) -> (x, tsol)

Concentration profiles at the ages `T_OBS`, on a grid of `N` cells. `D`, `c_in` and `phi`
may be dual numbers.
"""
function solve_profiles(D, c_in; phi = PHI, N = 100)
    T = promote_type(typeof(D), typeof(c_in), typeof(phi))
    grid = simplexgrid(range(0, L; length = N + 1))

    model = FickModel(; phi, D, dirichlet = ((1, c_in),))
    sys = fvm_system(model, grid; valuetype = T)

    inival = unknowns(sys; inival = zero(T))
    inival[1, 1] = c_in    # consistent with the Dirichlet value, as in the forward example

    problem = ODEProblem(sys, inival, (0.0, T_OBS[end]))
    sol = OrdinaryDiffEq.solve(
        problem, Rosenbrock23();
        abstol = 1.0e-9, reltol = 1.0e-7, saveat = T_OBS,
    )
    return grid[Coordinates][1, :], reshape(sol, sys)
end

"Simulated concentrations at the measurement points, in the order of `c_obs`."
function simulate(D, c_in; phi = PHI, N = 100)
    x, tsol = solve_profiles(D, c_in; phi, N)
    nodes = [argmin(abs.(x .- xo)) for xo in X_OBS]
    steps = [findfirst(==(t), tsol.t) for t in T_OBS]
    return [tsol[1, k, n] for n in steps for k in nodes]
end
nothing #hide

# The parameters are **dimensionless and scaled** to be of order one:
# ``\theta_1=D/D_*`` and ``\theta_2=c_{\mathrm{in}}/c_*``, where
# ``D_*=10^{-10}`` m²/s and ``c_*=1`` mol/m³. The code uses concentrations numerically
# in mol/m³, so multiplying the second component by the numerical scale 1 is implicit.
# Without scaling, the two sensitivity columns carry very different numerical
# scales because of the units. A raw condition number would then be dominated by
# that choice. Scaling gives interpretable parameter directions and step sizes;
# the diagonal damping used below provides additional curvature scaling.

simulate(θ::AbstractVector; kwargs...) = simulate(θ[1] * 1.0e-10, θ[2]; kwargs...)
θ_true = [D_TRUE / 1.0e-10, C_IN_TRUE]
nothing #hide

# ## 5. Check the derivatives
#
# The Jacobian from `ForwardDiff` is compared with central differences,
# ``\big(c(\theta + h e_k) - c(\theta - h e_k)\big)/2h``. Differences cost two extra solves
# per parameter and depend on the step ``h``. `ForwardDiff` needs one solve carrying two
# partial derivatives, and has no step to choose.

J_ad = ForwardDiff.jacobian(simulate, θ_true)
J_fd = reduce(
    hcat, map(1:2) do k
        h = 1.0e-6
        e = [i == k ? h : 0.0 for i in 1:2]
        (simulate(θ_true .+ e) .- simulate(θ_true .- e)) ./ (2h)
    end
)

println("  t [s]   x [m]   ∂c/∂θ₁ ForwardDiff   ∂c/∂θ₁ differences   ∂c/∂θ₂ ForwardDiff   ∂c/∂θ₂ differences")
for (i, (t, x)) in enumerate((t, x) for t in T_OBS for x in X_OBS)
    @printf(
        "  %.0e  %.2f   %+.8e      %+.8e      %+.8e      %+.8e\n",
        t, x, J_ad[i, 1], J_fd[i, 1], J_ad[i, 2], J_fd[i, 2]
    )
end

# Read the columns together: agreement should be assessed in both absolute and
# relative terms. At 0.30 m at the first age, the analytical concentration is only
# about ``2\times10^{-6}`` mol/m³, so relative derivative errors can be large even when
# absolute errors are small. Finite-difference accuracy depends on both ``h`` and
# solver tolerances. Repeat with several ``h`` values to seek an agreement plateau;
# ever smaller differences eventually amplify numerical error.
#
# The second column also has an exact answer. The problem is linear in ``c_\text{in}``: doubling
# the inlet concentration doubles the whole profile, so
# ``\partial c / \partial c_\text{in} = c / c_\text{in}``, which at ``c_\text{in} = 1`` is the
# simulated concentration itself.

c_sim = simulate(θ_true)
@printf("max |∂c/∂c_in − c / c_in| = %.1e\n", maximum(abs.(J_ad[:, 2] .- c_sim)))

# The printed gap is an absolute sensitivity discrepancy. It is affected by time
# integration and adaptive control; it is not itself the solver relative tolerance.
# Tightening tolerances should be checked before interpreting very small differences.

# ## 6. Ask which parameters the measurements can determine
#
# A parameter is determined by the data only if changing it changes the simulated
# measurements. Locally, this is read from the Jacobian: a zero column or a linear
# combination of other columns signals a parameter direction that these measurements
# cannot distinguish to first order. Its singular values quantify this sensitivity.
# A singular value close to zero means
# a direction in scaled parameter space along which predictions barely move to
# first order. Full column rank establishes local sensitivity, not global uniqueness.
#
# The porosity is added as a third parameter to see what happens.

J_phi = ForwardDiff.jacobian(θ -> simulate(θ[1] * 1.0e-10, θ[2]; phi = θ[3] * PHI), [θ_true; 1.0])
s = svdvals(J_phi)
@printf("singular values for (D, c_in, φ): %.3e  %.3e  %.3e\n", s...)
@printf("largest |∂c/∂(φ/PHI)|: %.1e\n", maximum(abs.(J_phi[:, 3])))

# The third singular value and the porosity column should be numerically tiny.
# That column differentiates with respect to ``\theta_3=\phi/\mathrm{PHI}``, so it
# equals ``\mathrm{PHI}\,\partial c/\partial\phi``; the conclusion is unchanged. The
# porosity multiplies both the storage and the flux, ``\varphi\,\partial c/\partial t =
# \nabla\cdot(D\varphi\nabla c)``, so it cancels from the equation: **no measurement of
# concentration in the pore solution can determine it.** It would take a measurement of the
# stored amount, for instance the total tracer uptake ``\int \varphi\, c \, dx``.
#
# Two other limits do not show in this Jacobian but follow from the solution:
#
# - ``D`` and ``t`` only appear as the product ``D t``. An error on the age of a sample is
#   therefore compensated, to first order, by an opposite relative error on ``D``
#   when all ages share that scale error.
# - ``D`` and ``c_\text{in}`` are separated by the *shape* of the profile, not by its level.
#   A single measurement cannot do it: a high concentration can mean a large inlet value or a
#   fast diffusion. Several suitably chosen depths or ages are needed. Points only
#   at the inlet constrain its concentration but carry no information about ``D``.
#
# With ``\varphi`` set aside, the Jacobian for ``(D, c_\text{in})`` is well conditioned:

@printf("condition number of J for (D, c_in): %.1f\n", cond(J_ad))

# ## 7. Derive and run the calibration step
#
# ![Forward solve, residuals, Jacobian, and parameter update](../assets/identification_workflow.svg)
#
# *The loop reuses the same physical model at each trial parameter pair. Its
# sensitivities also support the later identifiability and uncertainty checks.*
#
# Set ``r_i(\theta)=c_i^{\mathrm{sim}}(\theta)-c_i^{\mathrm{obs}}``. Equal independent
# noise variance makes ordinary least squares appropriate. With different known
# standard deviations, use ``r_i/\sigma_i`` and scale the Jacobian rows likewise;
# correlated errors require a corresponding covariance weighting.
#
# Linearize around the current estimate:
# ``r(\theta+\delta)\approx r(\theta)+J\delta``. Minimizing the squared length of
# this approximation gives the **Gauss–Newton normal equations**
#
# ```math
# J^{\mathsf T}J\delta=-J^{\mathsf T}r.
# ```
#
# Levenberg–Marquardt adds damping to control the trial step. The implementation
# below uses diagonal scaling, specifically
#
# ```math
# \left[H+\lambda\operatorname{diag}(H_{11},H_{22})\right]\delta=-g,
# \qquad H=J^{\mathsf T}J,\quad g=J^{\mathsf T}r.
# ```
#
# This is not ``H+\lambda I``. For large ``\lambda`` the step approaches
# ``-\operatorname{diag}(H)^{-1}g/\lambda``, a scaled negative-gradient direction.
# Small damping approaches Gauss–Newton. The code accepts a trial only if both
# parameters remain positive and the actual residual sum decreases. Success divides
# damping by three; failure multiplies it by five before another trial. Thus the
# small-step argument does not guarantee that an arbitrary trial will succeed.
#
# The printed history stores ``\sum r_i^2=2\Phi``; the factor two changes neither
# the minimizer nor the step. The starting diffusivity is three times the true value
# and the starting inlet concentration is 40% too low.
#
# This compact solver is suitable for the two sensitive parameters shown here.
# A zero-sensitivity parameter such as porosity would leave a zero diagonal entry;
# this damping cannot repair that lack of information. The solver stops after a
# small cost change, a failed set of trials, or the iteration limit, without a
# separate convergence-status object. Inspect the cost history and residuals before
# treating its returned parameters as a successful calibration. A related material
# example is in [Parameter identification](../demos/parameter_identification.md).

function levenberg_marquardt(f, θ; λ = 1.0e-3, maxiter = 40)
    r = f(θ)
    cost = sum(abs2, r)
    history = [cost]
    for _ in 1:maxiter
        J = ForwardDiff.jacobian(f, θ)
        H = J' * J
        g = J' * r
        improved = false
        for _ in 1:30
            θ_new = θ .- (H + λ * Diagonal(diag(H))) \ g
            if all(>(0), θ_new)
                r_new = f(θ_new)
                cost_new = sum(abs2, r_new)
                if cost_new < cost
                    θ, r, cost = θ_new, r_new, cost_new
                    λ = max(λ / 3, 1.0e-12)
                    improved = true
                    break
                end
            end
            λ *= 5
        end
        push!(history, cost)
        improved || break
        abs(history[end - 1] - history[end]) < 1.0e-12 * history[end] && break
    end
    return θ, history, r
end

residual(θ) = simulate(θ) .- c_obs
θ_start = [3.0, 0.6]      # D three times too large, c_in 40 % too small
θ_fit, history, r_fit = levenberg_marquardt(residual, θ_start)

println("iteration   sum of squared residuals")
for (k, c) in enumerate(history)
    @printf("  %2d        %.6e\n", k - 1, c)
end

# ## 8. Interpret parameter uncertainty and the fitted profiles
#
# Near a well-determined optimum, linearize the predictions using the fitted
# Jacobian. With independent, zero-mean, equal-variance noise and negligible model
# error, the local covariance estimate of the **scaled** parameters is:
#
# ```math
# \Sigma = \hat\sigma^2 \left(J^\top J\right)^{-1},
# \qquad
# \hat\sigma^2 = \frac{\sum_i r_i^2}{n - p},
# ```
#
# with ``n`` measurements and ``p`` parameters. The square roots of its diagonal are the
# standard errors, and ``\Sigma_{12}/\sqrt{\Sigma_{11}\Sigma_{22}}`` is the correlation
# between the two estimates. Here ``n=12``, ``p=2``, and ``n-p=10`` residual degrees
# of freedom. To recover physical parameter covariance, use
# ``\Sigma_{\mathrm{physical}}=B\Sigma B^{\mathsf T}`` with
# ``B=\operatorname{diag}(D_*,c_*)``. The printed standard error for ``D`` therefore
# multiplies the scaled value by ``10^{-10}`` m²/s.
#
# A standard error is a local estimate of sampling variability, not a guaranteed
# error bound or automatically a 95% confidence interval. Model mismatch, boundary
# uncertainty, and discretization bias are not included. Poor rank or severe
# nonlinearity can make this approximation unreliable.

J_fit = ForwardDiff.jacobian(simulate, θ_fit)
σ²_hat = sum(abs2, r_fit) / (length(c_obs) - length(θ_fit))
Σ = σ²_hat * inv(Symmetric(J_fit' * J_fit))
standard_error = sqrt.(diag(Σ))
correlation = Σ[1, 2] / (standard_error[1] * standard_error[2])

@printf("estimated noise: %.4f mol/m³ (true value %.4f)\n\n", sqrt(σ²_hat), SIGMA)
println("parameter        true          start         identified     standard error")
@printf("D [m²/s]         %.4e    %.4e    %.4e     ± %.1e\n", D_TRUE, θ_start[1] * 1.0e-10, θ_fit[1] * 1.0e-10, standard_error[1] * 1.0e-10)
@printf("c_in [mol/m³]    %.4f        %.4f        %.4f         ± %.1e\n", C_IN_TRUE, θ_start[2], θ_fit[2], standard_error[2])
@printf("\ncorrelation between D and c_in: %+.2f\n", correlation)

# Starting three times too high, ``D`` is recovered to 1.6 %, and ``c_\text{in}`` to 0.1 %.
# The estimated noise matches the one that was added. Both identified values lie within
# 1.3 standard errors of the values used to generate the data, as expected when the
# residuals are noise. These figures describe the seeded run and can shift with
# dependency versions. The uncertainty calculation requires one additional Jacobian
# evaluation at the fitted parameters, performed explicitly above.
#
# The correlation is negative: a slightly larger ``D`` spreads the profile, and a slightly
# smaller ``c_\text{in}`` brings its level back down, so the two errors partly compensate.
# The plot shows the starting profiles as dotted curves and fitted profiles as solid
# curves. The vertical bars are ``\pm2\,\mathrm{SIGMA}`` measurement-noise bars, not
# confidence bands for the fitted model or its parameters.

x, tsol_fit = solve_profiles(θ_fit[1] * 1.0e-10, θ_fit[2])
_, tsol_start = solve_profiles(θ_start[1] * 1.0e-10, θ_start[2])

p = plot(;
    xlabel = "depth x [m]", ylabel = "concentration c [mol/m³]",
    xlims = (0, 0.4), legend = :topright, size = (720, 440),
)
for (n, (t, color)) in enumerate(zip(T_OBS, (:steelblue, :darkorange)))
    label_t = @sprintf("t = %.0e s", t)
    plot!(p, x, tsol_start[1, :, n]; ls = :dot, color, label = "start, " * label_t)
    plot!(p, x, tsol_fit[1, :, n]; lw = 2, color, label = "identified, " * label_t)
    i = (n - 1) * length(X_OBS) .+ eachindex(X_OBS)
    scatter!(p, X_OBS, c_obs[i]; yerror = 2 * SIGMA, color, ms = 5, label = "measurements, " * label_t)
end
p

# ## 9. Separate numerical bias from measurement noise
#
# The standard errors above account for measurement noise, and for nothing else. The fitted
# model also carries a discretization error, which biases the identified values whatever
# the quality of the data.
#
# To isolate it, the model is fitted to the **exact** analytical values, without noise, on
# three grids. The remaining parameter difference combines spatial and temporal
# errors, optimizer termination error, and the finite-domain approximation to the
# semi-infinite reference. Here the far-boundary effect is negligible at the sampled
# points; the refinement trend tests whether spatial error dominates the others.
# Gauss–Newton starts from the calibrated values.

function fit_exact(N; θ = copy(θ_fit))
    f = θ -> simulate(θ; N) .- c_exact
    for _ in 1:10
        δ = -(ForwardDiff.jacobian(f, θ) \ f(θ))
        θ = θ .+ δ
        norm(δ) < 1.0e-10 && break
    end
    return θ
end

println("  N cells   h [mm]   relative bias on D   relative bias on c_in")
bias = map((100, 200, 400)) do N
    θ = fit_exact(N)
    @printf("  %4d      %4.1f      %+.3e           %+.3e\n", N, 1.0e3 * L / N, θ[1] - 1, θ[2] - 1)
    θ[1] - 1
end
@printf("\nrelative standard error on D from the noisy calibration: %.1e\n", standard_error[1] / θ_fit[1])

# The bias is divided by about four each time the grid is refined by two: the scheme is
# second order in ``h`` in this regime, and the parameter bias follows that trend.
# This interpretation depends on keeping observation points aligned and time and
# optimization errors small; it need not persist under unlimited mesh refinement.
#
# On the default grid, the bias on ``D`` is small compared with the standard error caused by
# the noise, so the grid is fine enough *for these measurements*. More precise measurements
# would shrink the standard error but not the bias, and the grid would then have to be refined
# until the bias is small compared with the standard error. A standard error smaller than the
# discretization bias gives a false sense of precision.
#
# ## 10. Checks and short exercises
#
# - **Units and storage:** at ``\phi=0.30`` and ``c=1`` mol/m³, one cubic meter of
#   material stores 0.30 mol. Explain why measuring that amount can reveal porosity
#   even when measuring pore-solution concentration alone cannot in this model.
# - **Parameter scaling:** a fitted ``\theta_1=1.02`` means
#   ``D=1.02\times10^{-10}`` m²/s. A scaled standard error of 0.01 means
#   ``10^{-12}`` m²/s. Verify the corresponding covariance transformation.
# - **Identifiability:** retain only the first measurement row of `J_ad`. Its rank
#   is at most one, so it cannot locally determine two parameters. Compare with
#   rows spanning several depths and both ages; consider their sensitivity relative
#   to noise, not just whether a singular value is mathematically nonzero.
# - **Derivative checks:** vary the central-difference step over several decades,
#   and tighten `abstol` and `reltol` in `solve_profiles`. Compare absolute errors at
#   low-concentration points and repeat the linearity check for ``c_{\mathrm{in}}``.
# - **Numerical bias:** compare the 100-, 200-, and 400-interval fits at the same
#   depths and ages. Halving the mesh spacing should reduce the leading spatial
#   bias by about four until other errors dominate. A small least-squares residual
#   by itself does not establish accuracy of the identified diffusivity.
# - **Experimental assumptions:** what changes if the inlet varies with time, the
#   solute binds to the solid, or measurements are slice averages? Adjust the forward
#   model and measurement operator before interpreting the fitted ``D`` physically.
#
# The conceptual illustrations can be regenerated with
# `python3 examples/fickian_identification/draw_schematics.py`.
