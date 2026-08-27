# # Steady Infiltration Above a Water Table
#
# The unsaturated counterpart of the poroelasticity benchmarks. Terzaghi and Mandel verify
# the coupled mechanics; this one verifies the Richards path — the retention and relative
# permeability curves of the constitutive layer, the VoronoiFVM flux with gravity, and the
# ability to reach a steady state — against a solution that can be written down exactly.
#
# ## Why this case has a closed form
#
# Richards' equation is nonlinear because ``k_{rl}`` depends on the unknown. For Gardner's
# exponential law [gardner1958](@cite) the nonlinearity is removable: a change of variable
# turns the steady equation into a *linear* first-order ODE.
#
# Take ``z`` upward, ``p_g = 0`` so that ``p_c = -p_l``, and write the vertical Darcy flux
# (positive upward) as
#
# ```math
# q = -K_s\,k_{rl}(p_c)\left(\frac{\mathrm{d}p_l}{\mathrm{d}z} + \rho_l g\right),
# \qquad K_s = \frac{k_\text{int}}{\mu_l}
# ```
#
# At steady state ``q`` is constant through the column. With
# ``k_{rl} = \exp(-\alpha p_c) = \exp(\alpha p_l)`` and the substitution
# ``v = \exp(\alpha p_l)``,
#
# ```math
# \frac{\mathrm{d}v}{\mathrm{d}z} = \alpha v \frac{\mathrm{d}p_l}{\mathrm{d}z}
#   = \alpha v \left(-\frac{q}{K_s v} - \rho_l g\right)
#   = -\frac{\alpha q}{K_s} - \alpha \rho_l g\, v
# ```
#
# which is linear in ``v``. Integrating from a water table at ``z = 0`` where ``p_l = 0``,
# hence ``v = 1``:
#
# ```math
# \boxed{\;p_l(z) = \frac{1}{\alpha}
#   \ln\!\left[(1 + Q)\,e^{-\beta z} - Q\right]\;}
# \qquad
# \beta = \alpha \rho_l g, \quad Q = \frac{q}{K_s \rho_l g}
# ```
#
# ``Q`` is the flux scaled by the saturated gravity-driven flux; it is negative for
# downward infiltration. For upward flux (evaporation) the bracket vanishes at a finite
# height — the water table can only sustain evaporation up to that depth — which is the
# physical content of Gardner's original paper.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearAlgebra
using Printf

# ## Model

Base.@kwdef struct GardnerColumn{R, K} <: AbstractPoroModel
    k_int::Float64 = 1.0e-12     # intrinsic permeability [m²]
    mu_l::Float64 = 1.0e-3       # dynamic viscosity [Pa·s]
    rho_l::Float64 = 1.0e3       # liquid density [kg/m³]
    gravite::Float64 = -9.81     # gravity, signed, z upward [m/s²]
    phi::Float64 = 0.35          # porosity [-]
    p_g::Float64 = 0.0           # gas pressure [Pa]
    alpha::Float64 = 5.0e-4      # Gardner exponent [Pa⁻¹]
    q_top::Float64 = -2.0e-7     # imposed flux at the top, upward positive [m/s]
    L::Float64 = 2.0             # column height above the water table [m]

    retention::R = Gardner(5.0e-4)
    rel_perm::K = GardnerKrl(5.0e-4)
end

PoroMechanics.nspecies(::GardnerColumn) = 1
PoroMechanics.species_names(::GardnerColumn) = [:p_l]

"""Saturated hydraulic conductivity in pressure form, ``K_s = k_\\text{int}/\\mu_l`` [m²/(Pa·s)]."""
saturated_conductivity(m::GardnerColumn) = m.k_int / m.mu_l

Sl(m::GardnerColumn, pc) = saturation(m.retention, pc)
krl(m::GardnerColumn, pc) = relative_permeability(m.rel_perm, pc)
Kl(m::GardnerColumn, pc) = saturated_conductivity(m) * krl(m, pc)

# ## Reference solution

"""
    gardner_profile(m, z) -> p_l [Pa]

Steady liquid pressure at height `z` above the water table. Returns `NaN` where the
bracket turns negative, i.e. where the imposed upward flux cannot be sustained.
"""
function gardner_profile(m::GardnerColumn, z)
    β = m.alpha * m.rho_l * abs(m.gravite)
    Q = m.q_top / (saturated_conductivity(m) * m.rho_l * abs(m.gravite))
    arg = (1 + Q) * exp(-β * z) - Q
    return arg <= 0 ? NaN : log(arg) / m.alpha
end

# ## Constitutive behaviour
#
# The gravity term follows the sign convention of the Richards example: `gravite` is signed
# (negative for `z` upward), and the flux returned to VoronoiFVM is divided by the edge
# length internally.

function PoroMechanics.flux!(f, u, edge, m::GardnerColumn, ::Any)
    pl1, pl2 = u[1, 1], u[1, 2]
    pc_avg = m.p_g - (pl1 + pl2) / 2
    Kl_avg = Kl(m, pc_avg)
    dx = edge.coord[1, 2] - edge.coord[1, 1]
    f[1] = Kl_avg * (pl1 - pl2) + Kl_avg * m.rho_l * m.gravite * dx
end

"""Storage: volumetric water content ``\\phi S_l(p_c)``."""
function PoroMechanics.storage!(f, u, ::Any, m::GardnerColumn, ::Any)
    f[1] = m.phi * Sl(m, m.p_g - u[1])
end

"""
Boundary conditions:
  Region 1 (z = 0) : Dirichlet p_l = 0 — the water table
  Region 2 (z = L) : Neumann, the imposed infiltration flux
"""
function PoroMechanics.bcondition!(f, u, bnode, m::GardnerColumn, ::Any)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 1, value = m.p_g)
    ## VoronoiFVM's Neumann `value` is an *inflow*, while `q_top` is signed upward.
    ## Downward infiltration (q_top < 0) is therefore an inflow of −q_top.
    boundary_neumann!(f, u, bnode; species = 1, region = 2, value = -m.q_top)
end

# ## Solving
#
# The column is marched from hydrostatic equilibrium until the profile stops moving. The
# storage term is what makes that march possible; at the steady state it drops out, which
# is why the answer depends on the relative permeability alone.

"""
    run_gardner(; m, N, t_end, n_save)

Return `(m, z, p_l)`: the node heights and the steady liquid pressure profile.
"""
function run_gardner(; m = GardnerColumn(), N = 200, t_end = 4.0e7, n_save = 40)
    grid = simplexgrid(range(0.0, m.L; length = N + 1))
    sys = fvm_system(m, grid)

    ## Start from hydrostatic equilibrium, p_l = -ρ g z, which is the zero-flux profile.
    inival = unknowns(sys)
    z0 = grid[Coordinates][1, :]
    inival[1, :] .= -m.rho_l * abs(m.gravite) .* z0
    inival[1, 1] = m.p_g

    ctrl = VoronoiFVM.SolverControl(;
        Δt = 1.0,
        Δt_max = t_end / 20,
        Δu_opt = 2.0e3,
        reltol = 1.0e-9,
        abstol = 1.0e-11,
        handle_exceptions = true,
        verbose = false,
    )

    tsol = solve(sys; inival, times = range(0.0, t_end; length = n_save + 1), control = ctrl)
    return m, z0, tsol[1, :, end], tsol
end

model, z, p_num, tsol = run_gardner()

# ## Results

using Plots

p_ref = [gardner_profile(model, zi) for zi in z]

err_L2 = norm(p_num .- p_ref) / norm(p_ref)
err_Linf = maximum(abs.(p_num .- p_ref))

Q = model.q_top / (saturated_conductivity(model) * model.rho_l * abs(model.gravite))
@printf("Gardner α        : %.3e Pa⁻¹\n", model.alpha)
@printf("Scaled flux Q    : %.6f   (negative = downward infiltration)\n", Q)
@printf("p_l at the top   : %.4e Pa   (reference %.4e Pa)\n", p_num[end], p_ref[end])
@printf("relative L2 error: %.3e\n", err_L2)
@printf("L∞ error         : %.4f Pa\n", err_Linf)

# ### Steady profile

plt = plot(;
    xlabel = "p_l  [Pa]", ylabel = "z above the water table  [m]",
    title = "Gardner steady infiltration",
    legend = :bottomleft, size = (700, 440),
)
zfine = range(0, model.L; length = 400)
plot!(plt, [gardner_profile(model, zi) for zi in zfine], zfine; lw = 2, color = :black, label = "closed form")
plot!(
    plt, p_num[1:6:end], z[1:6:end];
    seriestype = :scatter, ms = 3, mswidth = 0, color = :crimson, label = "finite volumes",
)
plot!(
    plt, -model.rho_l * abs(model.gravite) .* zfine, zfine;
    ls = :dash, color = :grey, label = "hydrostatic (q = 0)",
)
plt

# ### Convergence with the mesh
#
# Each halving of the mesh size divides the error by four: the two-point flux finite volume
# scheme is second order on a uniform grid, and the measurement says so to within 1 %.

println("  nodes  |  relative L2 error |  ratio")
println("-"^42)
prev = NaN
for N in (25, 50, 100, 200, 400)
    mm, zz, pp, _ = run_gardner(; N = N)
    ref = [gardner_profile(mm, zi) for zi in zz]
    e = norm(pp .- ref) / norm(ref)
    @printf("  %5d  |     %.3e      |  %s\n", N + 1, e, isnan(prev) ? "—" : @sprintf("%.2f", prev / e))
    global prev = e
end

# ## Notes
#
# - **The steady state does not depend on the retention curve** — only on ``k_{rl}``. The
#   retention curve controls how the column *gets* there, not where it settles, which is
#   why the closed form involves ``\alpha`` of the permeability alone.
# - **Where the closed form stops existing** — for upward flux the bracket
#   ``(1+Q)e^{-\beta z} - Q`` reaches zero at a finite height: a water table can only feed
#   evaporation down to a limited depth. `gardner_profile` returns `NaN` there rather than
#   pretending.
# - **Not a fitting curve** — Gardner's exponential law is chosen here because it makes the
#   steady equation integrable, not because it describes real soils well. For those,
#   [`VanGenuchten`](@ref) is the curve the other examples use.
