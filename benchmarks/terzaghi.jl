# # Terzaghi 1D Consolidation
#
# The canonical verification of a poroelastic solver: a saturated column, laterally
# confined, drained at the top, loaded instantaneously. Terzaghi's closed-form series gives
# the excess pore pressure at every depth and time, so the numerical solution can be
# checked against an exact answer rather than against another code
# [terzaghi1943](@cite).
#
# ## Problem
#
# A column of height ``H``, confined laterally (``u_x = 0`` on the sides), resting on a
# rigid impermeable base (``u_y = 0``, no flow), drained at the top (``p = 0``) where a
# constant surface load ``F`` is applied at ``t = 0``.
#
# The one-dimensional Biot equations reduce to
#
# ```math
# \sigma = M_o \varepsilon - b\,p, \qquad
# \frac{\partial}{\partial t}\left(N p + b \varepsilon\right)
#   = \frac{k}{\mu_l}\frac{\partial^2 p}{\partial z^2}
# ```
#
# with ``M_o = \lambda + 2\mu`` the oedometric modulus and ``\varepsilon = \partial u/\partial z``.
# Vertical equilibrium makes ``\sigma`` uniform and equal to ``-F`` at all times, so
# ``\varepsilon`` can be eliminated, leaving a diffusion equation for the pressure alone.
#
# ## Reference solution
#
# With ``Z = (H - y)/H`` the depth measured **from the drained surface** and
# ``T = c_v t / H^2`` the dimensionless time,
#
# ```math
# \frac{p(Z, T)}{p_0} = \sum_{m=0}^{\infty}
#   \frac{4}{(2m+1)\pi}\,
#   \sin\!\left(\frac{(2m+1)\pi Z}{2}\right)
#   \exp\!\left(-\left(\frac{(2m+1)\pi}{2}\right)^{2} T\right)
# ```
#
# The two constants follow from the equations above — the consolidation coefficient from
# eliminating ``\varepsilon``, and the initial pressure from the undrained limit
# ``N p + b\varepsilon = 0`` at ``t = 0^+``:
#
# ```math
# c_v = \frac{k/\mu_l}{N + b^2/M_o}, \qquad
# p_0 = \frac{F\,b}{M_o N + b^2}
# ```
#
# At ``T = 0`` the series is the Fourier expansion of a square wave and returns 1
# everywhere, as it must.

include("biot_common.jl")

# ## Model
#
# The material is the shared `HomogeneousBiot` of `benchmarks/biot_common.jl`; the
# geometry and the load belong to this benchmark.

const TERZAGHI_MATERIAL = HomogeneousBiot(;
    E = 1.0e7, nu = 0.2, k = 1.0e-13, mu_l = 1.0e-3, b = 1.0, N = 1.0e-10,
)

const H = 1.0        # column height [m]
const W = 0.1        # column width [m] — immaterial, the column is laterally confined
const F_LOAD = 1.0e4 # surface load [Pa]

"""Initial (undrained) excess pore pressure ``p_0 = F b / (M_o N + b^2)`` [Pa]."""
initial_pressure(m::HomogeneousBiot, F = F_LOAD) =
    F * m.b / (oedometric_modulus(m) * m.N + m.b^2)

# ## Reference series

"""
    terzaghi_pressure(Z, T; nterms = 400)

Excess pore pressure `p/p₀` at normalised depth `Z = (H-y)/H` below the drained surface
and dimensionless time `T = c_v t / H²`.

The series converges slowly for small `T` — the terms decay like `exp(-(2m+1)²π²T/4)` — so
`nterms` is generous by default; the summation cost is negligible next to the solve.
"""
function terzaghi_pressure(Z, T; nterms = 400)
    T <= 0 && return 1.0
    s = 0.0
    for m in 0:(nterms - 1)
        a = (2m + 1) * π / 2
        s += (2 / a) * sin(a * Z) * exp(-a^2 * T)
    end
    return s
end

"""Downward traction `-F_LOAD` on the loaded top facet."""
function PoroMechanics.facet_load!(fe, facet, m::HomogeneousBiot, fv_u)
    fill!(fe, 0.0)
    reinit!(fv_u, facet)
    for q in 1:getnquadpoints(fv_u)
        dΓ = getdetJdV(fv_u, q)
        traction = Vec{2}((0.0, -F_LOAD))
        for i in 1:getnbasefunctions(fv_u)
            fe[i] += (shape_value(fv_u, q, i) ⋅ traction) * dΓ
        end
    end
    return nothing
end

# ## Solving
#
# One short step to `T_start` captures the undrained response — the load produces ``p_0``
# instantaneously, and a first step that is too long would already have dissipated part of
# it. The rest is marched with a uniform ``\Delta T``. Backward Euler is first order in
# time, so it is ``\Delta T`` itself, not the number of steps, that sets the temporal
# error; it has to be small enough not to mask the spatial error being measured.

"""
    run_terzaghi(; m, nely, T_probe, T_start)

Solve the column and return `(m, y, times, pressures)` where `pressures[i]` is the
pressure profile sampled at the nodes of `y` at dimensionless time `T_probe[i]`.
"""
function run_terzaghi(;
        m = TERZAGHI_MATERIAL,
        nely = 60,
        T_probe = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        T_start = 1.0e-4,
        dT = 2.5e-4,
    )
    c_v = consolidation_coefficient(m)
    t_of_T(T) = T * H^2 / c_v

    grid = generate_grid(Quadrilateral, (1, nely), Vec(0.0, 0.0), Vec(W, H))

    ip_geo = Lagrange{RefQuadrilateral, 1}()
    ip_u = Lagrange{RefQuadrilateral, 1}()^2
    ip_p = Lagrange{RefQuadrilateral, 1}()

    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    add!(dh, :p, ip_p)
    close!(dh)

    qr = QuadratureRule{RefQuadrilateral}(2)
    qr_fac = FacetQuadratureRule{RefQuadrilateral}(2)
    cv_u = CellValues(qr, ip_u, ip_geo)
    cv_p = CellValues(qr, ip_p, ip_geo)
    fv_u = FacetValues(qr_fac, ip_u, ip_geo)

    ## Drained top, rollers on the sides, clamped base.
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:p, getfacetset(grid, "top"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> 0.0, [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, [2]))
    close!(ch)
    update!(ch, 0.0)

    n_loc = ndofs_per_cell(dh)
    K1 = allocate_matrix(dh)
    K2 = allocate_matrix(dh)
    A = allocate_matrix(dh)
    as1 = start_assemble(K1)
    as2 = start_assemble(K2)
    ke1 = zeros(n_loc, n_loc)
    ke2 = zeros(n_loc, n_loc)

    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        reinit!(cv_p, cell)
        biot_element_matrices!(ke1, ke2, m, cv_u, cv_p)
        assemble!(as1, celldofs(cell), ke1)
        assemble!(as2, celldofs(cell), ke2)
    end

    ## Surface load on the top facet
    f_ext = zeros(ndofs(dh))
    u_range = dof_range(dh, :u)
    fe_u = zeros(getnbasefunctions(fv_u))
    for facet in FacetIterator(dh, getfacetset(grid, "top"))
        PoroMechanics.facet_load!(fe_u, facet, m, fv_u)
        dofs = celldofs(facet)
        for (i, d) in enumerate(u_range)
            f_ext[dofs[d]] += fe_u[i]
        end
    end

    coords = [node.x for node in grid.nodes]
    y = [c[2] for c in coords]
    p_dof = node_dof_maps(dh, grid, :p).p

    ## One short step to T_start captures the undrained response, then uniform steps.
    ## Backward Euler is first order in time, so ΔT — not the number of steps — sets the
    ## temporal error; it must be small enough not to mask the spatial error.
    schedule = uniform_schedule(T_probe; T_start = T_start, dT = dT)

    x = zeros(ndofs(dh))
    apply!(x, ch)

    pressures = Vector{Vector{Float64}}()
    probes_left = sort(T_probe)
    t_prev = 0.0
    for T in schedule
        t = t_of_T(T)
        dt = t - t_prev
        combine!(A, K1, K2, 1.0 / dt)
        rhs = copy(f_ext)
        mul!(rhs, K2, x, 1.0 / dt, 1.0)
        apply!(A, rhs, ch)
        x = A \ rhs
        t_prev = t

        if !isempty(probes_left) && isapprox(T, probes_left[1]; rtol = 1.0e-9)
            push!(pressures, [x[p_dof[i]] for i in 1:getnnodes(grid)])
            popfirst!(probes_left)
        end
    end

    return m, y, sort(T_probe), pressures
end

model, y, T_probe, p_num = run_terzaghi()

# ## Comparison with the reference solution

using Plots

p0 = initial_pressure(model)
c_v = consolidation_coefficient(model)

@printf("Oedometric modulus M_o : %.4e Pa\n", oedometric_modulus(model))
@printf("Consolidation coeff c_v: %.4e m²/s\n", c_v)
@printf("Undrained pressure p₀  : %.4e Pa  (load F = %.1e Pa)\n", p0, F_LOAD)
println()

Z = (H .- y) ./ H

"""Relative L2 error of a numerical profile against the reference series."""
function l2_error(pnum, T)
    ref = [terzaghi_pressure(z, T) * p0 for z in Z]
    return norm(pnum .- ref) / norm(ref)
end

println("      T     |  L2 error  |  L∞ error [Pa]")
println("-"^44)
errors = Float64[]
for (T, pn) in zip(T_probe, p_num)
    ref = [terzaghi_pressure(z, T) * p0 for z in Z]
    e2 = norm(pn .- ref) / norm(ref)
    einf = maximum(abs.(pn .- ref))
    push!(errors, e2)
    @printf("  %8.4f  |  %.3e |  %10.3f\n", T, e2, einf)
end
println("-"^44)
@printf("worst relative L2 error: %.3e\n", maximum(errors))

# ### Convergence
#
# A single error figure proves little — it could hide a compensating pair of mistakes. What
# a benchmark has to show is that the error goes to zero *at the expected rate*.

function worst_error(; nely, dT)
    mm, yy, Tp, pn = run_terzaghi(; nely = nely, dT = dT)
    q0 = initial_pressure(mm)
    ZZ = (H .- yy) ./ H
    return maximum(
        let ref = [terzaghi_pressure(z, T) * q0 for z in ZZ]
            norm(pv .- ref) / norm(ref)
        end for (T, pv) in zip(Tp, pn)
    )
end

# Halving ``\Delta T`` halves the error: backward Euler is first order in time, and the
# measurement confirms it.

println("  ΔT        |  worst L2 error |  ratio")
println("-"^42)
prev = NaN
for dT in (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4)
    e = worst_error(; nely = 60, dT = dT)
    @printf("  %.2e  |    %.3e    |  %s\n", dT, e, isnan(prev) ? "—" : @sprintf("%.2f", prev / e))
    global prev = e
end

# Refining the mesh instead, at a fixed ``\Delta T``, the error stops falling once the
# spatial contribution drops below the temporal floor — which is why the default
# configuration uses 60 elements and no more.

println("\n  elements  |  worst L2 error")
println("-"^32)
for n in (15, 30, 60, 120)
    @printf("  %8d  |    %.3e\n", n, worst_error(; nely = n, dT = 2.5e-4))
end

# ### Pressure profiles
#
# Markers are the finite element solution, solid lines the Terzaghi series. The isochrones
# flatten as the pressure diffuses towards the drained surface at ``Z = 0``.

plt = plot(;
    xlabel = "p / p₀  [-]",
    ylabel = "Z = (H − y) / H  [-]",
    title = "Terzaghi consolidation — numerical vs analytical",
    yflip = true,
    legend = :bottomleft,
    size = (700, 460),
)

Zfine = range(0, 1; length = 400)
order = sortperm(Z)
palette = cgrad(:viridis, max(length(T_probe), 2); categorical = true)

for (i, (T, pn)) in enumerate(zip(T_probe, p_num))
    plot!(
        plt, [terzaghi_pressure(z, T) for z in Zfine], Zfine;
        color = palette[i], lw = 2, label = "T = $T",
    )
    plot!(
        plt, (pn ./ p0)[order], Z[order];
        color = palette[i], seriestype = :scatter, ms = 3, mswidth = 0, label = "",
    )
end
plt

# ## Notes
#
# - **Equal-order elements** — ``u`` and ``p`` both P1. This is not inf-sup stable in
#   general, but the Biot storage term ``N > 0`` regularises the pressure block. The error
#   is largest at the earliest probe time, where the pressure gradient at the drained
#   surface is steepest and the mesh resolves it least well.
# - **The first step sets the initial condition** — the column starts unloaded, and the
#   load is applied over the first step. That step must be short enough that the column is
#   still essentially undrained, so it produces ``p_0`` and not a partly dissipated
#   pressure. Everything after it is marched with a uniform ``\Delta T``.
# - **Series truncation** — the reference sum converges slowly at small ``T``; 400 terms
#   keeps the truncation error far below the discretisation error being measured.
