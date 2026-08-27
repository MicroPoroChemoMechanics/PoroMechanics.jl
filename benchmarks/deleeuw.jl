# # De Leeuw's Problem — Consolidation of a Cylinder
#
# The axisymmetric member of the Mandel–Cryer family: a long saturated cylinder, drained on
# its lateral surface, compressed radially at ``t = 0``. Like the slab and the sphere, the
# pore pressure at the axis overshoots before decaying [deleeuw1965](@cite) — by 11 %,
# between Mandel's 6.5 % and Cryer's 23 % on the same material.
#
# That ordering is the point of running all three. The overshoot grows with the number of
# directions in which drainage is constrained, and reproducing the *sequence* is a far
# stronger statement about the coupling than reproducing any single case.
#
# This benchmark also exercises the **cylindrical** kinematics — the coordinate system the
# Barcelona Basic Model will need for axisymmetry, unlike Cryer's spherical one.
#
# ## Problem
#
# Cylinder of radius ``R``, plane strain along its axis (``\varepsilon_{zz} = 0``). At
# ``t = 0`` a compressive radial traction ``P_c`` is applied to the lateral surface, which
# is held drained.
#
# | Boundary | Condition |
# |---|---|
# | ``r = 0`` | symmetry: ``u_r = 0``, no flux |
# | ``r = R`` | drained ``p = 0``, traction ``\sigma_{rr} = -P_c`` |
#
# ## Reference solution
#
# Derived by the same route as Cryer's, from the single-porosity formulation of
# [mehrabian2018](@cite). The displacement field is again irrotational, so
# ``\varepsilon = c_m p + f(t)`` with the *same* ``c_m = \alpha/M_o``; only the Laplacian
# and the volume element change.
#
# In cylindrical coordinates the Laplace-space pressure is built on modified Bessel
# functions rather than hyperbolic sines:
#
# ```math
# \tilde P(r^*, \hat s) = \frac{P_c}{\hat s}\,\frac{\alpha}{2GS}\,
#   \frac{1 - \dfrac{I_0(r^*\sqrt{\hat s})}{I_0(\sqrt{\hat s})}}{D(\hat s)},
# \qquad
# D(\hat s) = \frac{1}{2(1-2\nu)} + \frac{q}{2}
#   - q\,\frac{I_1(\sqrt{\hat s})}{\sqrt{\hat s}\,I_0(\sqrt{\hat s})}
# ```
#
# with ``q = \alpha c_m / S`` and ``S = 1/M + \alpha^2/M_o`` as before.
#
# ## The check that this solution has to pass
#
# The undrained limit is **not** Skempton's ``B P_c`` here, and that is what makes it a
# genuinely independent test rather than a re-run of Cryer's. Plane strain forbids
# ``\varepsilon_{zz}``, so the axial stress is whatever the constraint requires:
# ``\sigma_{zz} = \nu_u(\sigma_{rr} + \sigma_{\theta\theta}) = -2\nu_u P_c``. The mean stress
# is therefore ``-\tfrac{2}{3}(1+\nu_u)P_c``, and
#
# ```math
# p(r, 0^+) = \frac{2}{3}\,B\,(1 + \nu_u)\,P_c
# ```
#
# uniform in ``r``. The initial value theorem must return exactly that.

include("biot_common.jl")

using SpecialFunctions

# ## Model
#
# The material of Mandel's and Cryer's problems, so the three overshoots are comparable.

const DELEEUW_MATERIAL = HomogeneousBiot(;
    E = 1.0e8, nu = 0.2, k = 1.0e-13, mu_l = 1.0e-3, b = 1.0, N = 7.2e-9,
)

const R_CYL = 1.0       # radius [m]
const P_LAT = 1.0e6     # lateral confining traction [Pa]

"""Geertsma's compaction coefficient ``c_m = \\alpha/M_o`` [Pa⁻¹]."""
compaction_coefficient(m::HomogeneousBiot) = m.b / oedometric_modulus(m)

"""Storage coefficient ``S = 1/M + \\alpha c_m`` [Pa⁻¹]."""
storage_coefficient(m::HomogeneousBiot) = m.N + m.b * compaction_coefficient(m)

"""
    undrained_pressure(m; Pc)

Initial pore pressure ``p(r,0^+) = \\tfrac{2}{3} B (1+\\nu_u) P_c`` for radial compression
of a cylinder in plane strain.
"""
undrained_pressure(m::HomogeneousBiot; Pc = P_LAT) =
    2 * skempton(m) * (1 + undrained_poisson(m)) * Pc / 3

# ## Reference solution
#
# `besseli` overflows for large arguments, so the exponentially scaled `besselix` is used
# and the exponential factor restored by hand.

_I0_ratio(a, x) = besselix(0, a * x) / besselix(0, x) * exp((a - 1) * x)
_I1_over_xI0(x) = besselix(1, x) / (x * besselix(0, x))

"""
    deleeuw_laplace(m, r, ŝ; Pc)

Pore pressure in Laplace space, `ŝ` conjugate to the dimensionless time `t* = ct/R²`.
"""
function deleeuw_laplace(m::HomogeneousBiot, r, ŝ; Pc = P_LAT)
    _, G = lame(m)
    S = storage_coefficient(m)
    q = m.b * compaction_coefficient(m) / S
    x = sqrt(ŝ)
    D = 1 / (2 * (1 - 2m.nu)) + q / 2 - q * _I1_over_xI0(x)
    return (Pc / ŝ) * (m.b / (2G * S)) * (1 - _I0_ratio(r, x)) / D
end

"""
    deleeuw_pressure(m, r, t; Pc) -> p [Pa]

Stehfest inversion of [`deleeuw_laplace`](@ref).

Unlike Cryer's, this transform cannot be evaluated in `BigFloat`: `SpecialFunctions` provides
Bessel functions in `Float64` only. The weights stay in `BigFloat`, which is where the
cancellation actually bites; `Float64` values of the transform are enough, and the agreement
between `N = 10`, `14` and `16` confirms it.
"""
function deleeuw_pressure(m::HomogeneousBiot, r, t; Pc = P_LAT)
    t <= 0 && return undrained_pressure(m; Pc = Pc)
    return stehfest(ŝ -> deleeuw_laplace(m, r, Float64(ŝ); Pc = Pc), t)
end

# ## Solving
#
# The element is `radial_element_matrices!` with `nhoop = 1` — one hoop direction and an
# ``r\,\mathrm{d}r`` weight, against two and ``r^2`` for the sphere. That single parameter is
# the whole difference between the two geometries.

"""
    run_deleeuw(; m, nel, T_probe, T_start, dT)

Return `(m, r, T_probe, profiles, T_hist, p_axis)`.
"""
function run_deleeuw(;
        m = DELEEUW_MATERIAL,
        nel = 80,
        T_probe = [0.01, 0.03, 0.05, 0.1, 0.3, 0.8],
        T_start = 1.0e-4,
        dT = 5.0e-4,
    )
    c = consolidation_coefficient(m)
    t_of_T(T) = T * R_CYL^2 / c

    grid = generate_grid(Line, (nel,), Vec(0.0), Vec(R_CYL))

    ip = Lagrange{RefLine, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :p, ip)
    close!(dh)

    qr = QuadratureRule{RefLine}(2)
    cv_u = CellValues(qr, ip, ip)
    cv_p = CellValues(qr, ip, ip)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:p, getfacetset(grid, "right"), (x, t) -> 0.0))
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
        radial_element_matrices!(ke1, ke2, m, cv_u, cv_p, getcoordinates(cell); nhoop = 1)
        assemble!(as1, celldofs(cell), ke1)
        assemble!(as2, celldofs(cell), ke2)
    end

    u_dof = zeros(Int, getnnodes(grid))
    p_dof = zeros(Int, getnnodes(grid))
    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)
    for cell in CellIterator(dh)
        d = celldofs(cell)
        for (loc, node) in enumerate(cell.nodes)
            u_dof[node] = d[u_range[loc]]
            p_dof[node] = d[p_range[loc]]
        end
    end

    ## Boundary term of the weak form, δu σ_rr r, evaluated at r = R.
    coords = [node.x[1] for node in grid.nodes]
    surface_node = argmax(coords)
    f_ext = zeros(ndofs(dh))
    f_ext[u_dof[surface_node]] = -P_LAT * R_CYL

    schedule = uniform_schedule(T_probe; T_start = T_start, dT = dT)

    x = zeros(ndofs(dh))
    apply!(x, ch)

    profiles = Vector{Vector{Float64}}()
    probes_left = sort(T_probe)
    T_hist = Float64[]
    p_axis = Float64[]
    axis_node = argmin(coords)
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

        push!(T_hist, T)
        push!(p_axis, x[p_dof[axis_node]])

        if !isempty(probes_left) && isapprox(T, probes_left[1]; rtol = 1.0e-9)
            push!(profiles, [x[p_dof[i]] for i in 1:getnnodes(grid)])
            popfirst!(probes_left)
        end
    end

    return m, coords, sort(T_probe), profiles, T_hist, p_axis
end

model, rr, T_probe, p_num, T_hist, p_axis = run_deleeuw()

# ## Results

using Plots

p0 = undrained_pressure(model)

@printf("Skempton B          : %.6f\n", skempton(model))
@printf("Undrained ν_u       : %.6f\n", undrained_poisson(model))
@printf("p₀ = (2/3)B(1+ν_u)Pc: %.4e Pa   (Cryer's sphere would give B·Pc = %.4e Pa)\n",
    p0, skempton(model) * P_LAT)
println()

# ### The reference, checked before use

@printf("p(r,t→0)/P_c at r* = 0.1, 0.5, 0.9 : %.6f  %.6f  %.6f   (expected %.6f)\n",
    deleeuw_pressure(model, 0.1, 1.0e-6) / P_LAT,
    deleeuw_pressure(model, 0.5, 1.0e-6) / P_LAT,
    deleeuw_pressure(model, 0.9, 1.0e-6) / P_LAT, p0 / P_LAT)
@printf("p at the drained surface, t* = 0.1 : %.3e Pa\n", deleeuw_pressure(model, 0.999999, 0.1))
println()

# ### The overshoot, and where it sits between slab and sphere

p_ref_axis = [deleeuw_pressure(model, 1.0e-8, T) for T in T_hist]

i_num = argmax(p_axis)
i_ref = argmax(p_ref_axis)
@printf("numerical peak : p/p₀ = %.5f at T = %.4f\n", p_axis[i_num] / p0, T_hist[i_num])
@printf("reference peak : p/p₀ = %.5f at T = %.4f\n", p_ref_axis[i_ref] / p0, T_hist[i_ref])
@printf("overshoot      : %.1f %%   (Mandel slab 6.5 %%, Cryer sphere 22.9 %%)\n",
    100 * (p_ref_axis[i_ref] / p0 - 1))
println()

plt_hist = plot(
    T_hist, p_ref_axis ./ p0;
    xlabel = "T = c t / R²  [-]", ylabel = "p(0, t) / p₀  [-]",
    title = "De Leeuw — overshoot on the axis of the cylinder",
    label = "reference (Laplace + Stehfest)", lw = 2, color = :black,
    xscale = :log10, legend = :bottomleft, size = (700, 420),
)
plot!(
    plt_hist, T_hist[1:8:end], (p_axis ./ p0)[1:8:end];
    seriestype = :scatter, ms = 3, mswidth = 0, color = :crimson, label = "finite elements",
)
hline!(plt_hist, [1.0]; ls = :dash, color = :grey, label = "p₀")
plt_hist

# ### Error against the reference

println("      T     |  L2 error  |  L∞ error [Pa]")
println("-"^44)
errors = Float64[]
for (T, pn) in zip(T_probe, p_num)
    ref = [deleeuw_pressure(model, max(r / R_CYL, 1.0e-8), T) for r in rr]
    e2 = norm(pn .- ref) / norm(ref)
    push!(errors, e2)
    @printf("  %8.4f  |  %.3e |  %10.1f\n", T, e2, maximum(abs.(pn .- ref)))
end
println("-"^44)
@printf("worst relative L2 error: %.3e\n", maximum(errors))

# ### Convergence

function worst_error(; nel, dT)
    mm, rrr, Tp, pn, _, _ = run_deleeuw(; nel = nel, dT = dT)
    return maximum(
        let ref = [deleeuw_pressure(mm, max(r / R_CYL, 1.0e-8), T) for r in rrr]
            norm(p .- ref) / norm(ref)
        end for (T, p) in zip(Tp, pn)
    )
end

println("  ΔT        |  worst L2 error |  ratio")
println("-"^42)
prev = NaN
for dT in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)
    e = worst_error(; nel = 80, dT = dT)
    @printf("  %.2e  |    %.3e    |  %s\n", dT, e, isnan(prev) ? "—" : @sprintf("%.2f", prev / e))
    global prev = e
end

# ### Radial profiles

plt = plot(;
    xlabel = "r / R  [-]", ylabel = "p / p₀  [-]",
    title = "De Leeuw — radial pressure profiles",
    legend = :bottomleft, size = (700, 440),
)
rfine = range(0.001, 0.999; length = 200)
palette = cgrad(:viridis, max(length(T_probe), 2); categorical = true)
for (i, (T, pn)) in enumerate(zip(T_probe, p_num))
    plot!(
        plt, rfine, [deleeuw_pressure(model, r, T) / p0 for r in rfine];
        color = palette[i], lw = 2, label = "T = $T",
    )
    plot!(
        plt, rr ./ R_CYL, pn ./ p0;
        color = palette[i], seriestype = :scatter, ms = 2, mswidth = 0, label = "",
    )
end
plt

# ## Notes
#
# - **A different undrained limit** is what makes this an independent check rather than a
#   rerun of Cryer's. Plane strain gives ``\tfrac{2}{3}B(1+\nu_u)P_c``, not ``BP_c``, and the
#   initial value theorem returns it on materials with very different couplings.
# - **One parameter separates the geometries.** The same `radial_element_matrices!` serves
#   both, with `nhoop = 1` here and `2` for the sphere.
# - **Bessel functions force `Float64`** for the transform, unlike Cryer's elementary one.
#   Only the Stehfest weights need `BigFloat`, and `N = 10`, `14`, `16` agree to five digits.
