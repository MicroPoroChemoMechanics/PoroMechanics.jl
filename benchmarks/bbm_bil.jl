# # Barcelona Basic Model — the Bil reference cases
#
# The three reference cases distributed with [Bil](https://github.com/dangla/bil) as
# `base/BBM`, `base/BBM2` and `base/BBM_pcst`. Bil is the C code this package is meant to
# replace, and its BBM implementation is independent of this one, so agreeing with it is a
# real check — the more so because the Barcelona Basic Model has no closed-form solution to
# fall back on.
#
# Each case is a single axisymmetric element of unit size, loaded by a normal pressure on
# its two free faces and by a suction imposed uniformly over the domain. The response is
# therefore homogeneous, and the finite element problem reduces exactly to a stress-driven
# material point — which is how it is run here. That reduction is not an assumption: the
# axisymmetric solver in this package was checked against the material point on the same
# model and agreed to twelve digits, so nothing is hidden by taking the shorter route, and
# the discretisation error of a one-element mesh is not there to muddy the comparison.
#
# ## The three paths
#
# | Case | Path | What it exercises |
# |---|---|---|
# | `BBM` | isotropic ``p`` cycled 1→40→1→80→1→160→1 kPa, suction stepped to 40, 80, 160 kPa | the loading–collapse curve |
# | `BBM_pcst` | saturated, ``p`` to 20 kPa then ``q`` to 23.76 kPa, constant ``G`` | the yield surface, approached to within 1 % |
# | `BBM2` | ``p`` and ``q`` cycled together, with suction | the deviatoric return map |
#
# `BBM_pcst` is the sharpest of the three. At ``\bar p = 20`` kPa with ``p_0 = 40`` kPa and
# ``s = 0``, the yield surface is reached at ``q = M\sqrt{\bar p (p_0 - \bar p)} = 24`` kPa,
# and the case stops at 23.76 kPa. A yield surface misplaced by one per cent would show up
# as spurious plastic strain.

using PoroMechanics
using Tensors
using Printf

# ## Parameters
#
# Taken verbatim from the Bil decks: ``\kappa = 0.011``, ``\lambda(0) = 0.065``,
# ``r = 0.75``, ``\beta = 2\times10^{-5}`` Pa⁻¹, ``M = 1.2``, ``k_s = 0.8``,
# ``\kappa_s = 0.005``, ``\nu = 0.15``, ``p^c = 10`` kPa, ``p_{c0}^* = 40`` kPa, and an
# initial porosity of 0.25, so ``e_0 = \phi/(1-\phi) = 1/3``. These are the defaults of
# [`BBM`](@ref), so the constructor is called bare.

material = BBM()
pc_star0 = 40.0e3       # initial preconsolidation [Pa]
σ0 = -1.0e3 * one(SymmetricTensor{2, 3})   # in-situ stress, isotropic 1 kPa compression

# ## Loading
#
# The decks prescribe piecewise-linear histories of ``p`` and ``q`` in kPa and of the
# suction, and the two free faces carry the resulting normal pressures. Working back from
# the deck's field values, the faces impose
#
# ```math
# \sigma_{rr} = \sigma_{\theta\theta} = -(p - 0.33\,q), \qquad \sigma_{zz} = -(p + 0.66\,q)
# ```
#
# whose mean is ``-p`` and whose deviator is ``0.99\,q`` — the deck rounds ``1/3`` and
# ``2/3``, so the deviatoric stress it actually applies is 99 % of its nominal value. That
# rounding is kept rather than corrected, because the point is to reproduce the case as
# run.

function piecewise_linear(ts, fs, t)
    t <= first(ts) && return first(fs)
    t >= last(ts) && return last(fs)
    i = findlast(<=(t), ts)
    i == length(ts) && return last(fs)
    return fs[i] + (fs[i + 1] - fs[i]) * (t - ts[i]) / (ts[i + 1] - ts[i])
end

"Stress imposed by the two loaded faces, for nominal `p` and `q` in kPa."
function face_stress(p_kPa, q_kPa)
    return SymmetricTensor{2, 3}(
        (i, j) -> i != j ? 0.0 :
            i == 2 ? -(1.0e3 * p_kPa + 0.66e3 * q_kPa) : -(1.0e3 * p_kPa - 0.33e3 * q_kPa)
    )
end

# ## Driver
#
# The path is prescribed in stress, so the strain is found by
# [`stress_controlled_response`](@ref) — Newton on the material response, using the same
# algorithmic tangent the global solve uses.
#
# Every case is run **twice**, because the two codes do not integrate the elastic law the
# same way. Bil freezes the bulk modulus at the incoming state and steps forward, which is
# first-order accurate; this package integrates ``K = \bar p(1+e_0)/\kappa`` in closed form,
# which has no step-size error at all. Comparing with Bil therefore needs
# [`ExplicitPredictor`](@ref), which reproduces Bil's scheme, and the exact result is shown
# beside it so the difference between the two is visible rather than argued about.

function run_case(m, p_of, q_of, s_of, t_end, dates; Δt = 1.0e-3)
    state = initial_state(m, σ0, pc_star0; suction = s_of(0.0))
    out = Dict{Float64, NamedTuple}()
    for k in 1:round(Int, t_end / Δt)
        t = k * Δt
        ε, σ, state, _ = stress_controlled_response(
            m, face_stress(p_of(t), q_of(t)), s_of(t), state, Δt
        )
        for d in dates
            isapprox(t, d; atol = Δt / 4) && (
                out[d] = (
                    p = mean_pressure(σ), q = equivalent_stress(σ), εv = tr(ε),
                    εv_p = state.εv_p, pc_star = state.pc_star,
                )
            )
        end
    end
    return out
end

# ## Case 1 — `base/BBM`: the loading–collapse curve
#
# Isotropic compression cycled three times, each cycle followed by a step of drying. The
# suction steps expand the yield surface through the LC curve, so each successive loading
# reaches further before yielding, and the plastic strain accumulates in steps.

p_bbm = t -> piecewise_linear([0, 1, 2, 3, 4, 5, 6], [1, 40, 1, 80, 1, 160, 1], t)
s_bbm = t -> 1.0e3 * piecewise_linear(
    [0, 1.999, 2, 3.999, 4, 5.999, 6], [0, 0, 40, 40, 80, 80, 160], t
)
bbm_dates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
bbm = run_case(ExplicitPredictor(material), p_bbm, t -> 0.0, s_bbm, 6.0, bbm_dates)
bbm_exact = run_case(material, p_bbm, t -> 0.0, s_bbm, 6.0, bbm_dates)

# Bil's own results at the same dates, read from `base/BBM/BBM.p1`. The volumetric strain
# is recovered from the void-ratio change it reports, ``\Delta e = (1+e_0)\,\mathrm{tr}\,
# \varepsilon``, and the hardening variable it stores is ``\ln p_c^*``.

bbm_ref = Dict(                          # (tr ε, εv_p, pc* [Pa])
    1.0 => (-3.058474e-2, 0.0, 39999.8),
    2.0 => (-1.550205e-3, 0.0, 39999.8),
    3.0 => (-5.207435e-2, 1.411818e-2, 56683.0),
    4.0 => (-1.711651e-2, 1.411818e-2, 56683.0),
    5.0 => (-7.440369e-2, 2.917845e-2, 82214.7),
    6.0 => (-3.426270e-2, 2.917845e-2, 82214.7),
)

rel(a, b) = abs(b) < 1.0e-12 ? abs(a - b) : abs(a - b) / abs(b)

function comparison_table(explicit, exact, ref)
    println("      │           Bil │   Bil's scheme reproduced │  exact elastic integration")
    println("    t │  εv_p     pc* │      εv_p      pc*    tr ε │      εv_p      pc*    tr ε")
    for (d, (rεv, rεp, rpc)) in sort(collect(ref))
        a, b = explicit[d], exact[d]
        @printf(
            "%5.0f │ %.5f %6.2f │ %.1e %.1e %.1e │ %.1e %.1e %.1e\n",
            d, rεp, rpc / 1.0e3,
            rel(a.εv_p, rεp), rel(a.pc_star, rpc), rel(a.εv, rεv),
            rel(b.εv_p, rεp), rel(b.pc_star, rpc), rel(b.εv, rεv)
        )
    end
    return nothing
end

comparison_table(bbm, bbm_exact, bbm_ref)

# Reproducing Bil's scheme, the plastic variables — the substance of the model — agree to
# about ``10^{-4}``. Integrating exactly moves the total strain away from Bil by up to 20 %,
# and that is the expected direction: the exact answer is not Bil's answer, it is the one
# Bil is converging towards. The convergence study at the end of this page shows it is Bil
# that moves.
#
# ## Case 2 — `base/BBM_pcst`: the yield surface, approached to 1 %
#
# Saturated, so the LC curve is inactive and the suction terms drop out. The deck fixes
# ``G = 10^8`` Pa instead of deriving it from ``\nu``, which [`BBM`](@ref) supports through
# `G_const`.

pcst_args = (
    t -> piecewise_linear([0, 1], [1, 20], t),
    t -> piecewise_linear([0, 1, 2], [0, 0, 24], t),
    t -> 0.0, 2.0, [1.0, 2.0],
)
pcst = run_case(ExplicitPredictor(BBM(G_const = 1.0e8)), pcst_args...)
pcst_exact = run_case(BBM(G_const = 1.0e8), pcst_args...)

comparison_table(
    pcst, pcst_exact,
    Dict(1.0 => (-2.478797e-2, 0.0, 39999.8), 2.0 => (-2.478797e-2, 0.0, 39999.8))
)

# Both codes stay elastic throughout, as they must: the deviatoric loading stops 1 % short
# of the yield surface. The volumetric strain is unchanged by the deviatoric leg, which is
# the signature of an elastic response at constant mean stress, and it matches Bil to
# ``6\times10^{-5}``.
#
# ## Case 3 — `base/BBM2`: a reference that disagrees with its own model
#
# The third case cycles ``p`` and ``q`` together while stepping the suction. Run as
# specified, this package and Bil agree exactly while the sample is saturated and then part
# company completely once the suction is applied — 45 % on the plastic strain, 50 % on the
# hardening variable.
#
# The disagreement is not a matter of opinion, and it is not resolved by trusting either
# code. Bil's published output can be tested against Bil's own yield function, with Bil's
# own parameters:
#
# ```math
# f = \frac{q^2}{M^2} + (p - k_s s)\,(p + p_c), \qquad
# p_c = p^c\left(\frac{p_c^*}{p^c}\right)^{\lambda(0)-\kappa)/(\lambda(s)-\kappa)}
# ```
#
# A state that is accumulating plastic strain must sit on the yield surface, ``f = 0``.
# Evaluated along `BBM2.p1`, ``f`` is zero to within round-off while the sample is
# saturated and then falls to ``-2823``, ``-5468``, ``-11585`` and finally ``-59532``
# kPa² — strictly *inside* the elastic domain, while the plastic strain keeps growing.
# That is not something a consistent return mapping can produce.
#
# Solving instead for the loading–collapse exponent that would put Bil's reported state
# back on its own surface gives ``lc = 1.00052`` at ``s = 40`` kPa, where the deck's LC
# curve calls for 1.1986. And the deck explains where 1.00052 comes from: `BBM2` loads a
# file `wrc` that `base/BBM` does not, whose fifth column is a two-point table of the LC
# factor, 1 at zero suction and 1.39 at 30 MPa. Interpolated linearly,
# ``1 + 0.39 \times 40/30000 = 1.00052`` — the solved value, to five digits. The reference
# run picked up that coarse table instead of the finely sampled `Curves = lc` line further
# down the deck, so it was computed with the loading–collapse coupling effectively switched
# off.
#
# The case is therefore reproduced as it was actually run, with ``\lambda(s) = \lambda(0)``,
# which `r = 1` produces exactly.

bbm2_args = (
    t -> piecewise_linear([0, 1, 2, 3, 4, 5], [1, 40, 1, 80, 1, 160], t),
    t -> piecewise_linear([0, 1, 2, 3, 4, 5], [1, 20, 1, 40, 1, 80], t),
    t -> 1.0e3 * piecewise_linear([0, 1.999, 2, 3.999, 4, 5], [0, 0, 40, 40, 80, 80], t),
    5.0, [1.0, 2.0, 3.0, 4.0, 5.0],
)
bbm2 = run_case(ExplicitPredictor(BBM(r = 1.0)), bbm2_args...)
bbm2_exact = run_case(BBM(r = 1.0), bbm2_args...)

comparison_table(
    bbm2, bbm2_exact, Dict(
        1.0 => (-3.693599e-2, 6.364060e-3, 46806.2),
        2.0 => (-7.892678e-3, 6.364060e-3, 46806.2),
        3.0 => (-7.055920e-2, 3.267174e-2, 89620.6),
        4.0 => (-3.555090e-2, 3.267174e-2, 89620.6),
        5.0 => (-1.056925e-1, 6.066891e-2, 178908.5),
    )
)

# The agreement returns: ``2\times10^{-3}`` on the plastic strain and ``3\times10^{-3}`` on
# the hardening variable, against 45 % and 50 % before. What is left is itself accounted
# for — the two-point table gives ``lc = 1.00104`` at ``s = 80`` kPa rather than exactly 1,
# which is a 0.3 % effect on ``p_c^*``, and 0.3 % is what remains at ``t = 5``.
#
# ## Which code is converging, and to what
#
# The first loading leg is elastic throughout, so it has a closed-form answer,
# ``\varepsilon_v = -\frac{\kappa}{1+e_0}\ln\frac{40}{1}``, and the two schemes can be
# measured against it rather than against each other.

exact_leg1 = -(material.κ / (1 + material.e0)) * log(40.0)
@printf("tr(ε) at t = 1, closed form: %+.9e\n\n", exact_leg1)
@printf("%-9s  %-12s %-12s   %-12s %-12s\n", "Δt", "Bil's scheme", "error", "exact", "error")
for Δt in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4)
    a = run_case(ExplicitPredictor(material), p_bbm, t -> 0.0, s_bbm, 6.0, [1.0]; Δt = Δt)
    b = run_case(material, p_bbm, t -> 0.0, s_bbm, 6.0, [1.0]; Δt = Δt)
    @printf(
        "%-9.1e  %+.6e %.3e   %+.6e %.3e\n", Δt,
        a[1.0].εv, rel(a[1.0].εv, exact_leg1), b[1.0].εv, rel(b[1.0].εv, exact_leg1)
    )
end
@printf(
    "%-9s  %+.6e %.3e\n", "Bil", bbm_ref[1.0][1], rel(bbm_ref[1.0][1], exact_leg1)
)

# Two different things are on display. The incremental scheme's error halves as the step
# halves — first order, as expected of a forward Euler step on ``d\bar p = K\,
# d\varepsilon_v`` — and Bil's own error sits on that same line, between this package's
# values at ``\Delta t = 10^{-3}`` and ``5\times10^{-4}``, which is where Bil should be:
# it starts from ``\Delta t = 10^{-4}`` and ramps up to ``10^{-3}``, so its effective step
# is slightly the finer.
#
# The exact scheme has no error to converge, at any step. It is not a better approximation
# of the elastic law; it *is* the elastic law, because
# ``d\varepsilon_v = \frac{\kappa}{1+e_0}\frac{d\bar p}{\bar p}`` separates and integrates.
# The 20 % gap against Bil at ``t = 6`` is six loading legs' worth of accumulated
# first-order error in the reference, not a disagreement about the model.
#
# ## What this does and does not change
#
# The plastic variables barely move between the two schemes, and barely move with the step
# either.

for Δt in (2.0e-3, 5.0e-4)
    a = run_case(ExplicitPredictor(material), p_bbm, t -> 0.0, s_bbm, 6.0, [6.0]; Δt = Δt)
    b = run_case(material, p_bbm, t -> 0.0, s_bbm, 6.0, [6.0]; Δt = Δt)
    @printf(
        "Δt = %.1e   εv_p at t=6:  Bil's scheme %+.7e   exact %+.7e\n",
        Δt, a[6.0].εv_p, b[6.0].εv_p
    )
end
@printf("%22s  Bil reports  %+.7e\n", "", bbm_ref[6.0][2])

# That is why the plastic variables carry the comparison with Bil, and why the agreement to
# ``10^{-4}`` on them is the meaningful result: they are fixed by the yield surface and the
# hardening law, which are algebraic relations both codes implement identically, not by how
# the elastic predictor is stepped.
#
# What the exact integration buys is that the total strain — the quantity a laboratory
# actually measures, and the one a calibration would fit — no longer carries a discretisation
# error that a user would have to discover by refining. It also removes a spurious
# dependence on the load path: an incremental hypoelastic law does not return to the same
# state after a closed stress cycle, and the exact one does.

let m = material
    st = initial_state(m, σ0, pc_star0; suction = 0.0)
    ste = initial_state(ExplicitPredictor(m), σ0, pc_star0; suction = 0.0)
    for k in 1:200                                   # 1 → 20 → 1 kPa, entirely elastic
        p_k = k <= 100 ? 1 + 19 * k / 100 : 20 - 19 * (k - 100) / 100
        _, _, st, _ = stress_controlled_response(m, face_stress(p_k, 0.0), 0.0, st, 1.0)
        _, _, ste, _ = stress_controlled_response(
            ExplicitPredictor(m), face_stress(p_k, 0.0), 0.0, ste, 1.0
        )
    end
    @printf("residual strain after a closed elastic cycle 1 → 20 → 1 kPa\n")
    @printf("  Bil's scheme : %+.3e\n", tr(ste.ε))
    @printf("  exact        : %+.3e\n", tr(st.ε))
end
