# Validation benchmarks — the numerical solution against a closed-form reference.
#
# These are the counterpart of the regression tests. Regression asks "did the answer
# change?"; validation asks "is the answer right?". A benchmark script is included once,
# in its own module, and the errors it computed at load time are asserted here — the same
# arrangement as `regression.jl`, and for the same reason: the numbers the documentation
# page displays are exactly the numbers the tests check.

using LinearAlgebra: norm

module _Terzaghi
    include("../benchmarks/terzaghi.jl")
end

module _Mandel
    include("../benchmarks/mandel.jl")
end

module _Cryer
    include("../benchmarks/cryer.jl")
end

module _DeLeeuw
    include("../benchmarks/deleeuw.jl")
end

module _Gardner
    include("../benchmarks/gardner_infiltration.jl")
end

module _GardnerTransient
    include("../benchmarks/gardner_transient.jl")
end

module _BilBBM
    include("../benchmarks/bbm_bil.jl")
end

@testset "Terzaghi 1D consolidation" begin
    ## Tolerance sits an order of magnitude above the measured error, so ordinary
    ## floating-point or solver-version drift does not trip it, while a real regression
    ## in the poroelastic assembly would.
    @test maximum(_Terzaghi.errors) < 5.0e-3

    ## The reference series must reproduce the square wave it expands at T = 0.
    for Z in (0.1, 0.35, 0.6, 0.9)
        @test _Terzaghi.terzaghi_pressure(Z, 0.0) ≈ 1.0
        @test isapprox(_Terzaghi.terzaghi_pressure(Z, 1.0e-6), 1.0; atol = 1.0e-2)
    end

    ## Drained surface and impermeable base
    @test _Terzaghi.terzaghi_pressure(0.0, 0.1) ≈ 0.0 atol = 1.0e-12
    @test _Terzaghi.terzaghi_pressure(1.0, 0.1) > _Terzaghi.terzaghi_pressure(0.5, 0.1)

    ## Derived constants, against the closed forms they come from
    m = _Terzaghi.TERZAGHI_MATERIAL
    F = _Terzaghi.F_LOAD
    Mo = _Terzaghi.oedometric_modulus(m)
    @test Mo ≈ m.E * (1 - m.nu) / ((1 + m.nu) * (1 - 2m.nu))
    @test _Terzaghi.consolidation_coefficient(m) ≈ (m.k / m.mu_l) / (m.N + m.b^2 / Mo)
    @test _Terzaghi.initial_pressure(m) ≈ F * m.b / (Mo * m.N + m.b^2)

    ## With b = 1 the undrained pressure falls short of the load by exactly the
    ## compressibility of the fluid and grains, p₀ = F / (1 + M_o N) — about 0.1 % here.
    @test m.b == 1.0
    @test _Terzaghi.initial_pressure(m) ≈ F / (1 + Mo * m.N)
    @test 0.998 < _Terzaghi.initial_pressure(m) / F < 1.0

    ## Backward Euler is first order: halving ΔT must halve the error.
    e_coarse = _Terzaghi.worst_error(; nely = 60, dT = 1.0e-3)
    e_fine = _Terzaghi.worst_error(; nely = 60, dT = 5.0e-4)
    @test 1.8 < e_coarse / e_fine < 2.2
end

@testset "Mandel's problem" begin
    m = _Mandel.MANDEL_MATERIAL
    roots = _Mandel.mandel_roots(m)

    @test maximum(_Mandel.errors) < 5.0e-3

    ## Derived poroelastic constants, against their closed forms
    K = _Mandel.bulk_modulus(m)
    B = _Mandel.skempton(m)
    @test B ≈ m.b / (m.N * K + m.b^2)
    @test _Mandel.undrained_poisson(m) > m.nu          # draining stiffens the material
    @test _Mandel.undrained_poisson(m) < 0.5

    ## The eigenvalue condition, and the first root that is easy to miss
    k = (1 - m.nu) / (_Mandel.undrained_poisson(m) - m.nu)
    @test k > 1
    @test 0 < roots[1] < π / 2                          # outside the (nπ, nπ+π/2) pattern
    for (n, α) in enumerate(roots[1:6])
        @test tan(α) ≈ k * α rtol = 1.0e-8
        n > 1 && @test (n - 1) * π < α < (n - 1) * π + π / 2
    end

    ## At t → 0 the series must return the uniform initial pressure. It converges slowly
    ## there, so the check is that the gap *shrinks with the term count* — truncation, not
    ## a wrong formula.
    p0 = _Mandel.initial_pressure(m)
    gaps = [
        abs(_Mandel.mandel_pressure(m, _Mandel.mandel_roots(m; nterms = n), 0.3, 1.0e-12) - p0) / p0
            for n in (60, 200, 800, 3000)
    ]
    @test issorted(gaps; rev = true)
    @test gaps[1] < 1.0e-2
    @test gaps[end] < 2.0e-4
    for x in (0.0, 0.3, 0.7)
        @test isapprox(_Mandel.mandel_pressure(m, roots, x, 1.0e-12), p0; rtol = 1.5e-2)
    end
    @test _Mandel.mandel_pressure(m, roots, _Mandel.A_HALF, 0.1) ≈ 0.0 atol = 1.0e-6

    ## The Mandel–Cryer effect itself: the center pressure must overshoot p₀, and the
    ## numerical solution must reproduce both the height and the timing of the peak.
    ref = [_Mandel.mandel_pressure(m, roots, 0.0, T) for T in _Mandel.T_hist]
    i_ref = argmax(ref)
    i_num = argmax(_Mandel.p_centre)

    @test ref[i_ref] / p0 > 1.05                        # a genuine overshoot, not noise
    @test _Mandel.p_centre[i_num] / p0 > 1.05
    @test isapprox(_Mandel.p_centre[i_num], ref[i_ref]; rtol = 5.0e-3)
    @test isapprox(_Mandel.T_hist[i_num], _Mandel.T_hist[i_ref]; rtol = 5.0e-2)

    ## ...and it must be a rise then a fall, not a monotone decay
    @test _Mandel.p_centre[i_num] > _Mandel.p_centre[1]
    @test _Mandel.p_centre[end] < _Mandel.p_centre[i_num]
end

@testset "Gardner steady infiltration" begin
    m = _Gardner.model

    @test _Gardner.err_L2 < 1.0e-4

    ## The closed form must satisfy the equation it was derived from: recover the imposed
    ## flux by differentiating the profile. This checks the reference itself, independently
    ## of the finite volume solution it is used to validate.
    Ks = _Gardner.saturated_conductivity(m)
    for z in (0.2, 0.8, 1.5)
        h = 1.0e-6
        dpdz = (_Gardner.gardner_profile(m, z + h) - _Gardner.gardner_profile(m, z - h)) / 2h
        krl = _Gardner.krl(m, m.p_g - _Gardner.gardner_profile(m, z))
        q = -Ks * krl * (dpdz + m.rho_l * abs(m.gravite))
        @test isapprox(q, m.q_top; rtol = 1.0e-6)
    end

    ## Water table condition, and a profile drier than hydrostatic because water is
    ## being added at the top.
    @test _Gardner.gardner_profile(m, 0.0) ≈ 0.0 atol = 1.0e-12
    for z in (0.5, 1.0, 2.0)
        @test _Gardner.gardner_profile(m, z) > -m.rho_l * abs(m.gravite) * z
    end

    ## Upward flux cannot be sustained indefinitely: above a finite height the closed form
    ## ceases to exist, and must say so rather than return a number.
    dry = _Gardner.GardnerColumn(; q_top = 1.0e-7)
    @test isnan(_Gardner.gardner_profile(dry, 50.0))

    ## Second order in space on a uniform grid.
    errs = map((50, 100, 200)) do N
        mm, zz, pp, _ = _Gardner.run_gardner(; N = N)
        ref = [_Gardner.gardner_profile(mm, zi) for zi in zz]
        norm(pp .- ref) / norm(ref)
    end
    @test 3.8 < errs[1] / errs[2] < 4.2
    @test 3.8 < errs[2] / errs[3] < 4.2
end

@testset "Cryer's sphere" begin
    m = _Cryer.CRYER_MATERIAL
    B = _Cryer.skempton(m)
    Pc = _Cryer.P_CONF
    p0 = B * Pc

    @test maximum(_Cryer.errors) < 2.0e-2

    ## The reference is derived here, so it is checked here too, by the limit theorems it
    ## must satisfy — independently of the finite element solution it validates.
    ##
    ## Initial value: the undrained response to isotropic loading is Skempton's, uniform.
    for r in (0.1, 0.5, 0.9)
        @test isapprox(_Cryer.cryer_pressure(m, r, 1.0e-6), p0; rtol = 5.0e-3)
    end
    ## Final value: everything drains.
    @test _Cryer.cryer_pressure(m, 0.5, 20.0) / p0 < 1.0e-6
    ## Drained surface.
    @test abs(_Cryer.cryer_pressure(m, 0.999999, 0.1)) < 1.0e-5 * Pc

    ## The storage coefficient of the derivation must be the package's own.
    @test _Cryer.storage_coefficient(m) ≈ m.N + m.b^2 / _Cryer.oedometric_modulus(m)
    @test _Cryer.consolidation_coefficient(m) ≈ (m.k / m.mu_l) / _Cryer.storage_coefficient(m)

    ## The Mandel–Cryer effect, and the fact that the sphere overshoots far harder than the
    ## plane-strain slab — 23 % against 6.5 %.
    ref = [_Cryer.cryer_pressure(m, 1.0e-8, T) for T in _Cryer.T_hist]
    i_ref = argmax(ref)
    i_num = argmax(_Cryer.p_centre)

    @test ref[i_ref] / p0 > 1.2
    @test _Cryer.p_centre[i_num] / p0 > 1.2
    @test isapprox(_Cryer.p_centre[i_num], ref[i_ref]; rtol = 5.0e-3)
    @test isapprox(_Cryer.T_hist[i_num], _Cryer.T_hist[i_ref]; rtol = 5.0e-2)
    @test _Cryer.p_centre[end] < _Cryer.p_centre[i_num]

    ## Backward Euler, first order in time.
    e_coarse = _Cryer.worst_error(; nel = 80, dT = 2.0e-3)
    e_fine = _Cryer.worst_error(; nel = 80, dT = 1.0e-3)
    @test 1.8 < e_coarse / e_fine < 2.2
end

@testset "De Leeuw's cylinder" begin
    m = _DeLeeuw.DELEEUW_MATERIAL
    Pc = _DeLeeuw.P_LAT
    p0 = _DeLeeuw.undrained_pressure(m)

    @test maximum(_DeLeeuw.errors) < 2.0e-2

    ## The undrained limit here is NOT Skempton's B·Pc. Plane strain forbids ε_zz, so
    ## σ_zz = ν_u(σ_rr + σ_θθ) and the mean stress carries an extra (1+ν_u) factor. That
    ## different limit is what makes this an independent check rather than a rerun of Cryer.
    @test p0 ≈ 2 * _DeLeeuw.skempton(m) * (1 + _DeLeeuw.undrained_poisson(m)) * Pc / 3
    @test p0 < _DeLeeuw.skempton(m) * Pc
    for r in (0.1, 0.5, 0.9)
        @test isapprox(_DeLeeuw.deleeuw_pressure(m, r, 1.0e-6), p0; rtol = 5.0e-3)
    end
    @test _DeLeeuw.deleeuw_pressure(m, 0.5, 20.0) / p0 < 1.0e-5
    @test abs(_DeLeeuw.deleeuw_pressure(m, 0.999999, 0.1)) < 1.0e-5 * Pc

    ## The overshoot, and its place in the sequence slab < cylinder < sphere.
    ref = [_DeLeeuw.deleeuw_pressure(m, 1.0e-8, T) for T in _DeLeeuw.T_hist]
    i_ref = argmax(ref)
    i_num = argmax(_DeLeeuw.p_axis)

    @test 1.08 < ref[i_ref] / p0 < 1.15
    @test isapprox(_DeLeeuw.p_axis[i_num], ref[i_ref]; rtol = 5.0e-3)
    @test isapprox(_DeLeeuw.T_hist[i_num], _DeLeeuw.T_hist[i_ref]; rtol = 5.0e-2)
    @test _DeLeeuw.p_axis[end] < _DeLeeuw.p_axis[i_num]

    ## Cylinder strictly between slab and sphere, on the same material.
    cryer_ref = [_Cryer.cryer_pressure(_Cryer.CRYER_MATERIAL, 1.0e-8, T) for T in _Cryer.T_hist]
    cryer_peak = maximum(cryer_ref) / (_Cryer.skempton(_Cryer.CRYER_MATERIAL) * _Cryer.P_CONF)
    @test ref[i_ref] / p0 < cryer_peak

    ## Backward Euler, first order in time.
    e_coarse = _DeLeeuw.worst_error(; nel = 80, dT = 2.0e-3)
    e_fine = _DeLeeuw.worst_error(; nel = 80, dT = 1.0e-3)
    @test 1.8 < e_coarse / e_fine < 2.2
end

@testset "Gardner transient drainage" begin
    M = _GardnerTransient
    m = M.model

    @test maximum(M.errors) < 5.0e-4

    ## The change of variable that makes this solvable: the retention and permeability
    ## curves must share their exponent, or the equation is not linear in K*.
    @test m.retention.α == m.rel_perm.α == m.alpha

    ## The reference must return the initial steady state at t → 0 and the final one at
    ## t → ∞ — the two limit theorems, on the profile rather than a single point.
    for zi in (0.4, 1.0, 1.8)
        @test isapprox(
            M.transient_pressure(m, zi, 1.0e-3),
            log(M.kstar_steady(m, zi, M.Q_BEFORE)) / m.alpha; rtol = 1.0e-6
        )
        @test isapprox(
            M.transient_pressure(m, zi, 1.0e9),
            log(M.kstar_steady(m, zi, M.Q_AFTER)) / m.alpha; rtol = 1.0e-6
        )
    end

    ## Heavier infiltration wets the column: pressures rise everywhere above the water table.
    for zi in (0.4, 1.0, 1.8)
        @test log(M.kstar_steady(m, zi, M.Q_AFTER)) > log(M.kstar_steady(m, zi, M.Q_BEFORE))
    end
    ## ...and monotonically in time, since a single flux step drives it.
    for zi in (0.5, 1.5)
        ps = [M.transient_pressure(m, zi, t) for t in (1.0e4, 1.0e5, 1.0e6, 1.0e7)]
        @test issorted(ps)
    end

    ## The steady limit of this problem is the companion benchmark's solution.
    @test isapprox(
        log(M.kstar_steady(m, 1.0, M.Q_AFTER)) / m.alpha,
        _Gardner.gardner_profile(_Gardner.GardnerColumn(; q_top = M.Q_AFTER), 1.0);
        rtol = 1.0e-10
    )

    ## Second order in space, cleanly. Two levels only — the reference costs a BigFloat
    ## Stehfest inversion per node, so a third level would dominate the suite's runtime.
    e = [M.errors_at(; N = N, Δt = 2.0e3)[end] for N in (100, 200)]
    @test 3.8 < e[1] / e[2] < 4.2
end

@testset "BBM — cas de référence Bil" begin
    B = _BilBBM
    rel(a, b) = abs(a - b) / abs(b)

    ## `base/BBM` — the loading–collapse curve. The plastic variables carry the comparison:
    ## they are set by the yield surface and the hardening law, which are algebraic, while
    ## the total strain also carries the first-order error of the explicit hypoelastic
    ## predictor that both codes share.
    for (d, (εv, εv_p, pc)) in B.bbm_ref
        r = B.bbm[d]
        @test rel(r.εv, εv) < 3.0e-2
        @test rel(r.pc_star, pc) < 1.0e-3
        εv_p > 0 && @test rel(r.εv_p, εv_p) < 1.0e-3
    end
    @test B.bbm[6.0].εv_p ≈ 2.917845e-2 rtol = 1.0e-3

    ## `base/BBM_pcst` — the deviatoric leg stops 1 % short of the yield surface, so the
    ## response must stay elastic. A yield surface misplaced by a per cent would not.
    @test B.pcst[2.0].εv_p == 0
    @test B.pcst[2.0].pc_star == 40.0e3
    @test B.pcst[2.0].εv ≈ B.pcst[1.0].εv          # no volumetric strain at constant p̄
    @test rel(B.pcst[2.0].εv, -2.478797e-2) < 1.0e-4
    @test B.pcst[2.0].q ≈ 23.76e3 rtol = 1.0e-6

    ## `base/BBM_pcst` sits just inside the surface, and that is what makes it sharp: at
    ## p̄ = 20 kPa with p₀ = 40 kPa and no suction, yield is at q = M√(p̄(p₀−p̄)) = 24 kPa.
    m = BBM(G_const = 1.0e8)
    @test yield_function(m, 20.0e3, 24.0e3, 0.0, 40.0e3) ≈ 0 atol = 1.0e-6
    @test yield_function(m, 20.0e3, 23.76e3, 0.0, 40.0e3) < 0

    ## `base/BBM2` — reproduced with the loading–collapse coupling switched off, which is
    ## how the reference was actually run. See the benchmark page for the two independent
    ## proofs; the assertion here is the one that does not depend on the diagnosis, namely
    ## that Bil's own output violates Bil's own yield function once suction is applied.
    @test rel(B.bbm2[5.0].εv_p, 6.066891e-2) < 5.0e-3
    @test rel(B.bbm2[5.0].pc_star, 178908.5) < 5.0e-3

    ## The published `BBM2` state at t = 3 is strictly inside the yield surface implied by
    ## its own reported hardening variable, while carrying plastic strain — impossible for
    ## a consistent return mapping, and the reason the case is run with r = 1.
    lc = (0.065 - 0.011) / (0.065 * (0.25 * exp(-2.0e-5 * 40.0e3) + 0.75) - 0.011)
    p0_lc = 1.0e4 * (89620.6 / 1.0e4)^lc
    @test yield_function(BBM(), 80.0e3, 39.6e3, 40.0e3, 89620.6) < -5.0e6
    @test p0_lc > 130.0e3                          # 138.5 kPa, against p̄ = 80 kPa

    ## With the coupling off the same state is on the surface, to a per cent.
    @test abs(yield_function(BBM(r = 1.0), 80.0e3, 39.6e3, 40.0e3, 89620.6)) < 2.0e-2 *
        1.2^2 * (80.0e3 + 32.0e3) * 10.0e3

    ## The first loading leg is elastic throughout, so it has a closed form to measure both
    ## integration schemes against.
    closed_form = -(BBM().κ / (1 + BBM().e0)) * log(40.0)
    errs = Float64[]
    for Δt in (2.0e-3, 1.0e-3, 5.0e-4)
        a = B.run_case(
            ExplicitPredictor(BBM()), B.p_bbm, t -> 0.0, B.s_bbm, 6.0, [1.0, 6.0]; Δt = Δt
        )
        b = B.run_case(BBM(), B.p_bbm, t -> 0.0, B.s_bbm, 6.0, [1.0, 6.0]; Δt = Δt)
        push!(errs, abs(a[1.0].εv - closed_form) / abs(closed_form))

        ## The exact scheme has no step-size error to converge: it is the elastic law, not
        ## an approximation of it.
        @test b[1.0].εv ≈ closed_form rtol = 1.0e-10

        ## And the plastic strain is the same either way, which is why it — not the total
        ## strain — is what carries the comparison with Bil.
        @test a[6.0].εv_p ≈ b[6.0].εv_p rtol = 1.0e-8
        @test b[6.0].εv_p ≈ 2.9176e-2 rtol = 1.0e-3
    end

    ## Bil's scheme is first order, and Bil's own error sits on that line.
    @test errs[1] / errs[2] ≈ 2 atol = 0.05
    @test errs[2] / errs[3] ≈ 2 atol = 0.05
    bil_err = abs(B.bbm_ref[1.0][1] - closed_form) / abs(closed_form)
    @test errs[3] < bil_err < errs[2]

    ## A closed elastic cycle must return the sample to where it started. An incremental
    ## hypoelastic law does not: it leaves a residual strain that is neither elastic nor
    ## plastic, only numerical. Integrating in closed form removes it exactly.
    function elastic_cycle(mat)
        st = initial_state(mat, B.σ0, B.pc_star0; suction = 0.0)
        for k in 1:200
            p_k = k <= 100 ? 1 + 19 * k / 100 : 20 - 19 * (k - 100) / 100
            _, _, st, _ = stress_controlled_response(mat, B.face_stress(p_k, 0.0), 0.0, st, 1.0)
        end
        return tr(st.ε)
    end
    @test abs(elastic_cycle(BBM())) < 1.0e-14
    @test abs(elastic_cycle(ExplicitPredictor(BBM()))) > 1.0e-3
end
