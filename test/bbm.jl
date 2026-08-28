# The Barcelona Basic Model at a material point.
#
# The tests are ordered from the surface inwards: the LC curve, the yield surface, then the
# stress paths. The one that matters most is collapse on wetting — a BBM that does not
# reproduce it has missed the reason it exists.

using Tensors
using ForwardDiff
using LinearAlgebra: norm
import Ferrite

const I3 = one(SymmetricTensor{2, 3})

"""Drive an isotropic compression path, returning the state history."""
function isotropic_path(m, pc0, s; nstep = 20, dεv = 0.004)
    st = initial_state(m, -1.0e3 * I3, pc0; suction = s)
    hist = [(0.0, mean_pressure(st.σ), st.pc_star, st.εv_p)]
    εv = 0.0
    for _ in 1:nstep
        εv += dεv
        σ, _, st = material_response(m, -(εv / 3) * I3, st, 1.0)
        push!(hist, (εv, mean_pressure(σ), st.pc_star, st.εv_p))
    end
    return st, hist
end

"""Isotropic step driven to a target mean pressure rather than a target strain."""
function stress_step(m, st, p_target; nit = 60)
    εv = -tr(st.ε)
    local σ, st_new
    for _ in 1:nit
        σ, C, st_new = material_response(m, -(εv / 3) * I3, st, 1.0)
        abs(mean_pressure(σ) - p_target) < 1.0 && break
        K = ((C ⊡ I3) ⊡ I3) / 9
        εv += (p_target - mean_pressure(σ)) / max(K, 1.0e3)
    end
    return st_new, σ
end

@testset "BBM — the LC curve" begin
    m = BBM()

    ## λ falls with suction, towards r·λ(0)
    @test compression_index(m, 0.0) ≈ m.λ0
    @test compression_index(m, 1.0e6) < compression_index(m, 1.0e5) < m.λ0
    @test compression_index(m, 1.0e9) ≈ m.r * m.λ0 rtol = 1.0e-6

    ## At zero suction the preconsolidation is the hardening variable itself
    @test preconsolidation(m, 0.0, 4.0e4) ≈ 4.0e4

    ## Drying expands the elastic domain — the whole point of the model
    @test preconsolidation(m, 2.0e5, 4.0e4) > preconsolidation(m, 0.0, 4.0e4)
    p0s = [preconsolidation(m, s, 4.0e4) for s in (0.0, 5.0e4, 1.0e5, 2.0e5)]
    @test issorted(p0s)

    ## Differentiable in the parameters one would calibrate
    for (f, x) in (
            (κ -> preconsolidation(BBM(; κ = κ), 1.0e5, 4.0e4), 0.011),
            (λ0 -> preconsolidation(BBM(; λ0 = λ0), 1.0e5, 4.0e4), 0.065),
            (r -> preconsolidation(BBM(; r = r), 1.0e5, 4.0e4), 0.75),
        )
        @test ForwardDiff.derivative(f, x) != 0
        h = 1.0e-7 * x
        @test isapprox(ForwardDiff.derivative(f, x), (f(x + h) - f(x - h)) / 2h; rtol = 1.0e-5)
    end
end

@testset "BBM — the yield surface" begin
    m = BBM()
    pc = 4.0e4

    ## Zero on the surface, negative inside
    @test yield_function(m, preconsolidation(m, 0.0, pc), 0.0, 0.0, pc) ≈ 0.0 atol = 1.0e-6
    @test yield_function(m, pc / 2, 0.0, 0.0, pc) < 0

    ## Suction shifts the tensile intercept to −k_s s: apparent cohesion.
    s = 1.0e5
    @test yield_function(m, -m.k_s * s, 0.0, s, pc) ≈ 0.0 atol = 1.0e-6
    ## ...which is zero when saturated, so the surface then passes through the origin.
    @test yield_function(m, 0.0, 0.0, 0.0, pc) ≈ 0.0 atol = 1.0e-6

    ## The derivatives used by the flow rule, against automatic differentiation
    for p in (1.0e4, 2.0e4, 3.5e4)
        @test PoroMechanics.dyield_dp(m, p, 0.0, pc) ≈
            ForwardDiff.derivative(x -> yield_function(m, x, 0.0, 0.0, pc), p)
    end
    @test PoroMechanics.dyield_dq(m, 1.0e4) ≈
        ForwardDiff.derivative(x -> yield_function(m, 2.0e4, x, 0.0, pc), 1.0e4)
end

@testset "BBM — isotropic compression" begin
    m = BBM()
    st, hist = isotropic_path(m, 4.0e4, 0.0)

    εv = [h[1] for h in hist]
    p = [h[2] for h in hist]
    pc = [h[3] for h in hist]
    evp = [h[4] for h in hist]

    ## Elastic while p < p₀, and stiffening as it goes: K ∝ p
    @test all(evp[p .< 3.9e4] .< 1.0e-12)
    @test issorted(p)

    ## Yield begins at the preconsolidation pressure, not before or after
    first_plastic = findfirst(>(1.0e-12), evp)
    @test p[first_plastic - 1] < 4.05e4
    @test p[first_plastic] > 4.0e4

    ## On the normal compression line the stress point rides the surface: p ≈ p₀ = pc*
    @test isapprox(p[end], pc[end]; rtol = 0.05)

    ## Hardening is monotone, and plastic strain accumulates
    @test issorted(pc)
    @test issorted(evp)
    @test pc[end] > 2 * pc[1]

    ## Unloading from the plastic state is elastic: no further plastic strain
    εv_end = -tr(st.ε)
    σ_un, _, st_un = material_response(m, -((εv_end - 0.002) / 3) * I3, st, 1.0)
    @test st_un.εv_p ≈ st.εv_p
    @test st_un.pc_star ≈ st.pc_star
    @test mean_pressure(σ_un) < mean_pressure(st.σ)
end

@testset "BBM — collapse on wetting" begin
    ## The signature phenomenon. A sample is dried, loaded to a stress that is safely
    ## elastic *because* it is dry, then wetted at constant stress. The yield surface
    ## shrinks past the stress point and the soil compresses — with nothing applied to it.
    m = BBM()
    s_dry = 2.0e5

    st = initial_state(m, -1.0e3 * I3, 4.0e4; suction = s_dry)
    for pt in (1.0e4, 3.0e4, 5.0e4)
        st, _ = stress_step(m, st, pt)
    end
    p_hold = mean_pressure(st.σ)

    ## Still elastic at 50 kPa, which it would not be if it were saturated
    @test st.εv_p ≈ 0.0 atol = 1.0e-12
    @test yield_function(m, p_hold, 0.0, s_dry, st.pc_star) < 0
    @test yield_function(m, p_hold, 0.0, 0.0, st.pc_star) > 0     # ...but wet, it is outside

    ## Wet it, holding the stress
    st_wet = BBMState(st.σ, st.ε, st.pc_star, st.εv_p, 0.0)
    εv_before = -tr(st_wet.ε)
    st_after, σ_after = stress_step(m, st_wet, p_hold)

    @test isapprox(mean_pressure(σ_after), p_hold; rtol = 1.0e-3)   # stress unchanged
    @test -tr(st_after.ε) > εv_before                                # yet it compressed
    @test st_after.εv_p > st.εv_p                                    # and plastically
    @test st_after.εv_p > 1.0e-3                                     # appreciably so
end

@testset "BBM — the tangent" begin
    m = BBM()

    ## On an elastic step the tangent is exactly the elastic one, and automatic
    ## differentiation agrees because no return map is involved.
    st = initial_state(m, -1.0e4 * I3, 4.0e4)
    ε_el = -(0.001 / 3) * I3
    σ, C, st2 = material_response(m, ε_el, st, 1.0)
    @test st2.εv_p ≈ 0.0 atol = 1.0e-14
    @test Tensors.gradient(e -> material_response(m, e, st, 1.0)[1], ε_el) ≈ C

    ## Machinery for comparing against the true Jacobian of a plastic step.
    stp, _ = isotropic_path(m, 4.0e4, 0.0; nstep = 13)
    idx = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    to_mat(C4) = [C4[i, j, k, l] for (i, j) in idx, (k, l) in idx]
    function fd_tangent(ε0, state; h = 1.0e-9)
        M = zeros(6, 6)
        for (col, (k, l)) in enumerate(idx)
            d = SymmetricTensor{2, 3}((i, j) -> ((i, j) == (k, l) || (i, j) == (l, k)) ? h : 0.0)
            Δ = (material_response(m, ε0 + d, state, 1.0)[1] -
                material_response(m, ε0 - d, state, 1.0)[1]) / 2h * (k == l ? 1.0 : 0.5)
            for (row, (i, j)) in enumerate(idx)
                M[row, col] = Δ[i, j]
            end
        end
        return M
    end

    ## **The algorithmic tangent is exact**, at every step size — it is the Jacobian of the
    ## discrete return map, not an approximation to it. What is left is finite-difference
    ## noise, which is why the tolerance is 1e-6 and not 1e-12.
    for dε in (4.0e-3, 1.0e-3, 2.5e-4)
        ε1 = -((-tr(stp.ε)) + dε) / 3 * I3
        _, Cp, _ = material_response(m, ε1, stp, 1.0)
        @test norm(fd_tangent(ε1, stp) - to_mat(Cp)) / norm(to_mat(Cp)) < 1.0e-6
    end

    ## The continuum tangent, kept for comparison, is a different object: correct only in
    ## the limit, with a gap that closes as the step shrinks. Asserting both makes the
    ## distinction between them a property of the code rather than a claim in a comment.
    errs = map((4.0e-3, 1.0e-3, 2.5e-4)) do dε
        ε1 = -((-tr(stp.ε)) + dε) / 3 * I3
        σ1, _, _ = material_response(m, ε1, stp, 1.0)
        p_n = max(mean_pressure(stp.σ), m.p_min)
        K, G = bbm_moduli(m, p_n)
        σ_tr = PoroMechanics.trial_stress(m, ε1, 0.0, stp, Val(:exact))
        C_tr = Tensors.gradient(e -> PoroMechanics.trial_stress(m, e, 0.0, stp, Val(:exact)), ε1)
        p_tr, q_tr = mean_pressure(σ_tr), equivalent_stress(σ_tr)
        pp, Δγ, qq, pcs, _, _ = PoroMechanics.solve_return_map(
            m, p_tr, q_tr, 0.0, stp.pc_star, p_n
        )
        Cc = elastoplastic_tangent(
            m, C_tr, pp, qq, 0.0, pcs, Δγ, K, G, Tensors.dev(σ_tr), q_tr, Val(3)
        )
        norm(fd_tangent(ε1, stp) - to_mat(Cc)) / norm(to_mat(Cc))
    end
    @test all(errs .> 1.0e-3)             # genuinely different from the algorithmic one
    @test issorted(errs; rev = true)      # and converging to it as the step shrinks
end

@testset "Newton global axisymétrique" begin
    ## A uniform problem has an answer that is known independently: every quadrature point
    ## sees the same strain, so the finite element stress must be the material point's. That
    ## is what verifies the assembly, the Gauss-point state bookkeeping and the solver at
    ## once, without needing an analytical solution for the BBM — which does not exist.
    grid = Ferrite.generate_grid(
        Ferrite.Quadrilateral, (2, 2), Ferrite.Vec(1.0, 0.0), Ferrite.Vec(2.0, 1.0)
    )
    ip = Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}()^2
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)
    qr = Ferrite.QuadratureRule{Ferrite.RefQuadrilateral}(2)
    cv = Ferrite.CellValues(qr, ip, Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}())
    nq = Ferrite.getnquadpoints(cv)

    I3 = one(SymmetricTensor{2, 3})
    σ0, pc0, εv_tot, nsteps = -1.0e3 * I3, 4.0e4, 0.08, 16

    function drive(mat)
        mk() = [[initial_state(mat, σ0, pc0) for _ in 1:nq] for _ in 1:Ferrite.getncells(grid)]
        states, old = mk(), mk()
        K = Ferrite.allocate_matrix(dh)
        f = zeros(Ferrite.ndofs(dh))
        u = zeros(Ferrite.ndofs(dh))
        history = Vector{Vector{Float64}}()
        for k in 1:nsteps
            εv = εv_tot * k / nsteps
            ch = Ferrite.ConstraintHandler(dh)
            for (set, comp) in (("left", 1), ("right", 1), ("bottom", 2), ("top", 2))
                Ferrite.add!(
                    ch, Ferrite.Dirichlet(
                        :u, Ferrite.getfacetset(grid, set), (x, t) -> -εv / 3 * x[comp], [comp]
                    )
                )
            end
            Ferrite.close!(ch)
            Ferrite.update!(ch, 0.0)
            push!(history, newton_solve!(u, K, f, dh, cv, mat, states, old, ch, 1.0))
            for c in eachindex(states)
                old[c] .= states[c]
            end
        end
        return old[1][1], history
    end

    ## The same path at the material point, as the reference.
    ref = initial_state(BBM(), σ0, pc0)
    for k in 1:nsteps
        ref = material_response(BBM(), -(εv_tot * k / nsteps / 3) * I3, ref, 1.0)[3]
    end

    st_fe, hist = drive(BBM())
    @test mean_pressure(st_fe.σ) ≈ mean_pressure(ref.σ) rtol = 1.0e-10
    @test st_fe.pc_star ≈ ref.pc_star rtol = 1.0e-10
    @test st_fe.εv_p ≈ ref.εv_p rtol = 1.0e-10
    @test ref.εv_p > 0.04                      # the path is genuinely plastic

    ## Quadratic convergence on the plastic steps that follow the elastic–plastic
    ## transition. Squaring the residual roughly doubles the number of correct digits, so
    ## three successive norms must satisfy e₃ ≲ e₂²/e₁ up to a constant.
    for nr in hist[(end - 1):end]
        @test length(nr) <= 6
        e1, e2, e3 = nr[(end - 2):end]
        @test e3 < 10 * e2^2 / e1
    end

    ## Every step converges, and the elastic ones take a single iteration because the
    ## imposed field is linear and therefore exactly representable.
    @test all(nr -> last(nr) < 1.0e-8, hist)
    ## The elastic steps converge in five iterations rather than in one: with the elastic
    ## law integrated in closed form the pressure is exponential in the strain, so even an
    ## elastic step is a nonlinear solve. It is a *smooth* one, which is why the count stays
    ## small and the tail is quadratic — as it is on the plastic steps, and for the same
    ## reason, the tangent being exact in both cases.
    @test all(nr -> 4 <= length(nr) <= 7, hist)
    for nr in hist[1:6]
        e1, e2, e3 = nr[(end - 2):end]
        @test e3 < 10 * e2^2 / e1
    end

    ## The continuum tangent reaches the same answer, far more slowly. This is the
    ## measurement that justifies deriving the algorithmic one.
    st_c, hist_c = drive(ContinuumTangent(BBM()))
    @test mean_pressure(st_c.σ) ≈ mean_pressure(ref.σ) rtol = 1.0e-5
    @test sum(length, hist_c) > 3 * sum(length, hist)
end
