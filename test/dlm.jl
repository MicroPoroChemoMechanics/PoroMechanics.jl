# Surface complexation on C-S-H — `examples/chloride_ingress/dlm.jl`.
#
# This file used to exist three times, in `run_4.jl`, `tran2018.jl` and
# `chloride_ternary.jl`. The values below were measured on those three implementations
# **before** they were merged, so they pin the merge: they are a regression guard on a
# refactor, not a validation of the physics, which belongs to the published references
# the examples carry.
#
# The derivatives are tested as hard as the values. They are what the coupling with
# transport actually consumes — `K_d = dS/dc` enters `storage!` — and a refactor that
# preserved the values while losing the dual type would pass a value-only test and
# silently return a diagonal of zeros.

using ForwardDiff

module _DLM
    include(joinpath(@__DIR__, "..", "examples", "chloride_ingress", "dlm.jl"))
end

@testset "DLM surface complexation" begin
    D = _DLM

    ## Seawater on jennite-poor C-S-H — state 4 of the pre-merge capture.
    st = (546.0, 459.0, 9.71, 9.97, 52.2, 50.0, 400.0, 0.83)

    @testset "values and derivatives, against the pre-merge implementations" begin
        ## `solve_dlm_marks` of tran2018.jl, reproduced bit for bit on 288 quantities.
        seeded = (ForwardDiff.Dual{Nothing}(st[1], 1.0), st[2:end]...)
        out = D.solve_dlm(seeded...; dlm = D.DLM_TRAN2018(), T_K = 293.15)
        @test ForwardDiff.value(out[1]) ≈ 1.0479221759151138 rtol = 1.0e-14
        @test ForwardDiff.value(out[2]) ≈ 1.5589822692215305 rtol = 1.0e-14
        @test ForwardDiff.partials(out[1], 1) ≈ -0.00021228066677052198 rtol = 1.0e-12
        @test ForwardDiff.partials(out[2], 1) ≈ 0.0022871631570032841 rtol = 1.0e-12

        ## `solve_dlm_ternary` of chloride_ternary.jl. The ternary complex binds an order
        ## of magnitude more chloride at the same concentration, which is the point of it.
        outt = D.solve_dlm(seeded...; dlm = D.DLM_TERNARY(), T_K = 293.15)
        @test ForwardDiff.value(outt[1]) ≈ 1.444952569397109 rtol = 1.0e-14
        @test ForwardDiff.value(outt[2]) ≈ 66.783509908347554 rtol = 1.0e-14
        @test ForwardDiff.partials(outt[1], 1) ≈ -0.00013162013849402472 rtol = 1.0e-12
        @test ForwardDiff.partials(outt[2], 1) ≈ 0.087718096496282427 rtol = 1.0e-12
    end

    @testset "the returned β is a root of the residual" begin
        ## The contract between the solver and the residual: whatever `solve_dlm` returns
        ## must annihilate `dlm_residual`. A globally implicit scheme uses the residual
        ## alone, so the two must agree or the schemes will not.
        for dlm in (D.DLM_TRAN2018(), D.DLM_TERNARY())
            β = D.solve_dlm(st...; dlm = dlm, T_K = 293.15)[1]
            r = D.dlm_residual(β, st[1], st[2], st[3], st[4], st[5], st[6], st[8];
                dlm = dlm, T_K = 293.15)
            scale = abs(D.dlm_residual(β + 1.0e-3, st[1], st[2], st[3], st[4], st[5],
                st[6], st[8]; dlm = dlm, T_K = 293.15) - r) / 1.0e-3
            @test abs(r) < 1.0e-8 * max(scale, 1.0)
        end
    end

    @testset "derivatives against central differences" begin
        ## The AD derivative is exact; the difference quotient is the independent check.
        for dlm in (D.DLM_TRAN2018(), D.DLM_TERNARY()), k in (1, 2, 4)
            seeded = ntuple(i -> i == k ? ForwardDiff.Dual{Nothing}(st[i], 1.0) : st[i], 8)
            S_ad = ForwardDiff.partials(
                D.solve_dlm(seeded...; dlm = dlm, T_K = 293.15)[2], 1
            )
            h = 1.0e-4 * st[k]
            up = ntuple(i -> i == k ? st[i] + h : st[i], 8)
            dn = ntuple(i -> i == k ? st[i] - h : st[i], 8)
            S_fd = (D.solve_dlm(up...; dlm = dlm, T_K = 293.15)[2] -
                D.solve_dlm(dn...; dlm = dlm, T_K = 293.15)[2]) / (2h)
            @test S_ad ≈ S_fd rtol = 1.0e-5
        end
    end

    @testset "a Dual may live in a parameter" begin
        ## The capability the three replaced implementations did not have: every one of
        ## them declared its coefficients `::Float64`, so a profile could be differentiated
        ## with respect to a concentration but never with respect to `K_Cl` or `Γ_max`.
        ## That is what an inverse identification of the surface constants needs.
        for (name, build) in (
                ("K_Cl", k -> D.DLM_TRAN2018(K_Cl = k)),
                ("Gamma_max", g -> D.DLM_TRAN2018(Gamma_max_Tob = g, Gamma_max_Jen = g)),
            )
            p0 = name == "K_Cl" ? 4.47e-4 : 1.3e-6
            f(p) = D.solve_dlm(st...; dlm = build(p), T_K = 293.15)[2]
            d_ad = ForwardDiff.derivative(f, p0)
            h = 1.0e-6 * p0
            d_fd = (f(p0 + h) - f(p0 - h)) / (2h)
            @test isfinite(d_ad)
            @test d_ad != 0                      # a lost dual would come back exactly zero
            @test d_ad ≈ d_fd rtol = 1.0e-5
        end
    end

    @testset "no C-S-H, no surface" begin
        ## `zero(T)` rather than `0.0` on the early branch: the guard must not strip the
        ## dual type, or the caller's `K_d` silently becomes zero instead of small.
        seeded = (ForwardDiff.Dual{Nothing}(st[1], 1.0), st[2:6]..., 0.0, st[8])
        out = D.solve_dlm(seeded...; dlm = D.DLM_TRAN2018(), T_K = 293.15)
        @test all(o -> ForwardDiff.value(o) == 0, out)
        @test eltype(out) <: ForwardDiff.Dual
    end

    @testset "the two mechanisms are not one constant apart" begin
        ## The ternary complex is neutral, so it occupies a site but carries no charge.
        ## That absence from the charge sum — not a different `K_Cl` — is the difference,
        ## and it is why the mechanism is a type rather than a number.
        cH = 1.0e-4
        args = (0.3, 546.0, 9.97, 52.2, cH)
        b_outer = D.dlm_charge_sum(args[1], args[2], args[3], args[4], args[5],
            D.DLM_TRAN2018())
        b_tern = D.dlm_charge_sum(args[1], args[2], args[3], args[4], args[5],
            D.DLM_TERNARY())
        @test b_outer != b_tern
        ## With no chloride at all the two charge sums coincide, since the only term that
        ## distinguishes them is the chloride one.
        @test D.dlm_charge_sum(args[1], 0.0, args[3], args[4], args[5], D.DLM_TRAN2018()) ≈
            D.dlm_charge_sum(args[1], 0.0, args[3], args[4], args[5], D.DLM_TERNARY())
    end
end
