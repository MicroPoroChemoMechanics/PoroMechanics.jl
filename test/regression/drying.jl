using ForwardDiff
using PoroMechanics

@testset "Drying — near-saturated storage and completed transient" begin
    mod = _NonisothermalDrying
    m = mod.m
    initial = [m.p_l_ini2, m.p_a_ini2, m.T_ini]
    pc = mod._p_vapor(m, initial[1], initial[3]) + initial[2] - initial[1]
    @test pc < 0
    storage = function (u)
        f = similar(u)
        PoroMechanics.storage!(f, u, (; region = 2), m, nothing)
        return f
    end
    jac = ForwardDiff.jacobian(storage, initial)
    @test all(isfinite, jac)
    @test jac[mod.U_PL, mod.U_PL] > 0
    @test jac[mod.U_PL, mod.U_PA] < 0
    @test mod._compute_dUsdT(m, m.mat2, pc, m.T_ini - m.T_0) != 0

    @test first.(mod.results) == 3.1536e7 .* [1, 2, 4, 6, 8, 10, 20, 40, 50, 100]
    @test maximum(u[mod.U_TEM, 1] for (_, u) in mod.results) > m.T_ini + 1
    for (_, u) in mod.results
        @test all(isfinite, u)
        @test minimum(u[mod.U_PA, :]) > 0
        @test minimum(u[mod.U_TEM, :]) > 0
        @test maximum(u[mod.U_TEM, :]) < m.T_0 + 1 / m.alpha_T
        @test u[:, end] ≈ initial
        for region in (1, 2)
            nodes = findall(r -> region == 1 ? r <= m.r_int : r >= m.r_int, mod.r_all)
            @test all(i -> 0 < mod.liquid_saturation(m, u, i, region) <= 1, nodes)
        end
    end

    ## Durations below the default output schedule must not integrate past their end.
    _, _, short = mod.run_drying(; n_years = 1.0e-6)
    @test length(short) == 1
    @test short[end][1] == 1.0e-6 * 3.1536e7
end
