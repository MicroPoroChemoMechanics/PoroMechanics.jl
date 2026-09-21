using ForwardDiff
using PoroMechanics

@testset "Drying — near-saturated storage and completed transient" begin
    mod = _NonisothermalDrying
    m = mod.m
    initial = collect(mod.case.initial.rock)
    p = m.parameters
    geometry = mod.case.geometry
    pc = vapor_pressure(m, initial[1], initial[3]) + initial[2] - initial[1]
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
    @test PoroMechanics._drying_capillary_integral(p, drying_material(m, 2), pc, initial[3] - p.T_0) != 0

    @test first.(mod.results) == 3.1536e7 .* [1, 2, 4, 6, 8, 10, 20, 40, 50, 100]
    @test maximum(u[mod.U_TEM, 1] for (_, u) in mod.results) > mod.case.initial.clay[3] + 1
    for (_, u) in mod.results
        @test all(isfinite, u)
        @test minimum(u[mod.U_PA, :]) > 0
        @test minimum(u[mod.U_TEM, :]) > 0
        @test maximum(u[mod.U_TEM, :]) < p.T_0 + 1 / p.alpha_T
        @test u[:, end] ≈ initial
        for region in (1, 2)
            nodes = findall(r -> region == 1 ? r <= geometry.r_int : r >= geometry.r_int, mod.r_all)
            @test all(i -> 0 < mod.liquid_saturation(m, u, i, region) <= 1, nodes)
        end
    end

    ## Durations below the default output schedule must not integrate past their end.
    _, _, short = mod.run_drying(; n_years = 1.0e-6)
    @test length(short) == 1
    @test short[end][1] == 1.0e-6 * 3.1536e7
end

@testset "Drying — configured case without heating" begin
    mod = _NonisothermalDrying
    state = (-2.0e6, 1.0e5, 300.0)
    case = mod.drying_case(;
        parameters = DryingParameters(; mu_l = 1.2e-3),
        initial = (; clay = state, rock = state),
        heat_flux = t -> 0.0, heat_times = Float64[],
    )
    model, radii, snapshots = mod.run_drying(; case, n_clay = 6, h_rock_max = 1.0, n_years = 1.0e-6)
    @test model === case.model
    @test model.parameters.mu_l == 1.2e-3
    @test last(snapshots)[2] ≈ repeat(collect(state), 1, length(radii)) rtol = 1.0e-12
    @test isnothing(
        redirect_stdout(devnull) do
            mod.print_summary(case, radii, snapshots)
        end
    )
end
