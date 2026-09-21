using ForwardDiff
using ExtendableGrids: simplexgrid
using VoronoiFVM: VoronoiFVM

@testset "Drying model" begin
    clay = DryingMaterial(
        0.3, 1.0e-20, 1.12, 2.3e6,
        ExponentialCutoff(VanGenuchten(1.5e6, 1.06383, 0.06), 1.0e6),
        PowerLawKrl(3.0e6, 2.0, 0.5)
    )
    rock = DryingMaterial(
        0.05, 1.0e-19, 1.62, 2.0e6,
        ExponentialCutoff(VanGenuchten(10.0e6, 1.7, 0.4117), 2.0e5),
        PowerLawKrl(10.0e6, 2.0, 1.0)
    )
    m = DryingModel(; materials = (clay, rock, clay))
    p = m.parameters
    state = [-2.0e6, 1.1e5, 310.0]
    edge = hcat(state, [-1.0e6, 1.0e5, 300.0])
    stored = function (u, region = 1, model = m)
        f = similar(u)
        storage!(f, u, (; region), model, nothing)
        return f
    end
    transported = function (u, region = 1, model = m)
        f = similar(u, 3)
        flux!(f, reshape(u, 3, 2), (; region), model, nothing)
        return f
    end

    @testset "constitutive laws and region selection" begin
        @test PoroMechanics.nspecies(m) == 3
        @test PoroMechanics.species_names(m) == [:p_l, :p_a, :T]
        @test vapor_pressure(p, p.p_l0, p.T_0) == p.p_v0
        @test ForwardDiff.derivative(pl -> vapor_pressure(p, pl, p.T_0), p.p_l0) ≈
            p.p_v0 * p.M_vsR / (p.T_0 * p.rho_l)
        @test drying_material(m, 3) === clay
        @test_throws BoundsError drying_material(m, 4)
        mapped = DryingModel(; materials = Dict(7 => clay, 12 => rock))
        @test stored(state, 12, mapped) == stored(state, 2)
        @test stored(state, 3) == stored(state, 1)
        @test liquid_saturation(m, edge, 1, 1) == liquid_saturation(m, state, 1)
        @test liquid_saturation(m, state, 1) != liquid_saturation(m, state, 2)
        @test saturation(clay, 1.0e6) == saturation(clay.retention, 1.0e6)
        @test relative_permeability(clay, 1.0e6) == relative_permeability(clay.rel_perm, 1.0e6)
    end

    @testset "callback values and state derivatives" begin
        # Values captured from the original example before extracting its equations.
        storage_reference = (
            [283.78365784169057, 0.024910741870558705, 196672.3821419778],
            [48.47732407152593, 0.002723724633799327, 124234.86863126868],
        )
        flux_reference = (
            [-8.482873760157688e-9, -2.5464883468869334e-10, 0.029212413371863914],
            [-9.728519462401688e-8, -2.9678007372273805e-12, 0.05037452039507373],
        )
        for region in 1:2
            @test all(isapprox.(stored(state, region), storage_reference[region]; rtol = 1.0e-12))
            @test all(isapprox.(transported(edge, region), flux_reference[region]; rtol = 1.0e-12))
            @test transported(edge[:, [2, 1]], region) ≈ -transported(edge, region)
            @test transported(hcat(state, state), region) == zeros(3)
            # Scale pressures and temperature to resolve each column by finite differences.
            for (fun, x, scales) in (
                    (u -> stored(u, region), state, [1.0e6, 1.0e5, 300.0]),
                    (u -> transported(u, region), vec(edge), repeat([1.0e6, 1.0e5, 300.0], 2)),
                )
                scaled = z -> fun(z .* scales)
                z = x ./ scales
                jac = ForwardDiff.jacobian(scaled, z)
                for j in eachindex(z)
                    step = zeros(length(z)); step[j] = 1.0e-5
                    finite_difference = (scaled(z + step) - scaled(z - step)) / 2.0e-5
                    @test jac[:, j] ≈ finite_difference rtol = 2.0e-6
                end
            end
        end
    end

    @testset "parameter differentiation" begin
        @test ForwardDiff.derivative(v -> vapor_pressure(DryingParameters(; p_v0 = v), p.p_l0, p.T_0), p.p_v0) == 1
        mass_flux = function (scale)
            mat = DryingMaterial(;
                phi = clay.phi, k_int = scale * clay.k_int,
                lam_s = clay.lam_s, C_s = clay.C_s, retention = clay.retention, rel_perm = clay.rel_perm
            )
            model = DryingModel(; materials = (mat,))
            f = zeros(typeof(scale), 3)
            flux!(f, edge, (; region = 1), model, nothing)
            return f[1]
        end
        @test ForwardDiff.derivative(mass_flux, 1.0) ≈ (mass_flux(1.001) - mass_flux(0.999)) / 0.002 rtol = 1.0e-8
    end

    @testset "boundary data and uniform equilibrium" begin
        grid = simplexgrid([0.0, 0.5, 1.0])
        sys = fvm_system(m, grid)
        node = VoronoiFVM.BNode(sys)
        node.region = 7
        node.time = 2.0
        heated = DryingModel(; materials = (clay,), heat_flux = ((7, t -> 3 * t),))
        f = zeros(3)
        bcondition!(f, state, node, heated, nothing)
        @test f ≈ [0, 0, -6 / state[3]]
        node.region = 8
        fill!(f, 0)
        bcondition!(f, state, node, heated, nothing)
        @test f == zeros(3)
        fixed = DryingModel(;
            materials = (clay,),
            dirichlet = (((8, t -> state[1] + t),), ((8, state[2]),), ((8, state[3]),))
        )
        bcondition!(f, state, node, fixed, nothing)
        @test f[1] / node.Dirichlet ≈ -node.time
        @test f[2:3] == zeros(2)
        @test node.dirichlet_value == [state[1] + node.time, state[2], state[3]]
        initial = repeat(state, 1, 3)
        solution = VoronoiFVM.solve(sys; inival = initial, times = [0.0, 1.0])
        @test solution(1.0) ≈ initial rtol = 1.0e-12
    end
end
