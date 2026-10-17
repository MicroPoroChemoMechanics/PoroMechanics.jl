# PoroplastModel against closed forms.
#
# The cross-code comparison in `test/bil.jl` is the only other test of this model, and it
# needs Bil's meshes, which are not vendored: on CI it is skipped, and the model would run
# nowhere. These cases need nothing but the package. Each has an answer known
# independently of the code — the initial state must not move, a thick elastic cylinder
# drained to steady state is Lamé's, and steady radial flow is logarithmic — and the
# plastic case checks what a plastic answer must satisfy rather than a number.

using Tensors: SymmetricTensor

@testset "PoroplastModel" begin

    ## The material of `base/Poroplast`, with a cohesion argument: a cohesion of 1e12 Pa
    ## keeps the skeleton elastic for every load below, which is what the closed forms need.
    E, ν = 5.8e9, 0.3
    φ = deg2rad(25)
    model(c) = PoroplastModel(;
        material = BiotPlastic(;
            skeleton = DruckerPrager(; E = E, nu = ν, cohesion = c, friction = φ, dilatancy = φ),
            b = 0.8, beta = 0.8, N = 0.0, k = 1.0e-19, mu_l = 1.0e-3,
        ),
        phi0 = 0.15, rho_l0 = 1.0e3, k_l = 2.0e9, p_l0 = 4.7e6,
    )
    elastic = model(1.0e12)

    ## A cavity of 3 m in a ground of 30 m, graded towards the wall where the gradients are.
    a, b, ne = 3.0, 30.0, 200
    nodes = collect(a .* (b / a) .^ range(0, 1; length = ne + 1))
    s0 = -11.5e6                                   # total, isotropic, compression negative
    I3 = one(SymmetricTensor{2, 3})
    p0 = elastic.p_l0

    states = poroplast_initial_states(elastic, nodes, s0 * I3)
    dofs0 = zeros(2 * length(nodes))
    dofs0[2:2:end] .= p0

    ## A step this long makes the storage term vanish: the answer is the drained,
    ## steady one.
    Δt_steady = 1.0e18

    @testset "the model interface" begin
        @test PoroMechanics.nspecies(elastic) == 2
        @test PoroMechanics.species_names(elastic) == [:u, :p_l]
        @test fluid_density(elastic, p0) == elastic.rho_l0
        @test fluid_density(elastic, p0 + elastic.k_l) ≈ 2 * elastic.rho_l0
        @test PoroMechanics.mobility(elastic, p0) ≈ 1.0e3 * 1.0e-19 / 1.0e-3

        ## The hoop strain is the displacement over the radius, not a derivative.
        ε = axisymmetric_strain_1d(1.0e-3, 2.0e-3, 4.0)
        @test (ε[1, 1], ε[2, 2], ε[3, 3]) == (1.0e-3, 5.0e-4, 0.0)
        @test ε[1, 2] == 0

        ## The skeleton starts from the effective stress σ₀ + β p₀, not the total one.
        @test states[1][1].solid.σ ≈ (s0 + 0.8 * p0) * I3
        @test states[1][1].p == p0
    end

    @testset "at the initial state nothing moves" begin
        ## The deck's own stresses and pressures, imposed and stepped once. A prestress
        ## handed over as total rather than effective fails this by β p₀ = 3.76 MPa.
        dofs, _ = poroplast_step!(
            elastic, nodes, states, dofs0, 1.0e4;
            σ_inner = s0, σ_outer = s0, p_inner = p0, p_outer = p0,
        )
        @test maximum(abs, dofs[1:2:end]) < 1.0e-12
        @test maximum(abs, dofs[2:2:end] .- p0) < 1.0e-3     # Pa, against 4.7 MPa
    end

    @testset "drained elastic cylinder — Lamé" begin
        ## Unloading the wall from s0 to σa, the outer face held. In plane strain the
        ## increment is u = A r + B / r with σ_rr = 2(λ + μ) A − 2μ B / r².
        σa = -2.0e6
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        μ = E / (2 * (1 + ν))
        Δa, Δb = σa - s0, 0.0
        B = (Δb - Δa) / (2μ * (1 / a^2 - 1 / b^2))
        A = (Δa + 2μ * B / a^2) / (2 * (λ + μ))
        u_exact = A .* nodes .+ B ./ nodes

        dofs, _ = poroplast_step!(
            elastic, nodes, states, dofs0, Δt_steady;
            σ_inner = σa, σ_outer = s0, p_inner = p0, p_outer = p0,
        )
        u = dofs[1:2:end]

        ## Measured 3.9e-5 on 200 linear elements.
        @test maximum(abs, u .- u_exact) / maximum(abs, u_exact) < 1.0e-4
        @test u[1] < 0                                 # released, the wall converges inward
        @test maximum(abs, dofs[2:2:end] .- p0) < 1.0     # drained: the pressure is back at p₀
    end

    @testset "steady radial flow — logarithmic pressure" begin
        ## Pressure drawn down at the wall, stresses held. The mobility is evaluated at the
        ## previous, uniform pressure, so it is constant and the profile is exactly the
        ## logarithm of the radius.
        pa = 1.0e6
        dofs, _ = poroplast_step!(
            elastic, nodes, states, dofs0, Δt_steady;
            σ_inner = s0, σ_outer = s0, p_inner = pa, p_outer = p0,
        )
        p = dofs[2:2:end]
        p_exact = pa .+ (p0 - pa) .* log.(nodes ./ a) ./ log(b / a)

        ## Measured 5.4e-11.
        @test maximum(abs, p .- p_exact) / (p0 - pa) < 1.0e-8
        @test (p[1], p[end]) == (pa, p0)               # the Dirichlet rows are exact
    end

    @testset "a plastic zone around the cavity" begin
        ## The deck's cohesion, 1 MPa. Released to the same wall stress, the ground yields
        ## near the cavity and stays elastic far from it.
        plastic = model(1.0e6)
        σa = -2.0e6
        states_p = poroplast_initial_states(plastic, nodes, s0 * I3)
        dofs_e, _ = poroplast_step!(
            elastic, nodes, states, dofs0, Δt_steady;
            σ_inner = σa, σ_outer = s0, p_inner = p0, p_outer = p0,
        )
        dofs_p, states_new = poroplast_step!(
            plastic, nodes, states_p, dofs0, Δt_steady;
            σ_inner = σa, σ_outer = s0, p_inner = p0, p_outer = p0,
        )

        γ = [s[1].solid.γp for s in states_new]
        @test γ[1] > 0                                 # yielded at the wall
        @test γ[end] == 0                              # elastic far away
        @test issorted(γ; rev = true)                  # and less the farther out

        ## Yielding softens the response: the wall moves more than it would elastically.
        @test dofs_p[1] < dofs_e[1] < 0

        ## Every converged stress is on or inside the cone.
        dp = plastic.material.skeleton
        cc = cohesion_intercept(dp)
        @test all(states_new) do es
            all(s -> yield_function(dp, s.solid.σ) <= 1.0e-6 * cc, es)
        end
    end

    @testset "a Newton that does not converge says so" begin
        @test_throws ErrorException poroplast_step!(
            model(1.0e6), nodes, poroplast_initial_states(model(1.0e6), nodes, s0 * I3),
            dofs0, Δt_steady;
            σ_inner = -2.0e6, σ_outer = s0, p_inner = p0, p_outer = p0, maxiter = 1,
        )
    end
end
