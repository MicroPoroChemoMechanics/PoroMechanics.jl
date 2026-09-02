# The transport models the package ships: `FickModel` and `DarcyModel`.
#
# Their profiles are already checked, end to end, by the regression harness — the examples
# now build them from the package instead of defining their own. What is checked here is
# what the harness cannot see, because the examples exercise only the plain case: the
# per-region coefficients, the boundary values that are functions of time, and a `Dual`
# living in a *parameter* rather than in an unknown.

using ForwardDiff: ForwardDiff
using ExtendableGrids: simplexgrid, cellmask!, Coordinates
using VoronoiFVM: VoronoiFVM, unknowns, solve

@testset "shipped transport models" begin

    @testset "the model interface" begin
        fick = FickModel()
        darcy = DarcyModel()

        @test PoroMechanics.nspecies(fick) == 1
        @test PoroMechanics.species_names(fick) == [:c]
        @test PoroMechanics.nspecies(darcy) == 1
        @test PoroMechanics.species_names(darcy) == [:p]

        ## Promotion, so that one `Dual` parameter does not force the others to be duals.
        @test FickModel(0.3, 1.0e-10, ()) isa FickModel{Float64, Float64}
        @test DarcyModel(1.0e-12, 1.0e-3, 1.0e-8, ()) isa DarcyModel{Float64, Float64, Float64}
        @test FickModel(; phi = 0.3, D = 1) isa FickModel{Float64, Float64}
    end

    @testset "coefficients per cell region" begin
        ## A scalar and a collection are the same model, told apart by dispatch.
        uniform = FickModel(; phi = 0.3, D = 1.0e-10)
        layered = FickModel(; phi = [0.3, 0.12], D = [1.0e-10, 2.0e-12])

        @test PoroMechanics.porosity(uniform, 1) == 0.3
        @test PoroMechanics.porosity(uniform, 2) == 0.3      # a scalar ignores the region
        @test PoroMechanics.porosity(layered, 2) == 0.12
        @test PoroMechanics.diffusivity(layered, 2) == 2.0e-12

        column = DarcyModel(; k_int = [1.0e-12, 1.0e-14], mu_l = 1.0e-3, storativity = 1.0e-8)
        @test PoroMechanics.intrinsic_permeability(column, 2) == 1.0e-14
        @test PoroMechanics.mobility(column, 2) == 1.0e-14 / 1.0e-3
        @test PoroMechanics.storativity(column, 1) == 1.0e-8
    end

    @testset "boundary values that depend on time" begin
        ## A number is imposed as it stands, anything else is called with the current time.
        @test PoroMechanics.dirichlet_value(1.5e5, 3.0) == 1.5e5
        @test PoroMechanics.dirichlet_value(t -> 2.0e4 * t, 3.0) == 6.0e4

        ## The ramp of the Darcy column, solved on a coarse grid: the imposed pressure must
        ## still be climbing at t = t_c/2 and have arrived at 2 t_c. Reading it off the
        ## solution rather than off the closure is what checks that the value reaches
        ## `bcondition!` through `apply_dirichlet!` at the right time.
        p_top, t_c = 1.0e5, 10.0
        m = DarcyModel(;
            k_int = 1.0e-12, mu_l = 1.0e-3, storativity = 1.0e-8,
            dirichlet = ((1, 0.0), (2, t -> p_top * min(1.0, t / t_c))),
        )
        grid = simplexgrid(range(0.0, 1.0; length = 21))
        sys = PoroMechanics.fvm_system(m, grid)
        ctrl = VoronoiFVM.SolverControl(; Δt = 0.25, Δt_min = 0.25, Δt_max = 5.0, Δu_opt = 1.0e5)
        tsol = solve(sys; inival = unknowns(sys; inival = 0.0), times = (0.0, 100.0), control = ctrl)

        @test tsol(t_c / 2)[1, end] ≈ p_top / 2 rtol = 1.0e-8      # still climbing
        @test tsol(2 * t_c)[1, end] ≈ p_top rtol = 1.0e-8          # arrived, and held there

        ## Once the ramp is over the column drains to the steady linear profile.
        x = grid[Coordinates][1, :]
        @test maximum(abs.(tsol(100.0)[1, :] .- p_top .* x)) < 0.001 * p_top
    end

    @testset "a layered barrier is a case, not a second model" begin
        ## Two regions, the second a hundred times less diffusive. The same struct covers
        ## it, and the front must not have crossed into the tight layer.
        grid = simplexgrid(range(0.0, 1.0; length = 51))
        cellmask!(grid, [0.5], [1.0], 2)

        m = FickModel(; phi = [0.3, 0.3], D = [1.0e-9, 1.0e-11], dirichlet = ((1, 1.0),))
        sys = PoroMechanics.fvm_system(m, grid)
        inival = unknowns(sys; inival = 0.0)
        inival[1, 1] = 1.0
        ctrl = VoronoiFVM.SolverControl(; Δt = 1.0e3, Δt_max = 1.0e6, Δu_opt = 0.1)
        tsol = solve(sys; inival, times = (0.0, 1.0e7), control = ctrl)

        c = tsol[1, :, end]
        @test all(diff(c) .<= 1.0e-12)          # monotone from the fed face inward
        @test c[1] ≈ 1.0
        @test c[end] < 1.0e-3                   # the tight layer has held the front back
    end

    @testset "a Dual in a parameter" begin
        ## The distinguishing claim, on these two models: the coefficients are type
        ## parameters, so a flux differentiates with respect to `D` and not only with
        ## respect to the concentrations. Both derivatives are known in closed form here,
        ## which is stronger than a finite difference.
        u = [2.0 0.5]      # one species, two nodes

        dfdD = ForwardDiff.derivative(1.0e-10) do D
            m = FickModel(; phi = 0.3, D = D)
            g = zeros(typeof(D), 1)
            PoroMechanics.flux!(g, u, (region = 1,), m, nothing)
            return g[1]
        end
        @test dfdD ≈ 0.3 * (u[1] - u[2])

        dmob = ForwardDiff.derivative(1.0e-12) do k
            PoroMechanics.mobility(DarcyModel(; k_int = k, mu_l = 1.0e-3), 1)
        end
        @test dmob ≈ 1 / 1.0e-3

        ## A dual in the *boundary value* survives too, which is what an identification of
        ## an imposed concentration from a measured profile needs.
        @test ForwardDiff.derivative(c -> PoroMechanics.dirichlet_value(c, 1.0), 2.0) == 1
    end

end
