# Drucker-Prager: the criterion, the two returns, and differentiability.
#
# The material exists to make `base/Poroplast` comparable, so the first thing checked is
# that its coefficients are Bil's, character for character. The rest are properties that
# hold for any correct implementation and would catch the mistakes this model invites: a
# return that lands beside the cone rather than on it, a missing apex, and a `Dual` that
# does not survive the corrector.

using ForwardDiff: ForwardDiff
using LinearAlgebra: norm
using Tensors: Tensors, SymmetricTensor, ⊡, dev, tr

@testset "Drucker-Prager" begin

    ## The parameters of `base/Poroplast`.
    E, ν, c = 5.8e9, 0.3, 1.0e6
    φ = deg2rad(25)
    m = DruckerPrager(; E = E, nu = ν, cohesion = c, friction = φ, dilatancy = φ)
    I3 = one(SymmetricTensor{2, 3})

    @testset "coefficients agree with Bil's" begin
        ## `PlasticityDruckerPrager.c`:
        ##   ff = 6 sin(af) / (3 - sin(af))
        ##   dd = 6 sin(ad) / (3 - sin(ad))
        ##   cc = 6 cos(af) / (3 - sin(af)) * cohesion
        @test friction_coefficient(m) ≈ 6 * sin(φ) / (3 - sin(φ))
        @test dilatancy_coefficient(m) ≈ 6 * sin(φ) / (3 - sin(φ))
        @test cohesion_intercept(m) ≈ 6 * cos(φ) / (3 - sin(φ)) * c
        @test apex_pressure(m) ≈ cohesion_intercept(m) / friction_coefficient(m)

        ## A frictionless material is von Mises: the cone becomes a cylinder of radius cc.
        vm = DruckerPrager(; E = E, nu = ν, cohesion = c, friction = 0.0, dilatancy = 0.0)
        @test friction_coefficient(vm) == 0
        @test cohesion_intercept(vm) ≈ 2 * c
    end

    @testset "the deck's initial state is inside the cone" begin
        ## -11.5 MPa isotropic, as `base/Poroplast` starts.
        @test yield_function(m, -11.5e6 * I3) < 0
        ## And a purely spherical stress has no deviator, so f is exactly ff*p - cc.
        @test yield_function(m, -11.5e6 * I3) ≈
            friction_coefficient(m) * (-11.5e6) - cohesion_intercept(m)
    end

    "Confining compression with a shear strain of `γ` superimposed."
    shear_strain(γ) = SymmetricTensor{2, 3}(
        (i, j) -> i == j ? -1.0e-3 : (((i, j) == (1, 2) || (i, j) == (2, 1)) ? γ : 0.0)
    )

    @testset "elastic branch, then a return that lands on the cone" begin
        state = initial_state(m, zero(SymmetricTensor{2, 3}))
        yielded = false
        for k in 1:8
            ε = shear_strain(4.0e-4 * k)
            σ, C, state = material_response(m, ε, state, 1.0)
            f = yield_function(m, σ)

            ## Never outside the cone, to round-off.
            @test f <= 1.0e-6 * cohesion_intercept(m)

            if state.γp > 0
                yielded = true
                ## On the cone, not merely inside it: this is what a wrong return misses.
                @test isapprox(f, 0.0; atol = 1.0e-6 * cohesion_intercept(m))
            else
                ## While elastic the tangent is exactly the elastic stiffness.
                @test C ≈ elastic_stiffness(m.elastic, Val(3))
            end
        end
        @test yielded
        @test state.γp > 0
    end

    @testset "the tangent is the derivative of the return actually performed" begin
        state = initial_state(m, zero(SymmetricTensor{2, 3}))
        for k in 1:8
            ε = shear_strain(4.0e-4 * k)
            εp_before = state.εp
            _, C, state = material_response(m, ε, state, 1.0)
            C_ad = Tensors.gradient(e -> first(drucker_prager_return(m, e, εp_before)), ε)
            @test norm(C - C_ad) <= 1.0e-12 * norm(C_ad)
        end
    end

    @testset "apex return" begin
        ## Isotropic tension well past the tip: the whole deviator is shed and the stress
        ## sits exactly on the apex. Without an apex branch the smooth return would produce
        ## a negative q here.
        state = initial_state(m, zero(SymmetricTensor{2, 3}))
        σ, C, state = material_response(m, 1.0e-2 * I3, state, 1.0)
        @test tr(σ) / 3 ≈ apex_pressure(m)
        @test sqrt(3 * (dev(σ) ⊡ dev(σ)) / 2) < 1.0e-9 * cohesion_intercept(m)
        @test isapprox(yield_function(m, σ), 0.0; atol = 1.0e-6 * cohesion_intercept(m))
        ## The tangent there is the elastic stand-in, not the true (zero) one.
        @test C ≈ elastic_stiffness(m.elastic, Val(3))
    end

    @testset "non-associated flow dilates less than associated" begin
        ## Same friction, smaller dilatancy: less plastic volume change for the same path.
        assoc = DruckerPrager(; E = E, nu = ν, cohesion = c, friction = φ, dilatancy = φ)
        nonassoc = DruckerPrager(;
            E = E, nu = ν, cohesion = c, friction = φ, dilatancy = deg2rad(5)
        )
        @test dilatancy_coefficient(nonassoc) < dilatancy_coefficient(assoc)

        volumetric(mat) = let state = initial_state(mat, zero(SymmetricTensor{2, 3}))
            for k in 1:8
                _, _, state = material_response(mat, shear_strain(4.0e-4 * k), state, 1.0)
            end
            tr(state.εp)
        end
        @test volumetric(nonassoc) < volumetric(assoc)
        @test volumetric(assoc) > 0          # an associated cone dilates
    end

    @testset "differentiable with respect to the material parameters" begin
        ## The point of the package: a Dual in the cohesion or the friction angle has to
        ## survive the corrector, not only the elastic branch.
        function sheared_q(θ)
            mat = DruckerPrager(;
                E = E, nu = ν, cohesion = θ[1], friction = θ[2], dilatancy = θ[2]
            )
            state = initial_state(mat, zero(SymmetricTensor{2, 3}))
            σ, _, _ = material_response(mat, shear_strain(3.0e-3), state, 1.0)
            s = dev(σ)
            return sqrt(3 * (s ⊡ s) / 2)
        end

        θ = [c, φ]
        g = ForwardDiff.gradient(sheared_q, θ)
        @test all(isfinite, g)
        @test g[1] != 0                       # more cohesion, more deviator carried
        @test g[2] != 0

        ## Against central differences, which is the only check available here.
        fd = map(eachindex(θ)) do i
            h = 1.0e-6 * abs(θ[i])
            θp = copy(θ); θp[i] += h
            θm = copy(θ); θm[i] -= h
            (sheared_q(θp) - sheared_q(θm)) / (2h)
        end
        @test isapprox(g, fd; rtol = 1.0e-5)
    end

    @testset "a state may mix Dual and Float64 tensors" begin
        ## Differentiating with respect to a parameter leaves the imposed strain a Float64
        ## while the stress becomes a Dual. A state demanding one shared tensor type would
        ## reject exactly that, so the mixture is asserted rather than assumed.
        ##
        ## The Dual has to come from a real `ForwardDiff.gradient` and not be built by hand:
        ## `material_response` runs its own `Tensors.gradient` inside, and ForwardDiff can
        ## only nest tags it can order. A hand-made `Dual{:tag}` carries a `Symbol` where the
        ## inner one carries a `Tag` type, and the two are incomparable.
        captured = Ref{Any}(nothing)
        ForwardDiff.gradient([c]) do θ
            mat = DruckerPrager(;
                E = E, nu = ν, cohesion = θ[1], friction = φ, dilatancy = φ
            )
            state = initial_state(mat, zero(SymmetricTensor{2, 3}))
            σ, _, new_state = material_response(mat, shear_strain(3.0e-3), state, 1.0)
            captured[] = (eltype(σ), eltype(new_state.ε), eltype(new_state.εp))
            return tr(σ)
        end

        σ_type, ε_type, εp_type = captured[]
        @test σ_type <: ForwardDiff.Dual
        @test εp_type <: ForwardDiff.Dual
        @test ε_type == Float64
    end
end
