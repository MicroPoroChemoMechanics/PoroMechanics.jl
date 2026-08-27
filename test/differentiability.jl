# Differentiation with respect to the *parameters*, which is the package's distinguishing
# claim. Differentiating with respect to the unknowns is what the solver packages already
# do; these tests are about `ForwardDiff.Dual` living in a material coefficient and
# surviving a return mapping and a stress-controlled Newton on top of it.

using ForwardDiff
using LinearAlgebra
using Random

module _Identification
    include("../demos/parameter_identification.jl")
end

@testset "differentiability with respect to parameters" begin
    I3 = one(SymmetricTensor{2, 3})

    ## A `Dual` in a parameter, through an elastic step and through a plastic one.
    function plastic_path(κ; nstep = 20)
        m = BBM(κ = κ)
        T = typeof(κ)
        st = initial_state(m, -1.0e3 * one(SymmetricTensor{2, 3, T}), T(4.0e4); suction = zero(T))
        for k in 1:nstep
            st = material_response(m, -(0.06 * k / nstep / 3) * one(SymmetricTensor{2, 3, T}), st, 1.0)[3]
        end
        return st.εv_p
    end
    d_ad = ForwardDiff.derivative(plastic_path, 0.011)
    h = 1.0e-7
    d_fd = (plastic_path(0.011 + h) - plastic_path(0.011 - h)) / (2h)
    @test isfinite(d_ad)
    @test d_ad ≈ d_fd rtol = 1.0e-5
    @test d_ad < 0                      # a stiffer elastic law yields later, so less plastic strain

    ## The apex of the deviatoric cone. `q = √(3/2 e:e)` has an infinite slope at `e = 0`,
    ## so an isotropic state — where the deviator is round-off — is exactly where a
    ## parameter derivative used to come back `NaN`. The value must stay right and the
    ## derivative must be zero, not large and not `NaN`.
    @test equivalent_stress(-1.0e3 * I3) == 0
    @test isfinite(ForwardDiff.derivative(κ -> equivalent_stress(-1.0e3 * one(SymmetricTensor{2, 3, typeof(κ)})), 0.011))
    @test ForwardDiff.derivative(x -> equivalent_stress(x * I3), -1.0e3) == 0

    ## A genuine deviator is untouched by the guard.
    σd = SymmetricTensor{2, 3}((i, j) -> i == j == 1 ? -2.0e3 : i == j ? -1.0e3 : 0.0)
    @test equivalent_stress(σd) ≈ 1.0e3

    ## The eight-parameter gradient of the demo, against central differences. The two the
    ## comparison cannot settle are asserted separately below.
    D = _Identification
    g = D.gradient_ad
    for i in 1:6
        @test g[i] ≈ D.gradient_fd[i] rtol = 1.0e-6
    end

    ## Structural zeros: under stress control the plastic strain is set by the yield surface
    ## and the hardening law, so the elastic constants cannot enter it. Automatic
    ## differentiation returns that exactly; finite differences return their own noise,
    ## which is why they are not the reference here.
    @test abs(g[7]) < 1.0e-15           # κ_s
    @test abs(g[8]) < 1.0e-10           # ν
    @test abs(D.gradient_fd[7]) > 1.0e-11
    @test abs(D.gradient_fd[8]) > 1.0e-11

    ## Experiment design: a plausible protocol that cannot determine β, and one that can.
    @test D.svd_plausible.S[end] / D.svd_plausible.S[1] < 1.0e-15
    @test abs(D.svd_plausible.V[4, end]) > 0.99          # the blind direction is β alone
    @test D.svd_informative.S[end] / D.svd_informative.S[1] > 0.02
    @test all(>(0.0), D.svd_informative.S)

    ## Calibration recovers the parameters it was given, from a start more than a third away.
    for i in 1:4
        @test D.θ_fit[i] ≈ 1.0 rtol = 1.0e-2
        @test abs(D.θ_fit[i] - 1) < abs(D.θ_start[i] - 1) / 20
    end
    @test length(D.history) - 1 <= 15
    @test last(D.history) < first(D.history) / 1000

    ## The covariance is positive definite and the true value sits within a few standard
    ## deviations of the fit — the check that the linearised uncertainty means something.
    @test isposdef(Symmetric(D.Σ))
    for i in 1:4
        @test abs(D.θ_fit[i] - 1) < 4 * sqrt(D.Σ[i, i])
    end
end
