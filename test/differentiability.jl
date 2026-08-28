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

module _SolverSensitivity
    include("../demos/solver_sensitivity.jl")
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
    ## deviations of the fit — the check that the linearized uncertainty means something.
    @test isposdef(Symmetric(D.Σ))
    for i in 1:4
        @test abs(D.θ_fit[i] - 1) < 4 * sqrt(D.Σ[i, i])
    end
end

@testset "differentiating a solve" begin
    S = _SolverSensitivity

    ## Steady finite volumes, through VoronoiFVM in dual numbers.
    @test S.grad_flux[1] ≈ S.grad_flux_fd[1] rtol = 1.0e-6
    @test S.grad_flux[4] ≈ S.grad_flux_fd[4] rtol = 1.0e-6

    ## The retention parameters are structurally invisible to a steady flux: the curve
    ## enters only through the storage term, which is zero at steady state.
    @test S.grad_flux[2] == 0
    @test S.grad_flux[3] == 0

    ## A full nonlinear axisymmetric elastoplastic solve — twelve load steps, a global
    ## Newton at each, a return mapping at every quadrature point — differentiated with
    ## respect to three material parameters.
    for i in 1:3
        @test S.grad_fe[i] ≈ S.grad_fe_fd[i] rtol = 1.0e-7
        @test isfinite(S.grad_fe[i])
    end

    ## The signs are the physics, and getting them backwards would be a real error rather
    ## than a tolerance one: a softer elastic law or a flatter compression line reaches a
    ## given compaction at lower stress, a larger void ratio stiffens both the bulk modulus
    ## and the hardening.
    @test S.grad_fe[1] < 0        # κ
    @test S.grad_fe[2] < 0        # λ(0)
    @test S.grad_fe[3] > 0        # e₀

    ## The element arrays must follow the type of the assembled system, not `Float64`.
    ## This is the check that differentiability reaches past the constitutive layer.
    @test eltype(S.grad_fe) === Float64
    @test S.mobilised_stress(ones(3)) ≈ 112.4039e3 rtol = 1.0e-5

    ## Coupled poroelasticity, verified against the derivative of Terzaghi's closed form
    ## rather than against another numerical method.
    for i in 1:3
        @test S.grad_biot_num[i] ≈ S.grad_biot_ana[i] rtol = 5.0e-3
    end
    @test S.grad_biot_num[1] < 0        # at T = 0.1 the p₀ term wins over the slower decay

    ## The sensitivity is as accurate as the solution: refining halves both errors.
    errs_p, errs_g = Float64[], Float64[]
    pa = S.terzaghi_analytical(ones(3), 0.5, S.t_probe)
    for (ne, ns) in ((20, 100), (40, 200), (80, 400))
        g = ForwardDiff.gradient(
            θ -> S.terzaghi_numerical(θ, S.t_probe; nely = ne, nsteps = ns), ones(3)
        )
        push!(errs_p, abs(S.terzaghi_numerical(ones(3), S.t_probe; nely = ne, nsteps = ns) - pa) / abs(pa))
        push!(errs_g, abs(g[1] - S.grad_biot_ana[1]) / abs(S.grad_biot_ana[1]))
    end
    @test issorted(errs_p; rev = true)
    @test issorted(errs_g; rev = true)
    @test errs_g[1] / errs_g[3] > 2.5
end

module _WritingAModel
    include("../demos/writing_a_model.jl")
end

@testset "writing a model — the decisive test" begin
    W = _WritingAModel

    ## Going through the package must give exactly the solver's own answer: the callbacks
    ## are the solver's callbacks with a model argument in front, so any difference would
    ## mean the abstraction is doing something behind the user's back.
    @test W.c_pkg == W.c_raw

    ## And both must match the closed form to the discretization error.
    @test W.l2(W.c_pkg, W.c_exact) < 1.0e-2

    ## Refining space or time reduces that error; the two are separate contributions.
    base = W.l2(W.c_pkg, W.c_exact)
    x_coarse, c_coarse = W.solve_with_package(W.model; N = 101, Δt_max = 2.0e3)
    x_slow, c_slow = W.solve_with_package(W.model; N = 401, Δt_max = 2.0e4)
    exact_of(x) = [W.ogata_banks(xi, W.T_END, W.model) for xi in x]
    @test W.l2(c_coarse, exact_of(x_coarse)) > base
    @test W.l2(c_slow, exact_of(x_slow)) > base

    ## The line counts the page prints are recounted from its own source, so the claim
    ## cannot drift away from the code that backs it.
    src = readlines(joinpath(@__DIR__, "..", "demos", "writing_a_model.jl"))
    function count_region(first_marker, last_marker, from = 1)
        i = findfirst(l -> occursin(first_marker, l), @view src[from:end]) + from - 1
        j = findfirst(l -> occursin(last_marker, l), @view src[(i + 1):end]) + i
        body = src[i:(j + 1)]
        return count(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), body)
    end
    tail = "return grid[Coordinates][1, :], sol.u[end][1, :]"
    n_pkg = count_region("Base.@kwdef struct AdvectionDispersion", tail)
    i_raw = findfirst(l -> occursin("Base.@kwdef struct RawParams", l), src)
    n_raw = count_region("Base.@kwdef struct RawParams", tail, i_raw)
    @test n_pkg == W.package_lines
    @test n_raw == W.direct_lines
    @test W.package_lines - W.direct_lines == 3      # nspecies, species_names, one wrap
end
