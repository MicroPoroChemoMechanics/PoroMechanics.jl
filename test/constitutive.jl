# Tests for the constitutive layer.
#
# Two kinds of check, and the second is the point of the layer:
#
#  1. Values — each law reproduces the closed-form expression it claims to implement.
#  2. Differentiability — every law is differentiable not only with respect to its
#     argument (which VoronoiFVM already needs for the Jacobian) but with respect to its
#     *parameters*. That is what makes inverse calibration and sensitivity analysis
#     possible later, and it only works because the coefficients are type-parameterized
#     rather than declared `Float64`.

using ForwardDiff
using FiniteDiff
using Tensors

"""
    @test_derivative f x

Check `ForwardDiff.derivative(f, x)` against a central finite difference.

The step is taken *relative to the point*. `FiniteDiff`'s default is scaled by
`max(|x|, 1)`, which is meaningless for a parameter like the storage modulus `N ≈ 7e-9`:
the difference would then be evaluated a thousand steps away from the point and disagree
with the exact derivative by orders of magnitude — as it did, until this was fixed.

Relative tolerance 1e-6: finite differences are the inaccurate side of this comparison.
"""
macro test_derivative(f, x)
    return quote
        local fun = $(esc(f))
        local pt = $(esc(x))
        local ad = ForwardDiff.derivative(fun, pt)
        local h = 1.0e-6 * max(abs(pt), floatmin())
        local fd = (fun(pt + h) - fun(pt - h)) / 2h
        @test isapprox(ad, fd; rtol = 1.0e-5, atol = 1.0e-12 * max(1, abs(fd)))
    end
end

@testset "Retention — VanGenuchten" begin
    vg = VanGenuchten(1.5e6, 0.06)          # constrained: n = 1/(1-m)

    @test vg.n ≈ 1 / (1 - 0.06)
    @test saturation(vg, -1.0) == 1.0        # saturated branch
    @test saturation(vg, 0.0) == 1.0
    @test dsaturation_dpc(vg, -1.0) == 0.0

    ## Closed form
    for pc in (1.0e4, 1.0e6, 1.5e6, 1.0e8)
        @test saturation(vg, pc) ≈ (1 + (pc / vg.a)^vg.n)^(-vg.m)
    end

    ## Monotone decreasing, bounded
    sats = [saturation(vg, pc) for pc in 10.0 .^ (3:0.5:9)]
    @test all(0 .< sats .<= 1)
    @test issorted(sats; rev = true)

    ## The free-exponent constructor must NOT re-derive n from m: published parameter
    ## sets quote a rounded n, and recomputing it shifts the curve.
    free = VanGenuchten(1.5e6, 1.06383, 0.06)
    @test free.n == 1.06383
    @test free.n != vg.n
    @test saturation(free, 1.0e6) != saturation(vg, 1.0e6)

    ## Analytic derivative against finite differences
    for pc in (1.0e4, 1.0e6, 1.0e7)
        fd = FiniteDiff.finite_difference_derivative(p -> saturation(vg, p), pc, Val{:central})
        @test isapprox(dsaturation_dpc(vg, pc), fd; rtol = 1.0e-6)
    end
end

@testset "Retention — ExponentialCutoff" begin
    raw = VanGenuchten(1.5e6, 1.06383, 0.06)
    reg = ExponentialCutoff(raw, 1.0e5)

    @test saturation(reg, -1.0) == 1.0
    @test saturation(reg, 2.0e5) == saturation(raw, 2.0e5)     # above the junction
    @test saturation(reg, 1.0e5) ≈ saturation(raw, 1.0e5)      # continuous at p_c3

    ## The whole point: a finite slope where the raw curve steepens without bound.
    @test abs(dsaturation_dpc(reg, 1.0)) < abs(dsaturation_dpc(raw, 1.0))
    @test isfinite(dsaturation_dpc(reg, 1.0e-8))

    for pc in (1.0e3, 5.0e4, 3.0e5)
        fd = FiniteDiff.finite_difference_derivative(p -> saturation(reg, p), pc, Val{:central})
        @test isapprox(dsaturation_dpc(reg, pc), fd; rtol = 1.0e-6)
    end
end

@testset "Relative permeability" begin
    mu = Mualem(3.0e6, 0.5)
    @test relative_permeability(mu, -1.0) == 1.0
    @test 0 <= relative_permeability(mu, 1.0e6) <= 1

    ## Far into the dry zone but above the guard (Se ≈ 3e-6 at pc = 1e12): still a
    ## genuine, vanishingly small value.
    @test 0 < relative_permeability(mu, 1.0e12) < 1.0e-20

    ## The guard fires only once Se < 1e-14, i.e. pc/a > 1e14.
    @test relative_permeability(mu, 1.0e22) == 0.0

    krs = [relative_permeability(mu, pc) for pc in 10.0 .^ (4:0.5:9)]
    @test issorted(krs; rev = true)

    pl = PowerLawKrl(3.0e6, 2.0, 0.5)
    for pc in (1.0e5, 3.0e6, 1.0e8)
        @test relative_permeability(pl, pc) ≈ (1 + (pc / 3.0e6)^2)^(-0.5)
    end

    @test gas_relative_permeability(1.0) == 0.0
    @test gas_relative_permeability(0.0) ≈ 1.0
    @test 0 < gas_relative_permeability(0.5) < 1
end

@testset "Tortuosity — OhJang" begin
    oj = OhJang(; phi_c = 0.18, n = 2.7, ds = 1.0e-4, tau_agg = 0.15)

    @test tortuosity(oj, 0.13) > 0
    @test tortuosity(oj, 0.13, 1.0) ≈ tortuosity(oj, 0.13)   # saturated default
    @test tortuosity(oj, 0.13, 0.5) < tortuosity(oj, 0.13, 1.0)

    ## More porosity, more transport
    @test tortuosity(oj, 0.2) > tortuosity(oj, 0.1)

    ## Saturation factor is exactly S_l^q
    @test tortuosity(oj, 0.13, 0.5) ≈ tortuosity(oj, 0.13, 1.0) * 0.5^4.5
end

# ── Differentiability with respect to the PARAMETERS ──────────────────────────
# This is the property the whole layer exists for. It works only because the structs
# are `VanGenuchten{T}` and not `VanGenuchten` with `Float64` fields.

@testset "Differentiable w.r.t. parameters" begin
    @testset "VanGenuchten.a" begin
        f = a -> saturation(VanGenuchten(a, 0.06), 1.0e6)
        @test_derivative f 1.5e6
        @test ForwardDiff.derivative(f, 1.5e6) != 0
    end

    @testset "VanGenuchten.m" begin
        f = m -> saturation(VanGenuchten(1.5e6, m), 1.0e6)
        @test_derivative f 0.06
    end

    @testset "Mualem.a" begin
        f = a -> relative_permeability(Mualem(a, 0.5), 1.0e6)
        @test_derivative f 3.0e6
    end

    @testset "OhJang.tau_agg" begin
        f = ta -> tortuosity(OhJang(; phi_c = 0.18, n = 2.7, ds = 1.0e-4, tau_agg = ta), 0.13)
        @test_derivative f 0.15
        ## τ is linear in tau_agg, so the derivative is τ/tau_agg exactly
        @test ForwardDiff.derivative(f, 0.15) ≈ f(0.15) / 0.15
    end

    @testset "OhJang.phi_c" begin
        f = pc -> tortuosity(OhJang(; phi_c = pc, n = 2.7, ds = 1.0e-4, tau_agg = 0.15), 0.13)
        @test_derivative f 0.18
    end

    @testset "gradient over several parameters at once" begin
        ## The shape an inverse-calibration problem actually takes: one scalar output,
        ## a gradient with respect to the whole parameter vector.
        g = p -> saturation(VanGenuchten(p[1], p[2], p[3]), 1.0e6)
        p0 = [1.5e6, 1.06383, 0.06]
        ad = ForwardDiff.gradient(g, p0)
        fd = FiniteDiff.finite_difference_gradient(g, p0)
        @test isapprox(ad, fd; rtol = 1.0e-5)
        @test all(ad .!= 0)
    end
end

# ── Differentiability with respect to the argument ────────────────────────────
# What VoronoiFVM needs for the Jacobian, and where the guards matter.

@testset "Differentiable w.r.t. pc" begin
    vg = VanGenuchten(1.5e6, 0.06)
    mu = Mualem(3.0e6, 0.5)

    for pc in (1.0e4, 1.0e6, 1.0e8)
        @test_derivative (p -> saturation(vg, p)) pc
        @test_derivative (p -> relative_permeability(mu, p)) pc
    end

    ## The dry-zone guard must give a finite (zero) gradient, not a NaN. Without it,
    ## d√Se/dSe = 1/(2√Se) blows up as Se → 0.
    d = ForwardDiff.derivative(p -> relative_permeability(mu, p), 1.0e22)
    @test isfinite(d)
    @test d == 0.0

    ## Just above the guard the gradient is finite too
    @test isfinite(ForwardDiff.derivative(p -> relative_permeability(mu, p), 1.0e12))

    ## And the saturated branch must not produce NaN either
    @test ForwardDiff.derivative(p -> saturation(vg, p), -1.0) == 0.0
end

# ── Poroelasticity ────────────────────────────────────────────────────────────

@testset "Poroelasticity — coefficients from a microstructure" begin
    ## The interface with MeanFieldHomogenization.jl, guarded from this side.
    ##
    ## That package computes the Biot tensor and modulus from a microstructure; this one
    ## consumes a scalar `b` and a storage modulus `N` and asks no questions. The live
    ## comparison is `benchmarks/mfh_poroelastic.jl`, which the documentation build runs.
    ## What is frozen here are the numbers it produced, so that a change on either side of
    ## the bridge is caught by a suite that does not depend on the other package.
    ##
    ## Microstructure: spherical pores at φ = 0.2 in a solid with E = 60 GPa, ν = 0.25,
    ## homogenized by Mori-Tanaka.
    φ, k_s = 0.2, 60.0e9 / (3 * (1 - 2 * 0.25))
    E_hom, ν_hom = 39.963800904977366e9, 0.23981900452488688
    b_mfh, N_mfh = 0.359999999995, 4.0e-12                    # MeanFieldHomogenization 0.7.0

    ## `par.B[1,1]` is this package's `b` and `par.inverse_modulus` is its `N` — no
    ## conversion in between, which is the whole content of the bridge.
    m = BiotPoroelastic(; E = E_hom, nu = ν_hom, k = 1.0e-16, mu_l = 1.0e-3, b = b_mfh, N = N_mfh)

    ## The classical relations, computed here, must reproduce what the tensorial route gave.
    @test b_mfh ≈ 1 - bulk_modulus(m) / k_s rtol = 1.0e-9
    @test N_mfh ≈ (b_mfh - φ) / k_s rtol = 1.0e-9

    ## And the consequence both packages agree on: with an incompressible fluid and
    ## compressible grains the Skempton coefficient exceeds one, because the pore volume is
    ## held fixed while the grains themselves compress. Written from Biot's relations here,
    ## warned about from the homogenization there.
    @test skempton(m) > 1
    @test skempton(m) ≈ 1.5517241379310347 rtol = 1.0e-10
end

@testset "Poroelasticity — constants" begin
    m = BiotPoroelastic()
    λ, μ = lame(m)

    ## The elastic constants must agree with each other, whichever way they are reached.
    @test μ ≈ shear_modulus(m)
    @test bulk_modulus(m) ≈ λ + 2μ / 3
    @test oedometric_modulus(m) ≈ λ + 2μ
    @test oedometric_modulus(m) ≈ bulk_modulus(m) + 4μ / 3
    @test bulk_modulus(m) ≈ m.E / (3 * (1 - 2m.nu))

    ## Storage, compaction and diffusivity are one relation seen three ways.
    @test compaction_coefficient(m) ≈ m.b / oedometric_modulus(m)
    @test storage_coefficient(m) ≈ m.N + m.b^2 / oedometric_modulus(m)
    @test consolidation_coefficient(m) ≈ hydraulic_conductivity(m) / storage_coefficient(m)

    ## Skempton and the undrained Poisson ratio, against their closed forms.
    B = skempton(m)
    @test B ≈ m.b * biot_modulus(m) / (bulk_modulus(m) + m.b^2 * biot_modulus(m))
    @test undrained_bulk_modulus(m) ≈ bulk_modulus(m) + m.b^2 * biot_modulus(m)
    @test m.nu < undrained_poisson(m) < 0.5      # draining always softens

    ## Independent cross-check: the diffusivity computed from the storage coefficient must
    ## equal Cheng & Detournay's form, which is written in entirely different variables.
    ν, νu = m.nu, undrained_poisson(m)
    κ = hydraulic_conductivity(m)
    c_cheng = 2κ * B^2 * μ * (1 - ν) * (1 + νu)^2 / (9 * (1 - νu) * (νu - ν))
    @test isapprox(consolidation_coefficient(m), c_cheng; rtol = 1.0e-10)

    ## Incompressible limit: B → 1 and ν_u → 1/2.
    stiff = BiotPoroelastic(; E = 1.0e8, nu = 0.2, k = 1.0e-13, mu_l = 1.0e-3, b = 1.0, N = 1.0e-18)
    @test isapprox(skempton(stiff), 1.0; rtol = 1.0e-8)
    @test isapprox(undrained_poisson(stiff), 0.5; rtol = 1.0e-8)
end

@testset "Poroelasticity — differentiable w.r.t. parameters" begin
    ## The property the whole layer exists for, on the poroelastic constants this time.
    ## It works only because `BiotPoroelastic{T}` carries its coefficients as a type
    ## parameter; with `E::Float64` none of these derivatives would exist.
    base = (E = 1.0e8, nu = 0.2, k = 1.0e-13, mu_l = 1.0e-3, b = 1.0, N = 7.2e-9)
    mk(; kwargs...) = BiotPoroelastic(; base..., kwargs...)

    @testset "consolidation coefficient w.r.t. b" begin
        f = b -> consolidation_coefficient(mk(; b = b))
        @test_derivative f 1.0
        ## More coupling stiffens the storage, so the diffusivity falls.
        @test ForwardDiff.derivative(f, 1.0) < 0
    end

    @testset "Skempton w.r.t. N" begin
        f = N -> skempton(mk(; N = N))
        @test_derivative f 7.2e-9
        ## Also against the exact derivative, dB/dN = −bK/(NK + b²)², which is worth
        ## writing out because N is nine orders of magnitude from unity and any
        ## finite-difference check has to be told so.
        K = bulk_modulus(mk())
        @test ForwardDiff.derivative(f, 7.2e-9) ≈ -K / (7.2e-9 * K + 1)^2
        @test ForwardDiff.derivative(f, 7.2e-9) < 0
    end

    @testset "undrained Poisson w.r.t. nu" begin
        @test_derivative (ν -> undrained_poisson(mk(; nu = ν))) 0.2
    end

    @testset "oedometric modulus w.r.t. E" begin
        f = E -> oedometric_modulus(mk(; E = E))
        @test_derivative f 1.0e8
        ## Linear in E, so the derivative is M_o/E exactly.
        @test ForwardDiff.derivative(f, 1.0e8) ≈ f(1.0e8) / 1.0e8
    end

    @testset "gradient over the whole parameter vector" begin
        ## The shape an inverse calibration of a consolidation curve would take.
        g = p -> consolidation_coefficient(BiotPoroelastic(p[1], p[2], p[3], p[4], p[5], p[6]))
        p0 = [1.0e8, 0.2, 1.0e-13, 1.0e-3, 1.0, 7.2e-9]
        ad = ForwardDiff.gradient(g, p0)
        fd = FiniteDiff.finite_difference_gradient(g, p0)
        @test isapprox(ad, fd; rtol = 1.0e-5)
        @test all(ad .!= 0)
    end

    @testset "a Dual really does enter the struct" begin
        d = ForwardDiff.Dual(1.0, 1.0)
        m = BiotPoroelastic(; base..., b = d)
        @test eltype(m) <: ForwardDiff.Dual
        @test skempton(m) isa ForwardDiff.Dual
    end
end

# ── The stress–strain interface ───────────────────────────────────────────────

@testset "Material interface — LinearElastic" begin
    mat = LinearElastic(; E = 1.0e8, nu = 0.2)
    ε = SymmetricTensor{2, 2}((1.0e-3, 2.0e-4, -5.0e-4))
    σ, C, st = material_response(mat, ε, initial_state(mat), 1.0)

    ## The two constructors agree
    @test LinearElastic(mat.λ, mat.μ).λ ≈ mat.λ

    ## No internal variables, and the state comes back untouched
    @test initial_state(mat) === NoState()
    @test st === NoState()

    ## The stress matches the closed form, and the tangent reproduces it
    @test σ ≈ mat.λ * tr(ε) * one(ε) + 2 * mat.μ * ε
    @test C ⊡ ε ≈ σ

    ## Isotropy: a pure shear produces no volumetric stress, a pure dilation no shear
    shear = SymmetricTensor{2, 2}((0.0, 1.0e-3, 0.0))
    @test tr(material_response(mat, shear, NoState(), 1.0)[1]) ≈ 0.0 atol = 1.0e-6
    dil = SymmetricTensor{2, 2}((1.0e-3, 0.0, 1.0e-3))
    @test material_response(mat, dil, NoState(), 1.0)[1][1, 2] ≈ 0.0 atol = 1.0e-6

    ## Symmetries of the stiffness: minor (from the symmetric tensor type) and major
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        @test C[i, j, k, l] ≈ C[k, l, i, j]
    end

    ## **The consistent-tangent check.** ∂σ/∂ε from the interface must equal the derivative
    ## of the stress itself. Trivial for linear elasticity — and exactly the test a return
    ## mapping will have to pass, where the tangent is anything but obvious.
    C_ad = Tensors.gradient(e -> material_response(mat, e, NoState(), 1.0)[1], ε)
    @test C_ad ≈ C

    ## Differentiable in its parameters, like every other law here.
    f = μ -> material_response(LinearElastic(mat.λ, μ), ε, NoState(), 1.0)[1][1, 1]
    @test_derivative f mat.μ
    @test ForwardDiff.derivative(f, mat.μ) ≈ 2 * ε[1, 1]
end

@testset "Material interface — poroelastic layer" begin
    m = BiotPoroelastic()
    ε = SymmetricTensor{2, 2}((1.0e-3, 2.0e-4, -5.0e-4))
    p = 1.0e5

    ## The skeleton is the hinge: below it, ordinary solid mechanics.
    sk = skeleton(m)
    @test sk isa LinearElastic
    @test (sk.λ, sk.μ) == lame(m)

    σ, C, ∂σ∂p, st = poro_response(m, ε, p, initial_state(sk), 1.0)
    σ_eff, _, _ = material_response(sk, ε, NoState(), 1.0)

    ## Total stress is the skeleton response less the pressure term
    @test σ ≈ σ_eff - m.b * p * one(σ_eff)
    @test σ ≈ total_stress(m.b, σ_eff, p)
    @test ∂σ∂p ≈ -m.b * one(σ_eff)

    ## A positive pore pressure relieves the skeleton: tension positive.
    @test tr(σ) < tr(σ_eff)

    ## Zero pressure recovers pure mechanics — the two layers really are separable.
    @test poro_response(m, ε, 0.0, NoState(), 1.0)[1] ≈ σ_eff

    ## Both tangents against automatic differentiation of the response itself.
    @test Tensors.gradient(e -> poro_response(m, e, p, NoState(), 1.0)[1], ε) ≈ C
    @test ForwardDiff.derivative(q -> poro_response(m, ε, q, NoState(), 1.0)[1][1, 1], p) ≈
        ∂σ∂p[1, 1]

    ## And differentiable through the material parameters, not only the fields.
    g = b -> tr(poro_response(BiotPoroelastic(; b = b), ε, p, NoState(), 1.0)[1])
    @test_derivative g 1.0
end

# ── Unsaturated effective stress ──────────────────────────────────────────────

@testset "Bishop coefficient and equivalent pore pressure" begin
    vg = VanGenuchten(1.5e6, 0.06)
    χ = SaturationBishop(vg)

    ## Saturated: χ = 1, and the equivalent pressure is just the liquid pressure. This is
    ## the check that the unsaturated theory contains the saturated one.
    @test bishop_coefficient(χ, 0.0) == 1.0
    @test equivalent_pore_pressure(χ, 1.0e5) ≈ 1.0e5
    @test suction_stress(χ, 1.0e5) == 0.0

    ## Unsaturated: χ falls with suction, and stays in [0,1]
    for pc in (1.0e4, 1.0e6, 1.0e8)
        c = bishop_coefficient(χ, pc)
        @test 0 < c <= 1
    end
    @test bishop_coefficient(χ, 1.0e8) < bishop_coefficient(χ, 1.0e4)

    ## π = p_g − χ p_c, i.e. the saturation-weighted average of the phase pressures
    p_l, p_g = -1.0e6, 0.0
    pc = p_g - p_l
    Sl = saturation(vg, pc)
    @test equivalent_pore_pressure(χ, p_l, p_g) ≈ Sl * p_l + (1 - Sl) * p_g

    ## Suction pulls the grains together: an isotropic tension on the skeleton
    @test suction_stress(χ, p_l) > 0
    @test suction_stress(χ, p_l) ≈ Sl * pc

    ## n = 1 recovers the saturation form; larger n weights the liquid less
    @test bishop_coefficient(PowerBishop(vg, 1.0), 1.0e6) ≈ bishop_coefficient(χ, 1.0e6)
    @test bishop_coefficient(PowerBishop(vg, 2.0), 1.0e6) < bishop_coefficient(χ, 1.0e6)

    ## The total stress reduces to the saturated form when the medium is saturated
    m = BiotPoroelastic()
    σ_eff = SymmetricTensor{2, 2}((-1.0e5, 1.0e4, -2.0e5))
    @test unsaturated_total_stress(m.b, χ, σ_eff, 1.0e5) ≈ total_stress(m.b, σ_eff, 1.0e5)
    ## ...and differs from it once there is suction
    @test unsaturated_total_stress(m.b, χ, σ_eff, -1.0e6) != total_stress(m.b, σ_eff, -1.0e6)

    ## Differentiable through the retention parameter, like everything else here
    f = a -> equivalent_pore_pressure(SaturationBishop(VanGenuchten(a, 0.06)), -1.0e6)
    @test_derivative f 1.5e6
end

# ── Pressure-dependent elasticity ─────────────────────────────────────────────

@testset "LogarithmicElastic" begin
    m = LogarithmicElastic()
    σ0 = -1.0e5 * one(SymmetricTensor{2, 2})     # 100 kPa isotropic compression

    ## No default state: the modulus is undefined without a stress to evaluate it at.
    @test_throws ErrorException initial_state(m)

    st = initial_state(m, σ0)
    @test st.σ == σ0
    @test st.ε == zero(σ0)

    ## K grows in proportion to the mean compressive stress
    K1, G1 = tangent_moduli(m, σ0)
    K2, G2 = tangent_moduli(m, 2σ0)
    @test K1 ≈ mean_compressive_stress(m, σ0) * (1 + m.e0) / m.κ
    @test K2 ≈ 2K1
    @test G1 / K1 ≈ 3 * (1 - 2m.nu) / (2 * (1 + m.nu))

    ## The floor engages in tension, where the law has no meaning
    @test mean_compressive_stress(m, -σ0) == m.p_min
    @test tangent_moduli(m, -σ0)[1] > 0

    ## The response is an increment from the stored state
    ε = SymmetricTensor{2, 2}((-1.0e-4, 0.0, -1.0e-4))
    σ, C, st2 = material_response(m, ε, st, 1.0)
    @test σ ≈ st.σ + C ⊡ (ε - st.ε)
    @test st2.σ == σ && st2.ε == ε
    ## Compression stiffens it: the next step is stiffer than this one.
    @test tangent_moduli(m, st2.σ)[1] > tangent_moduli(m, st.σ)[1]

    ## The consistent tangent, again against automatic differentiation of the response.
    ## The material now has a state and a stress-dependent stiffness, so this is no longer
    ## the tautology it was for linear elasticity.
    @test Tensors.gradient(e -> material_response(m, e, st, 1.0)[1], ε) ≈ C

    ## Differentiable in the swelling index — the parameter one would calibrate.
    f = κ -> tr(material_response(LogarithmicElastic(; κ = κ), ε, st, 1.0)[1])
    @test_derivative f 0.02
    @test ForwardDiff.derivative(f, 0.02) != 0
end
