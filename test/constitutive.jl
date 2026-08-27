# Tests for the constitutive layer.
#
# Two kinds of check, and the second is the point of the layer:
#
#  1. Values — each law reproduces the closed-form expression it claims to implement.
#  2. Differentiability — every law is differentiable not only with respect to its
#     argument (which VoronoiFVM already needs for the Jacobian) but with respect to its
#     *parameters*. That is what makes inverse calibration and sensitivity analysis
#     possible later, and it only works because the coefficients are type-parameterised
#     rather than declared `Float64`.

using ForwardDiff
using FiniteDiff

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
    @test tortuosity(oj, 0.20) > tortuosity(oj, 0.10)

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
