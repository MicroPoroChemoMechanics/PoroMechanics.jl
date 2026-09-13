module _EquilibratedTransportTests
using Test, LinearAlgebra, ForwardDiff, ChemistryLab, DynamicQuantities
using VoronoiFVM, ExtendableGrids
include("../../examples/chloride_ingress/repro_equilibrated_transport.jl")

# An uncertified solve is injected only to make the rejection path deterministic.
struct RejectedEquilibrium{S}
    state::S
end
function equilibrium_state(
        m::EquilibratedTransport{C, S, R, P, TT, B, <:RejectedEquilibrium}, totals,
    ) where {C, S, R, P, TT, B}
    return m.initial_state.state, (; optimal = false)
end

@testset "Equilibrium-coupled transport" begin
    m, totals = opc_equilibrated_case()
    cmp = m.components
    A = Float64.(m.system.SM.A)
    original = A * ustrip.(us"mol", m.initial_state.n)
    @test component_totals(m, totals) ≈ original atol = 1.0e-12
    @test totals[end] < 0 # Signed hydrogen component; never a species amount.

    @testset "complete basis and physical initial state" begin
        incomplete = ComponentSet(; names = cmp.names, z = cmp.z, D = cmp.D)
        @test_throws ArgumentError equilibrated_transport(incomplete, m.system; initial_state = m.initial_state)
        duplicate = ComponentSet(;
            names = cmp.names, z = cmp.z, D = cmp.D,
            fixed = merge(cmp.fixed, (; var"Cl-" = 1.0))
        )
        @test_throws ArgumentError equilibrated_transport(duplicate, m.system; initial_state = m.initial_state)
        @test_throws DimensionMismatch component_totals(m, totals[1:2])
    end

    eq, cert = equilibrium_state(m, totals)
    @test cert.optimal
    @test norm(A * ustrip.(us"mol", eq.n) - original, Inf) < 1.0e-7
    c, ok = speciate(m, totals)
    @test ok
    aq_amounts = ustrip.(us"mol", eq.n[m.system.idx_aqueous])
    reference = A[m.rows.transported, m.system.idx_aqueous] * aq_amounts /
        ustrip(us"m^3", eq.V_phases[].liquid)
    @test collect(c) ≈ reference rtol = 1.0e-9

    @testset "conserved, certified sensitivities" begin
        seeded = [ForwardDiff.Dual{Nothing}(v, k == 1 ? 1.0 : 0.0) for (k, v) in enumerate(totals)]
        dual_eq, dual_cert = equilibrium_state(m, seeded)
        @test dual_cert.optimal
        @test ChemistryLab.optimality_certificate(
            ChemistryLab.DualEquilibriumSolver(m.system), dual_eq; b = component_totals(m, seeded)
        ).optimal
        Jn = ForwardDiff.jacobian(t -> ustrip.(us"mol", first(equilibrium_state(m, t)).n), totals)
        db = zeros(size(A, 1), length(totals))
        db[m.rows.transported, :] .= Matrix{Float64}(I, length(totals), length(totals))
        @test norm(A * Jn - db, Inf) < 1.0e-7
        J = ForwardDiff.jacobian(t -> collect(first(speciate(m, t))), totals)
        @test all(isfinite, J)
        for k in (1, 2) # Chloride and sodium: compare on a fixed phase branch.
            h = 1.0e-3
            plus, minus = copy(totals), copy(totals)
            plus[k] += h
            minus[k] -= h
            cp, op = speciate(m, plus)
            cm, om = speciate(m, minus)
            @test op && om
            fd = (collect(cp) - collect(cm)) / (2h)
            @test J[:, k] ≈ fd rtol = 1.0e-4 atol = 1.0e-6
        end
    end

    @testset "failed chemistry cannot produce a flux" begin
        rejected = EquilibratedTransport(
            m.components, m.system, m.rows, m.phi,
            m.tortuosity, m.dirichlet, RejectedEquilibrium(eq)
        )
        f = fill(NaN, length(totals))
        @test_throws LocalEquilibriumError PoroMechanics.flux!(f, hcat(totals, totals), nothing, rejected, nothing)
        @test all(isnan, f)
        invalid = copy(totals)
        invalid[1] = -1.0
        @test_throws DomainError PoroMechanics.flux!(f, hcat(invalid, totals), nothing, m, nothing)
    end

    @testset "closed-column transient conserves components" begin
        x = collect(range(0.0, 0.01; length = 3))
        sys = fvm_system(m, simplexgrid(x))
        initial = repeat(totals, 1, 3)
        initial[1, 1] += 1.0 # A NaCl perturbation at the left node.
        initial[2, 1] += 1.0
        # Retry failed Newton steps with a smaller time step down to Δt_min.
        control = VoronoiFVM.SolverControl(;
            Δt = 1.0, Δt_max = 1.0, Δt_min = 1.0e-4, handle_exceptions = true,
        )
        solution = VoronoiFVM.solve(sys; inival = initial, times = [0.0, 1.0], control)
        final = solution(1.0)
        weights = [0.0025, 0.005, 0.0025]
        @test all(isfinite, final)
        @test final * weights ≈ initial * weights rtol = 1.0e-9 atol = 1.0e-10
        @test final[1, 1] < initial[1, 1]
        @test final[1, 2] > initial[1, 2]
    end
end
end
