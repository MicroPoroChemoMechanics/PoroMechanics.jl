# The monosulphate ⇌ Friedel's salt exchange, as a box complementarity.
#
# The piece that introduces non-smoothness, and the one the plan flagged as most likely to
# misbehave. What makes it testable is that a single solution contains all three regimes at
# once — the boundary is fully converted, a node at the front coexists, and the interior has
# no Friedel's salt at all — so the complementarity can be checked branch by branch rather
# than in aggregate.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using ForwardDiff

module _AFM
    const HERE = joinpath(@__DIR__, "..", "..", "examples", "chloride_ingress")
    include(joinpath(HERE, "dlm.jl"))
    using PoroMechanics
    include(joinpath(HERE, "sorbing_transport.jl"))
    include(joinpath(HERE, "afm_exchange.jl"))
end

@testset "AFm exchange" begin
    A = _AFM

    @testset "the reformulation is right on all four corners" begin
        ## Two feasible states, which must give zero, and two infeasible ones, which must
        ## not. The nested Fischer-Burmeister that one writes first gets this exactly
        ## backwards, so the corners are worth pinning.
        ε = 1.0e-10
        @test abs(A.box_mcp(0.0, 2.0, ε)) < 1.0e-8       # x = 0, F > 0 : feasible
        @test abs(A.box_mcp(1.0, -2.0, ε)) < 1.0e-8      # x = 1, F < 0 : feasible
        @test abs(A.box_mcp(0.4, 0.0, ε)) < 1.0e-8       # interior, F = 0 : feasible
        @test abs(A.box_mcp(0.0, -2.0, ε)) > 0.1         # x = 0 but F < 0 : infeasible
        @test abs(A.box_mcp(1.0, 2.0, ε)) > 0.1          # x = 1 but F > 0 : infeasible
    end

    @testset "the smoothing converges, and it is what gives the row a diagonal" begin
        ## In the interior branch `Φ = λF`, which does not involve `x` at all: the row's own
        ## diagonal is zero and `x̂` is fixed by the chloride and sulfate balances instead.
        ## The `ε` smoothing puts a small diagonal back, so it cannot be driven to zero for
        ## free — the linear system gets harder, not easier.
        d(ε) = ForwardDiff.derivative(x -> A.box_mcp(x, 0.0, ε; λ = 1.0), 0.4)
        @test abs(A.box_mcp(0.4, 0.0, 1.0e-2)) > abs(A.box_mcp(0.4, 0.0, 1.0e-6))
        @test abs(d(1.0e-2)) > abs(d(1.0e-6))
    end

    ## Cl, Na, K, Ca, SO₄, OH, then Ψ, β, x̂. Both compositions are electroneutral.
    CBC = (523.0, 523.0, 1.0, 0.05, 0.05, 1.0)
    CIC = (1.0e-3, 300.0, 100.0, 1.0, 5.0, 391.999)
    ZZ = (-1, 1, 1, 2, -2, -1)
    N_AFM = 100.0

    grid = simplexgrid(range(0, 0.05; length = 41))
    dlm = A.DLM_TRAN2018(n_csh0 = 635.0)
    ions = A.IonIndex(; Cl = 1, Na = 2, K = 3, Ca = 4, OH = 6, SO4 = 5)
    transport = A.NernstPlanck(;
        phi = 0.121,
        D = (2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9, 1.065e-9, 5.273e-9),
        z = ZZ,
        tortuosity = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27),
        dirichlet = ntuple(i -> i <= 6 ? ((1, CBC[i]),) : ((1, 0.0),), 7),
    )
    model = A.AFmExchange(;
        inner = A.SurfaceResolvedTransport(;
            transport = transport, dlm = dlm, ions = ions, n_csh = 635.0, x_cas = 1.5,
        ),
        ions = ions, n_afm = N_AFM, logK = 1.0, c_ref = 1000.0,
        eps_mcp = 1.0e-4, lambda = 0.1,
    )

    sys = fvm_system(model, grid; reaction = true)
    iv = unknowns(sys)
    for i in 1:6
        iv[i, :] .= CIC[i]
        iv[i, 1] = CBC[i]
    end
    iv[7, :] .= 0.0
    iv[8, :] .= A.solve_dlm(CIC[1], CIC[2], CIC[3], CIC[4], 0.0, CIC[6], 635.0, 1.5; dlm = dlm)[1]
    iv[8, 1] = A.solve_dlm(CBC[1], CBC[2], CBC[3], CBC[4], 0.0, CBC[6], 635.0, 1.5; dlm = dlm)[1]
    ## Consistent start for the algebraic unknown: fully converted where the affinity is
    ## positive, absent where it is not.
    iv[9, :] .= 0.0
    iv[9, 1] = 1.0
    ctrl = VoronoiFVM.SolverControl(;
        Δt = 1.0e2, Δt_min = 1.0e-3, Δt_max = 1.0e5, Δu_opt = 50.0,
        damp_initial = 0.2, damp_growth = 1.2,
    )
    tsol = solve(sys; inival = iv, times = [0.0, 3.1536e6], control = ctrl)
    u = tsol.u[end]
    aff = [A.affinity(model, u[:, k]) for k in axes(u, 2)]

    @testset "all three branches occur in one solution" begin
        ## This is what makes the complementarity testable rather than merely plausible.
        converted = findall(x -> x > 0.99, u[9, :])
        absent = findall(x -> x < 1.0e-6, u[9, :])
        coexisting = findall(x -> 1.0e-3 < x < 0.999, u[9, :])
        @test !isempty(converted)
        @test !isempty(absent)
        @test !isempty(coexisting)

        ## Each branch obeys its own condition.
        @test all(k -> aff[k] > 0, converted)              # x̂ = 1 ⟹ 𝒜 ≥ 0
        @test all(k -> aff[k] < 1.0e-6, absent)            # x̂ = 0 ⟹ 𝒜 ≤ 0
        @test all(k -> abs(aff[k]) < 1.0e-3, coexisting)   # 0 < x̂ < 1 ⟹ 𝒜 = 0
    end

    @testset "the box is respected" begin
        ## Not by clamping the answer — nothing clamps `x̂` — but because the residual is
        ## only zero inside `[0, 1]`. The overshoot is the smoothing's, and it is small.
        @test minimum(u[9, :]) > -1.0e-6
        @test maximum(u[9, :]) < 1 + 1.0e-6
    end

    @testset "the exchange releases sulfate" begin
        ## The physical signature: `MS + 2Cl⁻ → FS + SO₄²⁻` puts sulfate into solution where
        ## the conversion happens. Sulfate is imposed low at the boundary and starts at 5
        ## in the bulk, so any node above 5 has been fed by the exchange and not by
        ## transport from either side.
        @test maximum(u[5, :]) > 2 * CIC[5]
        @test u[5, 1] ≈ CBC[5]
    end

    @testset "and the whole thing still conserves" begin
        worst, _, _ = conservation_defect(sys, tsol, [1, 2]; species = 1:6)
        @test worst < 1.0e-8
        ## Chloride carries the exchange on one side and the double layer on the other, so
        ## it is the one worth naming.
        wcl, _, _ = conservation_defect(sys, tsol, [1, 2]; species = 1:1)
        @test wcl < 1.0e-9
    end
end
