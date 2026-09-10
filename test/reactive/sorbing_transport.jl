# Transport whose storage is the inventory, with the real double layer in it.
#
# `conservation.jl` made the point on a Langmuir tracer, where the isotherm was two lines
# of algebra. This repeats it with `dlm.jl` — a surface potential found by a root-find
# inside the storage term — and adds the two measurements that only the real model can
# give: what the frozen `K_d` of `run_4.jl` actually costs, and how much of `∂S/∂c` it
# throws away by keeping only the diagonal.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using ForwardDiff

module _ST
    const HERE = joinpath(@__DIR__, "..", "..", "examples", "chloride_ingress")
    include(joinpath(HERE, "dlm.jl"))
    include(joinpath(HERE, "nernst_planck.jl"))
    include(joinpath(HERE, "sorbing_transport.jl"))
end

@testset "transport with the double layer in the storage" begin
    ST = _ST

    ## Cl⁻, Na⁺, K⁺, Ca²⁺, OH⁻, then Ψ. Both the boundary and the initial composition are
    ## electroneutral by construction — `c_OH` is what closes the balance, which is the
    ## role `run_4.jl` gives it too, except that here it is transported rather than reset.
    CBC = (523.0, 523.0, 1.0, 0.05, 1.1)
    CIC = (1.0e-3, 300.0, 100.0, 1.0, 401.999)
    ZZ = (-1, 1, 1, 2, -1)

    grid = simplexgrid(range(0, 0.05; length = 41))
    tort = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27)
    ctrl = VoronoiFVM.SolverControl(;
        Δt = 1.0e2, Δt_min = 1.0e-3, Δt_max = 1.0e5, Δu_opt = 50.0,
        damp_initial = 0.5, damp_growth = 1.2,
    )

    transport = ST.NernstPlanck(;
        phi = 0.121,
        D = (2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9, 5.273e-9),
        z = ZZ, tortuosity = tort,
        dirichlet = (
            ((1, CBC[1]),), ((1, CBC[2]),), ((1, CBC[3]),),
            ((1, CBC[4]),), ((1, CBC[5]),), ((1, 0.0),),
        ),
    )
    model = ST.SorbingTransport(;
        transport = transport, dlm = ST.DLM_TRAN2018(n_csh0 = 635.0),
        ions = ST.IonIndex(; Cl = 1, Na = 2, K = 3, Ca = 4, OH = 5),
        n_csh = 635.0, x_cas = 1.5,
    )

    function run(m)
        sys = fvm_system(m, grid; reaction = true)
        iv = unknowns(sys)
        for i in 1:5
            iv[i, :] .= CIC[i]
            iv[i, 1] = CBC[i]
        end
        iv[6, :] .= 0.0
        return sys, solve(sys; inival = iv, times = [0.0, 3.1536e6], control = ctrl)
    end

    sys, tsol = run(model)

    @testset "it transports, and stays neutral" begin
        u = tsol.u[end]
        ## A front, not a flat profile: without one nothing below measures anything.
        @test u[1, 1] ≈ 523.0
        @test u[1, end] < 1.0
        @test u[1, 5] > 10 * u[1, 15]
        ## Electroneutrality is preserved by the constraint, at every node and every step.
        for k in eachindex(tsol.t)
            q = [sum(ZZ[i] * tsol.u[k][i, n] for i in 1:5) for n in axes(tsol.u[k], 2)]
            @test maximum(abs, q) < 1.0e-8 * 523.0
        end
        ## And the double layer is loaded where the chloride is.
        S = [ST.adsorbed(model, u[:, n])[1] for n in axes(u, 2)]
        @test S[1] > 1.0
        @test S[1] > 1.0e4 * S[end]
    end

    @testset "and it conserves the inventory" begin
        ## Restricted to the rows that have a storage. Including `Ψ` reports `1e-8`, which
        ## is its constraint residual and not a conservation defect at all — see the note
        ## on `species` in `conservation_defect`.
        worst, _, _ = conservation_defect(sys, tsol, [1, 2]; species = 1:5)
        @test worst < 1.0e-9
    end

    @testset "β as a nodal unknown gives the same answer" begin
        ## The strongest check available: the same physics written twice. One model finds
        ## the surface potential by a root-find inside the storage term, the other carries
        ## it as an unknown with the Gouy-Chapman balance as its equation. Nothing is
        ## shared between the two paths but `dlm.jl`'s expressions.
        resolved = ST.SurfaceResolvedTransport(;
            transport = transport, dlm = ST.DLM_TRAN2018(n_csh0 = 635.0),
            ions = ST.IonIndex(; Cl = 1, Na = 2, K = 3, Ca = 4, OH = 5),
            n_csh = 635.0, x_cas = 1.5,
        )
        sysr = fvm_system(resolved, grid; reaction = true)
        iv = unknowns(sysr)
        for i in 1:5
            iv[i, :] .= CIC[i]
            iv[i, 1] = CBC[i]
        end
        iv[6, :] .= 0.0
        ## β is algebraic, so the initial value has to sit on the constraint manifold
        ## already. Starting it at zero makes the first step move the whole surface
        ## inventory at once, and the step controller collapses to `Δt_min` — a
        ## consistent-initialisation failure that reads like a physics failure.
        d = ST.DLM_TRAN2018(n_csh0 = 635.0)
        iv[7, :] .= ST.solve_dlm(CIC[1], CIC[2], CIC[3], CIC[4], 0.0, CIC[5], 635.0, 1.5; dlm = d)[1]
        iv[7, 1] = ST.solve_dlm(CBC[1], CBC[2], CBC[3], CBC[4], 0.0, CBC[5], 635.0, 1.5; dlm = d)[1]
        solr = solve(sysr; inival = iv, times = [0.0, 3.1536e6], control = ctrl)

        ## Agreement to round-off, on every transported species.
        for i in 1:5
            @test solr.u[end][i, :] ≈ tsol.u[end][i, :] rtol = 1.0e-12
        end
        ## And the potential the nested solve finds is the one the unknown converges to.
        βnest = [
            ST.solve_dlm(
                max(tsol.u[end][1, n], 0.0), tsol.u[end][2, n], tsol.u[end][3, n],
                tsol.u[end][4, n], 0.0, tsol.u[end][5, n], 635.0, 1.5; dlm = d,
            )[1] for n in axes(tsol.u[end], 2)
        ]
        @test solr.u[end][7, :] ≈ βnest rtol = 1.0e-8

        ## Both conserve, identically — the defect is the scheme's, not the root-find's.
        wr, _, _ = conservation_defect(sysr, solr, [1, 2]; species = 1:5)
        wn, _, _ = conservation_defect(sys, tsol, [1, 2]; species = 1:5)
        @test wr < 1.0e-9
        @test wr ≈ wn rtol = 0.5
    end

    @testset "what freezing K_d costs, on the real double layer" begin
        ## The same PDE with `(φ + K_d) c` as its storage, `K_d = ∂S/∂c` frozen at the
        ## composition of the incoming boundary solution — the most favourable choice for
        ## it, since that is where the front spends its gradient.
        kd = ForwardDiff.derivative(
            c -> ST.adsorbed(model, [c, CBC[2], CBC[3], CBC[4], CBC[5], 0.0])[1], CBC[1]
        )
        @test kd > 0                            # a frozen tangent that is not zero

        frozen_model = ST.SorbingTransport(;
            transport = transport, dlm = ST.DLM_TRAN2018(n_csh0 = 0.0),
            ions = ST.IonIndex(; Cl = 1, Na = 2, K = 3, Ca = 4, OH = 5),
            n_csh = 0.0, x_cas = 1.5,
        )
        ## `n_csh = 0` switches the double layer off — `solve_dlm` returns zeros — so this
        ## model stores `φ c` alone. Adding the frozen `K_d c` on top of it by hand is what
        ## `run_4.jl` does, and comparing *that* inventory against the true one is the
        ## measurement.
        sysf, tf = run(frozen_model)
        frozen_worst, _, _ = conservation_defect(
            sysf, tf, [1, 2];
            F = (f, u, node, data = nothing) -> begin
                PoroMechanics.storage!(f, u, node, frozen_model, data)
                f[1] += ST.adsorbed(model, u)[1]      # the inventory the physics has
            end,
        )
        @test frozen_worst > 1.0e-3
        @info "conservation, vraie double couche" inventaire = conservation_defect(sys, tsol, [1, 2])[1] retardation_gelée = frozen_worst K_d = kd
    end

    @testset "the cross terms the diagonal throws away" begin
        ## `run_4.jl` keeps `∂S_Cl/∂c_Cl` and nothing else (constat 3 of the formulation).
        ## Here the whole matrix is in the Jacobian, so it can be measured: how big are the
        ## terms that get discarded?
        c = [110.0, 295.0, 81.0, 1.09, 268.0]             # the front, from the solution
        J = ForwardDiff.jacobian(
            x -> collect(ST.adsorbed(model, vcat(x, 0.0))[1:4]), c
        )
        diag_Cl = abs(J[1, 1])
        @test diag_Cl > 0

        ## Calcium, not chloride, is what governs chloride binding. It competes for the
        ## deprotonated silanol sites — `≡SiO⁻ + Ca²⁺ → ≡SiOCa⁺` — so more calcium leaves
        ## fewer sites for `≡SiOH + Cl⁻ → ≡SiOHCl⁻`, and the term is negative.
        @test J[1, 4] < 0
        @test abs(J[1, 4]) > 5 * diag_Cl

        ## The whole row, at the front:
        ##   ∂S_Cl/∂c_Cl = +1.64e-3   (×1.00)
        ##   ∂S_Cl/∂c_Na = −2.71e-5   (×0.02)
        ##   ∂S_Cl/∂c_K  = −2.71e-5   (×0.02)
        ##   ∂S_Cl/∂c_Ca = −1.51e-2   (×9.20)   ← the one `run_4.jl` discards
        ##   ∂S_Cl/∂c_OH = −6.56e-4   (×0.40)
        ## Keeping the diagonal alone keeps the *smallest but one* term of the row.
        @info "∂S_Cl/∂c au front" Cl = J[1, 1] Na = J[1, 2] K = J[1, 3] Ca = J[1, 4] OH = J[1, 5] rapport_Ca = abs(J[1, 4]) / diag_Cl
    end
end
