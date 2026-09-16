# Multi-ionic transport with a zero-current closure.
#
# Four separate questions, deliberately not one:
#
#   1. does it degenerate to Fick when it should — equal mobilities, or a single salt?
#   2. is the current actually zero?
#   3. does it stay electroneutral without anyone repairing it?
#   4. does it conserve, by the harness of `conservation.jl`?
#
# The first is the one that catches a sign error in the Bernoulli fitting, which is
# otherwise invisible: a flipped sign still gives a smooth, plausible profile.

using PoroMechanics
using VoronoiFVM
using ExtendableGrids
using LinearAlgebra: norm
using ForwardDiff
import OrdinaryDiffEq

module _NP
    include(joinpath(@__DIR__, "..", "..", "examples", "chloride_ingress", "nernst_planck.jl"))
end

@testset "Nernst-Planck with zero current" begin
    NP = _NP
    grid = simplexgrid(range(0, 0.05; length = 41))
    dx = 0.05 / 40
    ctrl = VoronoiFVM.SolverControl(;
        Δt = 1.0e2, Δt_min = 1.0e-4, Δt_max = 1.0e6, Δu_opt = 100.0
    )

    ## NaCl on a neutral background: z = (−1, +1), and the boundary values are equal, so
    ## the solution is electroneutral at the boundary and in the bulk.
    function nacl(; D_Cl = 2.032e-9, D_Na = 1.334e-9, c_bc = 523.0, c_ic = 1.0)
        m = NP.NernstPlanck(;
            phi = 0.121, D = (D_Cl, D_Na), z = (-1, 1),
            ## Without the tortuosity `D_eff` is four orders too large and the column
            ## reaches steady state long before `t_end` — at which point there is no
            ## concentration gradient, hence no diffusion potential, and the test would
            ## be measuring nothing. Same parameters as `run_4.jl`.
            tortuosity = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27),
            dirichlet = (((1, c_bc),), ((1, c_bc),), ((1, 0.0),)),
        )
        sys = fvm_system(m, grid; reaction = true)
        inival = unknowns(sys)
        inival[1, :] .= c_ic
        inival[2, :] .= c_ic
        inival[3, :] .= 0.0
        ## The boundary node starts at its imposed value. Otherwise the very first step
        ## carries the whole 1 → 523 jump whatever `Δt` is, and the step controller
        ## halves down to `Δt_min` without ever satisfying its criterion — a solver
        ## failure that looks like a physics failure.
        inival[1, 1] = c_bc
        inival[2, 1] = c_bc
        tsol = solve(sys; inival, times = [0.0, 3.1536e6], control = ctrl)
        return m, sys, tsol
    end

    @testset "equal mobilities reduce to Fick" begin
        ## With D_Cl == D_Na the two ions have no reason to separate, so the potential
        ## must stay flat and each profile must match a one-species Fick run exactly.
        m, sys, tsol = nacl(; D_Cl = 1.5e-9, D_Na = 1.5e-9)
        u = tsol.u[end]
        @test maximum(abs, u[3, :]) < 1.0e-9              # Ψ ≡ 0

        ## `FickModel`'s flux carries φ and this one does not — see the convention note
        ## in `nernst_planck.jl`. Compare against a Fick run whose D absorbs both that
        ## factor and the tortuosity.
        τ = tortuosity(OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27), 0.121, 1)
        mf = FickModel(; phi = 0.121, D = 1.5e-9 * τ / 0.121, dirichlet = ((1, 523.0),))
        sysf = fvm_system(mf, grid)
        iv = unknowns(sysf)
        iv .= 1.0
        iv[1, 1] = 523.0
        tf = solve(sysf; inival = iv, times = [0.0, 3.1536e6], control = ctrl)
        @test u[1, :] ≈ tf.u[end][1, :] rtol = 1.0e-8
    end

    @testset "the current is zero" begin
        ## Unequal mobilities: chloride is 1.5× more mobile than sodium, so without the
        ## potential the two profiles would separate and carry a net current.
        m, sys, tsol = nacl()
        u = tsol.u[end]
        cur, scale = NP.edge_current(m, u, dx)
        active = findall(>(1.0e-3 * maximum(scale)), scale)
        @test !isempty(active)
        @test maximum(abs.(cur[active]) ./ scale[active]) < 1.0e-10

        ## And the potential is genuinely doing work — a flat Ψ would mean the closure
        ## never engaged and the test above proves nothing.
        @test maximum(abs, u[3, :]) > 1.0e-3
    end

    @testset "electroneutrality is preserved, not repaired" begin
        m, sys, tsol = nacl()
        for k in eachindex(tsol.t)
            q = NP.net_charge(m, tsol.u[k])
            ## Relative to the concentration scale, not absolute.
            @test maximum(abs, q) < 1.0e-9 * 523.0
        end
    end

    @testset "and it conserves" begin
        ## Charge has no storage, so its row is excluded — `d/dt ∫T = Σ influx` is a
        ## statement about the ions, and including the constraint row reports its residual
        ## instead. See the note on `species` in `conservation_defect`.
        m, sys, tsol = nacl()
        worst, _, _ = conservation_defect(sys, tsol, [1, 2]; species = 1:2)
        @test worst < 1.0e-10
    end

    @testset "the diffusion potential is the analytical one" begin
        ## The validation, as opposed to the checks above. For a binary z:z salt the
        ## zero-current condition has a closed form: the two fluxes must be equal, which
        ## gives
        ##
        ##     ∇Ψ = (D₋ − D₊)/(D₋ + D₊) ∇ln c,      Ψ = r ln(c/c_bord)
        ##
        ## with r = 0.20737 for chloride against sodium. Ahead of the front the
        ## concentration is still the initial one, so Ψ plateaus at r·ln(c_ini/c_bord) and
        ## can be read off directly.
        r_th = (2.032 - 1.334) / (2.032 + 1.334)
        plateau(N) = let g = simplexgrid(range(0, 0.05; length = N))
            m = NP.NernstPlanck(;
                phi = 0.121, D = (2.032e-9, 1.334e-9), z = (-1, 1),
                tortuosity = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27),
                dirichlet = (((1, 523.0),), ((1, 523.0),), ((1, 0.0),)),
            )
            sys = fvm_system(m, g; reaction = true)
            iv = unknowns(sys)
            iv[1, :] .= 1.0
            iv[2, :] .= 1.0
            iv[3, :] .= 0.0
            iv[1, 1] = 523.0
            iv[2, 1] = 523.0
            c = VoronoiFVM.SolverControl(;
                Δt = 1.0e2, Δt_min = 1.0e-4, Δt_max = 1.0e5, Δu_opt = 20.0
            )
            sol = solve(sys; inival = iv, times = [0.0, 3.1536e6], control = c)
            sol.u[end][3, end] / log(1.0 / 523.0)
        end

        r41, r81 = plateau(41), plateau(81)
        @test r81 ≈ r_th rtol = 1.5e-2

        ## And it converges. Measured over 21…321 nodes the error goes
        ## −9.87, −3.51, −1.01, −0.26, −0.066 %, a ratio approaching four: second order in
        ## `dx`. Two points are enough to catch a scheme that has stopped converging.
        e41, e81 = abs(r41 - r_th), abs(r81 - r_th)
        @test e81 < e41 / 2.5
    end

    @testset "differentiating with respect to a diffusivity" begin
        ## This is the claim the whole plan rests on: the derivative of a transient
        ## multi-ionic solve with respect to a material parameter.
        ##
        ## It does not go through `VoronoiFVM.solve`. Its step-size update involves Δu,
        ## which carries partials, so the time becomes a `Dual` and cannot be stored:
        ## `MethodError(Float64, Dual(…))` at the second step, fixed steps included.
        ## `ODEProblem(sys, inival, tspan)` keeps VoronoiFVM for the space discretization
        ## and hands the time stepping to OrdinaryDiffEq, whose controller carries
        ## `Dual`s — the route of `examples/fickian_identification/run.jl`.
        ##
        ## The potential row has zero storage, so this is a DAE with a singular mass
        ## matrix, which `Rosenbrock23` accepts. The initial state is on the constraint
        ## manifold (`c_Cl == c_Na`, `q_background = 0`); an inconsistent one would have
        ## to be projected first.
        function front(D_Cl)
            T = typeof(D_Cl)
            m = NP.NernstPlanck(;
                phi = 0.121, D = (D_Cl, 1.334e-9), z = (-1, 1),
                tortuosity = OhJang(; phi_c = 0.18, n = 2.7, ds = 2.0e-4, tau_agg = 0.27),
                dirichlet = (((1, 523.0),), ((1, 523.0),), ((1, 0.0),)),
            )
            sys = fvm_system(m, grid; reaction = true, valuetype = T)
            iv = unknowns(sys; inival = zero(T))
            iv[1, :] .= 1.0
            iv[2, :] .= 1.0
            iv[3, :] .= 0.0
            iv[1, 1] = 523.0
            iv[2, 1] = 523.0
            problem = OrdinaryDiffEq.ODEProblem(sys, iv, (0.0, 3.1536e6))
            sol = OrdinaryDiffEq.solve(
                problem, OrdinaryDiffEq.Rosenbrock23();
                abstol = 1.0e-9, reltol = 1.0e-9, saveat = [3.1536e6],
            )
            u = reshape(sol, sys)
            return m, u[:, :, end]
        end
        inventory(D_Cl) = sum(front(D_Cl)[2][1, :]) * dx

        D0 = 2.032e-9
        m, u = front(D0)
        ## Something is measured: the potential is far from flat, so the derivative
        ## goes through the migration term and not only through a Fick law.
        @test maximum(abs, u[3, :]) > 0.5
        ## The constraint still holds along the adaptive integration.
        @test maximum(abs, NP.net_charge(m, u)) < 1.0e-8

        ## Measured: 1.2978504547e8 against 1.2978503344e8, a relative gap of 9.3e-8.
        ## The difference quotient carries its own truncation and solver error, so the
        ## tolerance is set above that, not at the round-off of either.
        h = 1.0e-4 * D0
        d_ad = ForwardDiff.derivative(inventory, D0)
        d_fd = (inventory(D0 + h) - inventory(D0 - h)) / 2h
        @test d_ad > 0
        @test d_ad ≈ d_fd rtol = 1.0e-5
    end
end
