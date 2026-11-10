# # What a fully coupled chemistry costs — the P0 bench
#
# `EquilibratedTransport` solves a certified ChemistryLab equilibrium at every node, at
# every Newton iteration, derivatives included. That is the target scheme of the chloride
# work and the one whose run time is the open question. This script fixes a case and
# measures, so that every optimisation of the coupling can be weighed against the same
# numbers:
#
# - wall time, and how much of it is spent inside the chemistry;
# - how many time steps the controller takes, and how many Newton iterations each needs;
# - how many equilibria are solved, with and without derivative seeds;
# - a conservation defect, so that a faster scheme cannot silently become a wrong one.
#
# The case: a 50 mm column of the OPC paste of `repro_equilibrated_transport.jl`, 41
# nodes, with a NaCl boundary on the left and 30 days of diffusion. The free-water
# diffusivities of the case are scaled by `1e-3`, which is the order of the Oh–Jang
# tortuosity of a concrete, so that the front moves a few millimetres as it would in the
# material rather than crossing the column.
#
# Run from the repository root:
#
# ```sh
# julia +1.13 --project=examples benchmarks/equilibrated_transport_cost.jl [nodes...]
# ```

using PoroMechanics, ChemistryLab, DynamicQuantities, OptimaSolver, ForwardDiff
using VoronoiFVM, ExtendableGrids
using LinearAlgebra: norm
using Printf

import PoroMechanics: equilibrium_state
include(joinpath(@__DIR__, "..", "examples", "chloride_ingress", "repro_equilibrated_transport.jl"))

# ## Counting the chemistry
#
# `speciate` calls `equilibrium_state(m, totals)`; wrapping the model's `initial_state`
# in a counter selects an overload that times the call and hands it on. The model is
# otherwise untouched, so what is measured is the scheme as shipped.

struct Counted{S}
    state::S
    calls::Vector{Int}      # [Float64 calls, Dual calls]
    seconds::Vector{Float64}
end
Counted(state) = Counted(state, zeros(Int, 2), zeros(2))

function equilibrium_state(
        m::EquilibratedTransport{C, S, R, P, TT, B, <:Counted}, totals,
    ) where {C, S, R, P, TT, B}
    inner = EquilibratedTransport(
        m.components, m.system, m.rows, m.phi, m.tortuosity, m.dirichlet, m.initial_state.state,
    )
    k = eltype(totals) <: ForwardDiff.Dual ? 2 : 1
    t0 = time_ns()
    result = equilibrium_state(inner, totals)
    m.initial_state.seconds[k] += (time_ns() - t0) / 1.0e9
    m.initial_state.calls[k] += 1
    return result
end

# ## Conservation
#
# The identity `d/dt ∫T = Σ influx` through VoronoiFVM's test functions — the harness of
# `test/reactive/conservation.jl`, reduced to what a benchmark needs.

function conservation_defect(sys, tsol, regions)
    factory = TestFunctionFactory(sys)
    tfs = [testfunction(factory, [q for q in regions if q != r], [r]) for r in regions]
    worst = 0.0
    for k in 2:length(tsol.t)
        u, uold = tsol.u[k], tsol.u[k - 1]
        Δt = tsol.t[k] - tsol.t[k - 1]
        Δt > 0 || continue
        stored(v) = vec(sum(VoronoiFVM.integrate(sys, sys.physics.storage, v); dims = 2))
        rate = (stored(u) .- stored(uold)) ./ Δt
        influx = sum(vec(VoronoiFVM.integrate(sys, tf, u, uold, Δt)) for tf in tfs)
        s = max(norm(rate), norm(influx))
        worst = max(worst, norm(rate .- influx) / max(s, eps()))
    end
    return worst
end

# ## The case

const DAY = 86_400.0

function bench_case(; nodes = 41, length_m = 0.05, boundary_nacl = 60.0)
    m0, totals = opc_equilibrated_case()
    cmp = m0.components
    components = ComponentSet(; names = cmp.names, z = cmp.z, D = cmp.D .* 1.0e-3, fixed = cmp.fixed)
    left = copy(totals)
    left[1] += boundary_nacl          # Cl⁻
    left[2] += boundary_nacl          # Na⁺, so the boundary stays neutral
    ## One tuple of `(region, value)` pairs per transported component, region 1 the inlet.
    dirichlet = Tuple(((1, left[i]),) for i in eachindex(totals))
    m = equilibrated_transport(components, m0.system; initial_state = m0.initial_state, dirichlet)
    counted = EquilibratedTransport(
        m.components, m.system, m.rows, m.phi, m.tortuosity, m.dirichlet, Counted(m.initial_state),
    )
    grid = simplexgrid(collect(range(0.0, length_m; length = nodes)))
    sys = fvm_system(counted, grid)
    inival = repeat(totals, 1, nodes)
    ## The inlet node starts at its boundary value. Otherwise the first update is the
    ## whole boundary jump whatever Δt, and the step controller shrinks Δt to Δt_min.
    inival[:, 1] .= left
    return counted, sys, inival
end

function run_bench(; nodes = 41, days = 30.0, Δu_opt = 20.0)
    m, sys, inival = bench_case(; nodes)
    control = VoronoiFVM.SolverControl(;
        Δt = 1.0e3, Δt_min = 1.0, Δt_max = 5DAY, Δu_opt, Δt_grow = 1.5,
        handle_exceptions = true, log = true, verbose = "",
    )
    ## One warm-up step so compilation stays out of the measurement.
    VoronoiFVM.solve(sys; inival, times = [0.0, 1.0e3], control)
    m.initial_state.calls .= 0
    m.initial_state.seconds .= 0.0

    elapsed = @elapsed tsol = VoronoiFVM.solve(sys; inival, times = [0.0, days * DAY], control)
    h = history(tsol)
    steps = length(h)
    newton = sum(length(x.updatenorm) for x in h; init = 0)
    ## Snapshots: the conservation check below evaluates fluxes too, hence equilibria,
    ## and those must not be charged to the solve.
    calls = copy(m.initial_state.calls)
    secs = copy(m.initial_state.seconds)
    chem = sum(secs)
    defect = conservation_defect(sys, tsol, [1, 2])
    final = tsol.u[end]
    front = count(>(inival[1, end] + 1.0), final[1, :])   # nodes the chloride has reached
    profile = final[1, :]                                  # total chloride, mol/m³ of medium

    return (;
        nodes, days, steps, newton, elapsed, chem, profile,
        calls_f64 = calls[1], calls_dual = calls[2],
        per_call_f64 = secs[1] / max(calls[1], 1), per_call_dual = secs[2] / max(calls[2], 1),
        defect, front,
    )
end

# ## Report

function report(r)
    per_step_node = r.elapsed / r.steps / r.nodes
    @printf("\n## %d nodes, %g days\n\n", r.nodes, r.days)
    println("| measure | value |\n|---|---|")
    @printf("| wall time | %.1f s |\n", r.elapsed)
    @printf("| time in the chemistry | %.1f s (%.0f %%) |\n", r.chem, 100r.chem / r.elapsed)
    @printf("| time steps | %d |\n", r.steps)
    @printf("| Newton iterations | %d (%.1f per step) |\n", r.newton, r.newton / r.steps)
    @printf("| equilibria, no seeds | %d at %.2f ms |\n", r.calls_f64, 1.0e3r.per_call_f64)
    @printf("| equilibria, with seeds | %d at %.2f ms |\n", r.calls_dual, 1.0e3r.per_call_dual)
    @printf("| equilibria per node per Newton iteration | %.1f |\n", (r.calls_f64 + r.calls_dual) / r.newton / r.nodes)
    @printf("| per step and node | %.3f s |\n", per_step_node)
    @printf("| conservation defect | %.1e |\n", r.defect)
    @printf("| nodes reached by chloride | %d of %d |\n", r.front, r.nodes)
    every = max(1, (r.nodes - 1) ÷ 8)
    @printf("| total chloride, every %d-th node | %s |\n", every,
            join((@sprintf("%.3g", v) for v in r.profile[1:every:end]), ", "))
    ## Extrapolation in the node count only: the chemistry is linear in it, and the step
    ## sequence is what the controller chose for this duration.
    @printf("| same duration, 400 nodes | %.0f min |\n", r.elapsed * 400 / r.nodes / 60)
    @printf("| same duration, 10 000 nodes (2D) | %.1f h |\n", r.elapsed * 10_000 / r.nodes / 3600)
    return nothing
end

# Arguments: `days=365` and `nodes=41,81`; defaults 30 days and 41 nodes.
if abspath(PROGRAM_FILE) == @__FILE__
    opts = Dict(split(a, "=") |> x -> (x[1] => x[2]) for a in ARGS)
    days = parse(Float64, get(opts, "days", "30"))
    sizes = parse.(Int, split(get(opts, "nodes", "41"), ","))
    println("ChemistryLab ", pkgversion(ChemistryLab), ", OptimaSolver ", pkgversion(OptimaSolver),
            ", VoronoiFVM ", pkgversion(VoronoiFVM), ", Julia ", VERSION, ", threads ", Threads.nthreads())
    for n in sizes
        report(run_bench(; nodes = n, days))
    end
end
