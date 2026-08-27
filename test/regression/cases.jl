# Regression case registry.
#
# Each example is reduced to a deterministic vector of numbers — its "signature".
# The signature is recorded once (see `generate.jl`) and compared on every test
# run, so that a refactor of the constitutive layer cannot silently change a
# result.
#
# Two constraints shape the reduction:
#
#  1. It must not depend on the adaptive time stepping.  `VoronoiFVM.solve`
#     returns *every* internal step, and their number changes with the solver
#     version or the tolerance (richards_1d alone takes ~3400 steps for 6 output
#     times).  Transient solutions are therefore interpolated at fixed fractions
#     of the final time rather than recorded step by step.
#
#  2. Each example is included in its own module.  Several of them define
#     top-level helpers under the same names — `_Sl` and `_krl` exist in both
#     `richards_1d/run.jl` and `nonisothermal_drying/run.jl` — and the examples install
#     methods on the shared `PoroMechanics` interface.  Module isolation keeps
#     the helpers apart; the interface methods dispatch on distinct model types
#     and coexist safely.

using Printf: @printf

const REFERENCE_DIR = joinpath(@__DIR__, "references")

"Fractions of the final time at which a transient solution is sampled."
const PROBE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

"""
    sample_transient(tsol) -> Vector{Float64}

Flatten a `VoronoiFVM.TransientSolution` into a signature vector by interpolating
it at `PROBE_FRACTIONS` of the final time. Independent of the number of steps the
adaptive solver actually took.
"""
function sample_transient(tsol)
    t_end = tsol.t[end]
    return reduce(vcat, (vec(tsol(f * t_end)) for f in PROBE_FRACTIONS))
end

# ── The examples ──────────────────────────────────────────────────────────────
# Included from this file, so the relative paths resolve against `test/regression/`.

module _FickianDiffusion
    include("../../examples/fickian_diffusion/run.jl")
end

module _DarcyColumn
    include("../../examples/darcy_column/run.jl")
end

module _Richards1D
    include("../../examples/richards_1d/run.jl")
end

module _NonisothermalDrying
    include("../../examples/nonisothermal_drying/run.jl")
end

module _BiotConsolidation
    include("../../examples/biot_consolidation/run.jl")
end

# ── Case registry ─────────────────────────────────────────────────────────────

"""
    RegressionCase(name, signature)

`signature()` runs the example and returns the vector of numbers to compare.
`name` is both the test label and the reference file stem.
"""
struct RegressionCase
    name::String
    signature::Function
end

"""
    CASES

The examples covered by the harness. `richards_2d` is absent on purpose: it is a
bare top-level script with no entry function, and it pulls in Plots, Triangulate
and SimplexGridFactory, which are not dependencies of the package. It joins the
harness when it is converted to a Literate script with a `run_*` function.
"""
const CASES = [
    RegressionCase(
        # A Literate script runs its own simulation at include time and leaves the
        # result at module level. Reading it back, rather than calling `run_*` again,
        # halves the cost and guarantees the tests check the very object the
        # documentation page displays.
        "fickian_diffusion",
        () -> sample_transient(_FickianDiffusion.tsol),
    ),
    RegressionCase(
        "darcy_column",
        () -> sample_transient(_DarcyColumn.tsol),
    ),
    RegressionCase(
        "richards_1d",
        () -> sample_transient(_Richards1D.tsol),
    ),
    RegressionCase(
        # `results` is already a vector of (time, profile) pairs at ten fixed
        # output times, so it needs no resampling — but the times themselves are
        # part of the signature.
        "nonisothermal_drying",
        () -> let mod = _NonisothermalDrying
            reduce(
                vcat,
                vcat([mod.x_all], [vcat(t, vec(u)) for (t, u) in mod.results]),
            )
        end,
    ),
    RegressionCase(
        # Fixed mesh, one solve: the raw dof vector is already deterministic.
        "biot_consolidation",
        () -> _BiotConsolidation.result.x,
    ),
]

# ── Reference file I/O ────────────────────────────────────────────────────────
# Plain text at full precision rather than a binary format: the references are
# reviewable in a diff, which matters for a repository meant to be peer reviewed.

reference_path(name) = joinpath(REFERENCE_DIR, name * ".txt")

function write_reference(name, values)
    mkpath(REFERENCE_DIR)
    open(reference_path(name), "w") do io
        println(io, "# PoroMechanics.jl regression reference — ", name)
        println(io, "# Regenerate with: julia --project test/regression/generate.jl")
        println(io, "# ", length(values), " values, printed with %.17g")
        for v in values
            @printf(io, "%.17g\n", v)
        end
    end
    return reference_path(name)
end

function read_reference(name)
    path = reference_path(name)
    isfile(path) || error(
        "missing reference for \"$name\" at $path — " *
            "run `julia --project test/regression/generate.jl` to create it"
    )
    return [parse(Float64, l) for l in eachline(path) if !startswith(l, "#") && !isempty(strip(l))]
end

"""
    run_silently(f)

Run `f()` with stdout muted. The examples print progress tables that would bury
the test output.
"""
run_silently(f) = redirect_stdout(f, devnull)
