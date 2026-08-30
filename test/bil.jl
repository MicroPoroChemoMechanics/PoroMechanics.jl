# Cross-code comparison against Bil.
#
# `test/regression` asks "did *our* answer change?" and `test/benchmarks.jl` asks "is our
# answer right?" against closed forms. This asks a third question: **do two independently
# written codes agree on a problem neither has a closed form for?**
#
# Bil is not installed on CI and never will be — it is GPL-3.0, built with CMake, and wants
# HSL/SuperLU. Two consequences, and they are different:
#
#  * **The Bil half is frozen.** `test/bil/generate.jl` reads Bil's shipped outputs once and
#    writes `references/<name>.txt`; nothing here ever runs the binary.
#  * **Our half still needs Bil's *inputs*.** The whole point of these cases is to solve the
#    same problem on the same mesh with the same tabulated curve, and those files live in
#    Bil's tree. They are not vendored into this repository: copying GPL-3.0 data into an MIT
#    package is not a licensing question worth having, and a mesh copied out of another
#    project drifts from it silently.
#
# So this testset runs where `BIL_ROOT` is present and skips where it is not. That is not a
# weakness of the frozen references — they still remove the binary, the runtime and the
# version dependence, and they make the comparison reviewable in a diff.
#
# Tolerances come from the case registry, one per case, each with a recorded reason. They
# are not 1e-10 and cannot be: Bil prints seven significant digits, builds its Jacobian by
# finite differences with a deck-tuned perturbation, and takes different time steps. See the
# header of `test/bil/cases.jl`.

using LinearAlgebra: norm

include("bil/cases.jl")

## A case runs when both halves are available: the frozen reference, and Bil's source tree
## for the mesh and curve our side reads. Missing either is a skip with a reason, not a
## failure — the common case is a CI runner, where neither is a defect.
const BIL_RUNNABLE = if bil_root() === nothing
    @info """
    BIL_ROOT not set and no Bil checkout found — skipping the Bil comparison.
    These cases solve Bil's own decks on Bil's own meshes, which are not vendored here
    (Bil is GPL-3.0, this package is MIT). To run them:
      BIL_ROOT=/path/to/bil-master julia --project -e 'using Pkg; Pkg.test()'
    """
    BilCase[]
else
    runnable = filter(c -> isfile(bil_reference_path(c.name)), BIL_CASES)
    if isempty(runnable)
        @info """
        Bil checkout found, but no frozen references in test/bil/references.
        Create them with:
          julia --project test/bil/generate.jl
        """
    end
    runnable
end

@testset "Bil — $(case.name)" for case in BIL_RUNNABLE
    reference = read_bil_reference(case.name)
    ## Muted like the regression cases: a solve prints a convergence table nobody wants
    ## between two test results.
    current = redirect_stdout(case.ours, devnull)

    @test length(current) == length(reference)

    if length(current) == length(reference)
        deviation = norm(current .- reference) / max(norm(reference), eps())
        if deviation > case.rtol
            i = argmax(abs.(current .- reference))
            @info """
            $(case.name): relative L2 deviation from Bil is $(deviation), tolerance $(case.rtol).
            Largest single deviation at index $i:
              Bil  = $(reference[i])
              ours = $(current[i])

            Before regenerating, work out which code moved. `test/bil/generate.jl --drift`
            re-runs Bil and compares it to its own shipped output; if that is clean, the
            change is ours.

            Why this tolerance: $(strip(case.why))
            """
        end
        @test deviation ≤ case.rtol
    end
end
