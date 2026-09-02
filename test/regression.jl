# Regression tests — the safety net for the constitutive-layer refactor.
#
# These do not check that the examples are *right* (that is what the validation
# benchmarks against analytical solutions will do). They check that they do not
# silently change: factoring five copies of the Oh-Jang tortuosity into one must
# leave every profile bit-for-bit where it was.

using LinearAlgebra: norm

include("regression/cases.jl")

# Two tolerances, because "unchanged code gives an unchanged answer" is a statement about
# one machine, not about the world.
#
# On the machine that wrote the reference, re-running unchanged code is bit-identical and
# 1e-10 only absorbs reordering inside the linear solver. Anything larger is a real change,
# and that is the tolerance a refactor has to survive.
#
# Elsewhere it cannot hold. A different BLAS, a different `libm` (`exp`, `log` and `pow`
# differ in the last ulp) and different FMA contraction move the first Newton step by a few
# ulp; the adaptive controller then accepts a slightly different sequence of steps, and the
# profile interpolated at a fixed time carries that sequence's truncation error rather than
# the reference's. `sample_transient` makes the signature independent of the *number* of
# steps, not of where they fall.
#
# Measured, not guessed: on x86-64 Windows, `richards_1d` — the stiffest of the five, ~3400
# internal steps — deviates by 1.7e-5 in relative L2, with a worst single point at 3.2e-4.
# The other four stay under 1e-10 even there. A regression that matters, such as a
# constitutive law quietly changing, moves a profile by percent, so 1e-3 still catches it
# with three orders of magnitude to spare.
const REGRESSION_RTOL = 1.0e-10          # same machine as the reference
const REGRESSION_RTOL_PORTABLE = 1.0e-3  # anywhere else

@testset "Regression — $(case.name)" for case in CASES
    reference = read_reference(case.name)
    current = run_silently(case.signature)

    ## A reference with no recorded platform predates the header and cannot be claimed as
    ## same-machine, so it is compared portably.
    native = reference_platform(case.name) == Sys.MACHINE
    rtol = native ? REGRESSION_RTOL : REGRESSION_RTOL_PORTABLE

    @test length(current) == length(reference)

    if length(current) == length(reference)
        deviation = norm(current .- reference) / max(norm(reference), eps())
        if deviation > rtol
            i = argmax(abs.(current .- reference))
            @info """
            Regression in $(case.name): relative L2 deviation $(deviation), tolerance $(rtol).
            Reference generated on $(something(reference_platform(case.name), "an unrecorded platform")), \
            running on $(Sys.MACHINE) — $(native ? "same machine" : "not the same machine").
            Largest single deviation at index $i:
              reference = $(reference[i])
              current   = $(current[i])
            If this change is intended, regenerate with
              julia --project test/regression/generate.jl $(case.name)
            """
        end
        @test deviation ≤ rtol
    end
end
