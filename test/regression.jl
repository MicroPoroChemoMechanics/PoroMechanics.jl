# Regression tests — the safety net for the constitutive-layer refactor.
#
# These do not check that the examples are *right* (that is what the validation
# benchmarks against analytical solutions will do). They check that they do not
# silently change: factoring five copies of the Oh-Jang tortuosity into one must
# leave every profile bit-for-bit where it was.

using LinearAlgebra: norm

include("regression/cases.jl")

# Re-running unchanged code is normally bit-identical; the tolerance only absorbs
# reordering inside the linear solver. Anything larger is a real change.
const REGRESSION_RTOL = 1.0e-10

@testset "Regression — $(case.name)" for case in CASES
    reference = read_reference(case.name)
    current = run_silently(case.signature)

    @test length(current) == length(reference)

    if length(current) == length(reference)
        deviation = norm(current .- reference) / max(norm(reference), eps())
        if deviation > REGRESSION_RTOL
            i = argmax(abs.(current .- reference))
            @info """
            Regression in $(case.name): relative L2 deviation $(deviation).
            Largest single deviation at index $i:
              reference = $(reference[i])
              current   = $(current[i])
            If this change is intended, regenerate with
              julia --project test/regression/generate.jl $(case.name)
            """
        end
        @test deviation ≤ REGRESSION_RTOL
    end
end
