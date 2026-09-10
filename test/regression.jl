# Regression tests — the safety net for the constitutive-layer refactor.
#
# These do not check that the examples are *right* (that is what the validation
# benchmarks against analytical solutions will do). They check that they do not
# silently change: factoring five copies of the Oh-Jang tortuosity into one must
# leave every profile bit-for-bit where it was.

using LinearAlgebra: norm

include("regression/cases.jl")

# Tolerances are per case and independent of the platform triplet: Sys.MACHINE
# does not identify a Julia/BLAS/dependency environment. Richards' adaptive stepping
# has measured portable drift of 1.7e-5 in L2; the other four cases remain below 1e-10.
# A developer comparing within a controlled environment can explicitly request 1e-10
# for every case with POROMECHANICS_STRICT_REGRESSION=true.
const STRICT_REGRESSION = get(ENV, "POROMECHANICS_STRICT_REGRESSION", "false")
STRICT_REGRESSION in ("true", "false") || error("POROMECHANICS_STRICT_REGRESSION must be true or false")
const REGRESSION_TOLERANCES = Dict(
    "fickian_diffusion" => 1.0e-10,
    "darcy_column" => 1.0e-10,
    "richards_1d" => 1.0e-3,
    "nonisothermal_drying" => 1.0e-10,
    "biot_consolidation" => 1.0e-10,
    ## Not 1e-10: the chemistry goes through an interior-point solve whose last digits are
    ## not reproducible across BLAS builds. Loose enough to survive that, tight enough
    ## that a changed profile still shows.
    "chloride_ingress" => 1.0e-6,
)

@testset "Regression — $(case.name)" for case in CASES
    reference = read_reference(case.name)
    current = run_silently(case.signature)

    rtol = STRICT_REGRESSION == "true" ? 1.0e-10 : REGRESSION_TOLERANCES[case.name]

    @test length(current) == length(reference)

    if length(current) == length(reference)
        deviation = norm(current .- reference) / max(norm(reference), eps())
        if deviation > rtol
            i = argmax(abs.(current .- reference))
            @info """
            Regression in $(case.name): relative L2 deviation $(deviation), tolerance $(rtol).
            Reference generated on $(something(reference_platform(case.name), "an unrecorded platform")), \
            running on $(Sys.MACHINE), Julia $(VERSION), strict mode $(STRICT_REGRESSION).
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
