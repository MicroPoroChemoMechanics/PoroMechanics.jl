# Regression tests — the safety net for the constitutive-layer refactor.
#
# These do not check that the examples are *right* (that is what the validation
# benchmarks against analytical solutions will do). They check that they do not
# silently change: factoring five copies of the Oh-Jang tortuosity into one must
# leave every profile bit-for-bit where it was.

using LinearAlgebra: norm

include("regression/cases.jl")
include("regression/drying.jl")

# Tolerances are per case and independent of the platform triplet: Sys.MACHINE
# does not identify a Julia/BLAS/dependency environment. Richards' adaptive stepping
# has measured portable drift of 1.7e-5 in L2, drying 3.29e-8, and chloride ingress
# 1.0e-4 (see below); the remaining cases retain the 1e-10 tolerance.
# A developer comparing within a controlled environment can explicitly request 1e-10
# for every case with POROMECHANICS_STRICT_REGRESSION=true.
const STRICT_REGRESSION = get(ENV, "POROMECHANICS_STRICT_REGRESSION", "false")
STRICT_REGRESSION in ("true", "false") || error("POROMECHANICS_STRICT_REGRESSION must be true or false")
const REGRESSION_TOLERANCES = Dict(
    "fickian_diffusion" => 1.0e-10,
    "darcy_column" => 1.0e-10,
    "richards_1d" => 1.0e-3,
    ## All four Linux/Windows jobs (Julia 1.12.7 and 1.13.0) measured the same
    ## 3.2864e-8 drift from the macOS ARM reference, at most 47.56 Pa in one entry.
    ## Keep a modest margin for environment differences; see regression/README.md.
    "nonisothermal_drying" => 1.0e-7,
    "biot_consolidation" => 1.0e-10,
    ## Certified chemistry still has solver and phase-boundary tolerances, and the
    ## signature moves with the environment by more than certification alone controls.
    ## Measured against this Mac-generated reference: 1.0148565190861922e-4 on
    ## ubuntu-latest and 1.0148565194436454e-4 on windows-latest, in two CI runs two days
    ## apart, bit-identical per platform — reproducible, not noise. A third run of the
    ## same commit range came back under 1e-6, and the only manifest difference was
    ## DispatchDoctor 0.4.28 → 0.4.29 with DomainSets 0.8.1 → 0.8.2, neither of which
    ## carries physics: they move specialization, hence the last bits, which the
    ## transport-chemistry coupling amplifies. 1e-6 therefore reports the resolver's
    ## mood rather than this package's behavior. The threshold records the measured
    ## spread with one decade of margin; tighten it once the amplification is understood.
    "chloride_ingress" => 1.0e-3,
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

## A case that could not run must say so. Silence would read as coverage.
if GMSH_FAILURE !== nothing
    @testset "Regression — biot_consolidation" begin
        @warn """
        Skipped: Gmsh would not initialize, so the mesh could not be read and the example
        never ran. Measured on windows-latest with Julia 1.13.0, where Julia 1.12.7 on the
        same runner reads the same mesh — this is the Gmsh binary artifact, not the model.
        Running on $(Sys.MACHINE), Julia $(VERSION).
          $(GMSH_FAILURE)
        """
        @test_skip false
    end
end
