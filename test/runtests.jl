using Test
using PoroMechanics

const TEST_GROUPS = isempty(ARGS) ? Set(["core", "validation", "regression", "chemistry", "bil"]) : Set(ARGS)
issubset(TEST_GROUPS, Set(["core", "validation", "regression", "chemistry", "bil"])) ||
    error("Unknown test group: choose core, validation, regression, chemistry, or bil")

@testset "PoroMechanics.jl" begin

    # ── Package loads correctly ────────────────────────────────────────────────
    @testset "Imports" begin
        @test isdefined(PoroMechanics, :AbstractPoroModel)
        @test isdefined(PoroMechanics, :AbstractPoroSolver)
        @test isdefined(PoroMechanics, :storage!)
        @test isdefined(PoroMechanics, :flux!)
        @test isdefined(PoroMechanics, :bcondition!)
        @test isdefined(PoroMechanics, :assemble_element!)
    end

    # ── Abstract interface stubs raise informative errors ─────────────────────
    @testset "Interface stubs" begin
        struct _DummyModel <: AbstractPoroModel end
        m = _DummyModel()

        # stub methods should throw (not silently succeed)
        @test_throws ErrorException storage!(zeros(1), zeros(1), nothing, m, nothing)
        @test_throws ErrorException flux!(zeros(1), zeros(2), nothing, m, nothing)
        @test_throws ErrorException assemble_element!(zeros(2, 2), zeros(2), nothing, zeros(2), m, nothing, 1.0)

        # bcondition! default is a no-op (no error)
        @test isnothing(bcondition!(zeros(1), zeros(1), nothing, m, nothing))
    end

    # ── Constitutive layer: values and differentiability ──────────────────────
    if "core" in TEST_GROUPS
        include("constitutive.jl")

        # ── The Barcelona Basic Model ──────────────────────────────────────────────
        include("bbm.jl")
        include("newton.jl")

        # ── Drucker-Prager ─────────────────────────────────────────────────────────
        include("druckerprager.jl")

        # ── Computational homogenization on a periodic cell ────────────────────────
        include("homogenization.jl")

        # ── The transport models the package ships ────────────────────────────────
        include("models.jl")
    end

    # ── Validation against closed-form solutions ──────────────────────────────
    if "validation" in TEST_GROUPS
        include("benchmarks.jl")
        include("differentiability.jl")
    end

    # ── Examples still produce the profiles they used to ───────────────────────
    "regression" in TEST_GROUPS && include("regression.jl")

    # ── The dialog with ChemistryLab ─────────────────────────────────────────
    "chemistry" in TEST_GROUPS && include("chemistry_interface.jl")

    # ── Agreement with Bil, an independently written code ─────────────────────
    "bil" in TEST_GROUPS && include("bil.jl")

end
