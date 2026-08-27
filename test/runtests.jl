using Test
using PoroMechanics

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
    include("constitutive.jl")

    # ── The Barcelona Basic Model ──────────────────────────────────────────────
    include("bbm.jl")

    # ── Validation against closed-form solutions ──────────────────────────────
    include("benchmarks.jl")

    # ── Examples still produce the profiles they used to ───────────────────────
    include("regression.jl")

end
