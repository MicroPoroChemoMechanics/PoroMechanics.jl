# The dialogue with ChemistryLab.
#
# Not a test of the chemistry — whether portlandite has the right solubility is that
# package's business and is validated there. This tests the interface: that this package
# builds a well-formed system, and that what it puts into an equilibration comes back out
# of it. Both statements are independent of the thermodynamic data and of the version.

using ChemistryLab
using OptimaSolver
using DynamicQuantities

module _Balance
    using ChemistryLab, DynamicQuantities
    include("../examples/chloride_ingress/element_balance.jl")
end

@testset "ChemistryLab interface" begin
    db = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    substances = build_species(db)
    by_name = Dict(symbol(s) => s for s in substances)

    ## The construction every chloride example uses: seed a speciation with the phases of
    ## interest, then append the solids.
    seeds = split("Portlandite ettringite monosulphate12 H2O@ Ca+2 OH- Cl- Na+ K+ H+ Al+3 SO4-2")
    aqueous = speciation(substances, seeds; aggregate_state = [AS_AQUEOUS])
    solids = [by_name[n] for n in ("Portlandite", "ettringite", "monosulphate12")]
    offered = vcat(collect(aqueous), solids)

    @testset "the aqueous filter does not filter the seeds" begin
        ## `speciation` returns a seeded species whether or not it matches
        ## `aggregate_state`. This is the whole reason the deduplication exists, so it is
        ## asserted rather than assumed: if it ever stops being true, the guard below
        ## becomes dead weight and should be reconsidered rather than left in place.
        leaked = [string(symbol(s)) for s in aqueous if s.aggregate_state != AS_AQUEOUS]
        @test "Portlandite" in leaked
    end

    @testset "duplicates are found, and removed" begin
        ## Appending the solids therefore lists them twice. A duplicated column is a
        ## perfectly valid optimisation problem whose answer is meaningless, so it has to
        ## be caught here rather than diagnosed later as missing matter.
        @test !isempty(_Balance.duplicate_species(offered))
        @test "Portlandite" in _Balance.duplicate_species(offered)
        @test isempty(_Balance.duplicate_species(_Balance.unique_species(offered)))
        @test length(_Balance.unique_species(offered)) == length(offered) - 3
    end

    @testset "equilibrate conserves the elements it is given" begin
        ## The contract of the dialogue, and the only thing here that actually runs a
        ## solve. A mature-OPC pore solution: portlandite in excess, monosulphate, alkalis.
        system = ChemicalSystem(_Balance.unique_species(offered), CEMDATA_PRIMARIES)

        V_rev, φ = 1.0e-3, 0.121
        m_clinker = 350.0 * V_rev
        n_Na2O = 0.0016 * m_clinker / 61.98e-3
        n_K2O = 0.0049 * m_clinker / 94.20e-3

        state = ChemicalState(system; T = 293.15 * us"K")
        set_quantity!(state, "H2O@", (φ * V_rev * 55_500.0 - n_Na2O - n_K2O) * us"mol")
        set_quantity!(state, "Na+", 2n_Na2O * us"mol")
        set_quantity!(state, "K+", 2n_K2O * us"mol")
        set_quantity!(state, "OH-", (2n_Na2O + 2n_K2O) * us"mol")
        set_quantity!(state, "Cl-", 1.0e-16 * us"mol")
        set_quantity!(state, "SO4-2", 1.0e-16 * us"mol")
        set_quantity!(state, "Portlandite", 1.640 * us"mol")
        set_quantity!(state, "monosulphate12", 0.100 * us"mol")

        equilibrated = equilibrate(state, OptimaOptimizer(tol = 1.0e-10, verbose = false))

        ## The threshold is what this test can honestly claim. An interface breaking —
        ## a duplicated column, a unit slip, a species that cannot be read back — loses
        ## tens of percent of an element. The round trip currently closes to ~1e-3 on
        ## sulphur, a residue that belongs to the equilibrium solve and not to the
        ## marshalling, so the bound sits above it instead of pretending to measure it.
        element, err = _Balance.element_balance_error(state, equilibrated, system)
        @test err < 1.0e-2

        ## And the round trip is usable on the other side: a liquid volume to convert moles
        ## back into the concentrations transport works in. Zero would divide by zero at the
        ## first node, which is how this fails in practice rather than by raising.
        V_liq = ustrip(uconvert(us"m^3", equilibrated.V_phases[].liquid))
        @test V_liq > 0
        @test V_liq < φ * V_rev * 1.5      # a pore volume, not the whole REV
    end
end
