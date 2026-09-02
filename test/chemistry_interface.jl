# The dialog with ChemistryLab.
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

# ── Is the answer an equilibrium at all? ──────────────────────────────────────
#
# The element balance below asks whether `equilibrate` gave back the matter it was
# handed. It cannot ask whether the composition it returned is the *minimum*: a solve
# that stops early conserves every element perfectly and is still not an equilibrium.
#
# The certificate for that is first-order optimality of `min G(n)` s.t. `A n = b`,
# `n >= 0`. Written per reaction it is the mass-action law: for a reaction whose species
# are all present, `Δ(μ/RT) = Σ_products ν μ − Σ_reactants ν μ` vanishes. Dividing by
# `ln 10` puts it in log-units, where it reads as a saturation index.
#
# This stays a test of the interface rather than of the chemistry. It never compares
# against a number measured elsewhere — `μ` is built from the same standard Gibbs
# energies the solve itself used, so the check is independent of the database, of the
# activity model and of the package version. It only asks the answer to be
# self-consistent.

const R_GAS = 8.31446261815324      # J/mol/K

"""
    potential_params(state) -> NamedTuple

The parameter tuple the potential closure expects: standard Gibbs energies of formation
over RT, plus T, P and the regularization floor. Rebuilt here from public accessors
rather than reaching for `ChemistryLab._build_params`, so a rename inside the dependency
breaks this at the call site instead of silently.
"""
function potential_params(state; ϵ = 1.0e-16)
    T, P = state.T[], state.P[]
    RT = R_GAS * ustrip(us"K", T)
    g = [ustrip(us"J/mol", s[:ΔₐG⁰](T = T, P = P; unit = true)) / RT for s in state.system.species]
    return (ΔₐG⁰overT = g, T = ustrip(us"K", T), P = ustrip(us"Pa", P), ϵ = ϵ)
end

"""
    worst_mass_action_residual(state; presence = 1e-10) -> (residual, reaction, checked)

Largest `|Δ(μ/RT)| / ln 10` over the reactions whose species are all present, the reaction
that carries it, and how many reactions qualified. Zero for a converged equilibrium.

`presence` is relative to the largest amount in the state. Species below it sit at the
boundary of the feasible set, where the multiplier `zᵢ` is free to be positive and the
mass-action law is *not* required to hold — including them would report a correct answer
as a failure. It cuts both ways, so `checked` is returned and asserted on: raise the floor
far enough and no reaction qualifies, at which point the residual is zero because nothing
was measured. On the calcite control, `1e-10` admits 6 reactions and `1e-8` only 2.
"""
function worst_mass_action_residual(state; presence = 1.0e-10)
    n = [ustrip(us"mol", x) for x in state.n]
    μ = EquilibriumSolver(state.system, DiluteSolutionModel(), OptimaOptimizer()).μ(
        n, potential_params(state)
    )
    floor_n = presence * maximum(n)
    index = Dict(symbol(s) => i for (i, s) in enumerate(state.system.species))
    worst, culprit, checked = 0.0, "", 0
    for r in reactions(state.system.SM)
        participants = vcat(collect(keys(r.reactants)), collect(keys(r.products)))
        all(sp -> n[index[symbol(sp)]] > floor_n, participants) || continue
        Δ = 0.0
        for (sp, ν) in r.products
            Δ += ν * μ[index[symbol(sp)]]
        end
        for (sp, ν) in r.reactants
            Δ -= ν * μ[index[symbol(sp)]]
        end
        checked += 1
        abs(Δ) > worst && ((worst, culprit) = (abs(Δ), r.symbol))
    end
    return worst / log(10), culprit, checked
end

@testset "ChemistryLab interface" begin
    db = joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json")
    substances = build_species(db)
    by_name = Dict(symbol(s) => s for s in substances)

    ## The construction every chloride example uses: seed a speciation with the phases of
    ## interest, then append the solids.
    seeds = split("Portlandite ettringite monosulphate12 H2O@ Ca+2 OH- Cl- Na+ K+ H+ Al+3 SO4-2")
    aqueous = speciation(substances, seeds; aggregate_state = [AS_AQUEOUS])
    ## Friedel's salt is on the list because `_init_chemistry4` puts it there: the state
    ## below is meant to be the one the model actually poses, not a tidier cousin of it.
    solids = [by_name[n] for n in ("Portlandite", "ettringite", "monosulphate12", "C4AClH10")]
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
        ## perfectly valid optimization problem whose answer is meaningless, so it has to
        ## be caught here rather than diagnosed later as missing matter.
        @test !isempty(_Balance.duplicate_species(offered))
        @test "Portlandite" in _Balance.duplicate_species(offered)
        @test isempty(_Balance.duplicate_species(_Balance.unique_species(offered)))
        ## Three of the four solids are duplicated — Friedel's salt is not among the
        ## speciation seeds, so it is appended once and only once.
        @test length(_Balance.unique_species(offered)) == length(offered) - 3
    end

    ## The state both solve-based tests below work on, built once: the OPC initial
    ## condition of `examples/chloride_ingress/run_4.jl` — portlandite in excess,
    ## monosulphate, alkalis, no chloride yet.
    system = ChemicalSystem(_Balance.unique_species(offered), CEMDATA_PRIMARIES)

    V_rev, φ = 1.0e-3, 0.121
    m_clinker = 350.0 * V_rev
    n_Na2O = 0.0016 * m_clinker / 61.98e-3
    n_K2O = 0.0049 * m_clinker / 94.2e-3

    state = ChemicalState(system; T = 293.15 * us"K")
    set_quantity!(state, "H2O@", (φ * V_rev * 55_500.0 - n_Na2O - n_K2O) * us"mol")
    set_quantity!(state, "Na+", 2n_Na2O * us"mol")
    set_quantity!(state, "K+", 2n_K2O * us"mol")
    set_quantity!(state, "OH-", (2n_Na2O + 2n_K2O) * us"mol")
    set_quantity!(state, "Cl-", 1.0e-16 * us"mol")
    set_quantity!(state, "SO4-2", 1.0e-16 * us"mol")
    set_quantity!(state, "Portlandite", 1.64 * us"mol")
    set_quantity!(state, "monosulphate12", 0.1 * us"mol")

    equilibrated = equilibrate(state, OptimaOptimizer(tol = 1.0e-10, verbose = false))

    @testset "equilibrate conserves the elements it is given" begin
        ## The contract of the dialog: what goes into an equilibration comes back out
        ## of it.

        ## The threshold is what this test can honestly claim. An interface breaking —
        ## a duplicated column, a unit slip, a species that cannot be read back — loses
        ## tens of percent of an element. The round trip closes to 4e-8 on sulfur under
        ## ChemistryLab 0.13; it closed to only ~1e-3 under 0.3.1, which is why the bound
        ## sits well above both rather than pretending to measure either.
        element, err = _Balance.element_balance_error(state, equilibrated, system)
        @test err < 1.0e-2

        ## And the round trip is usable on the other side: a liquid volume to convert moles
        ## back into the concentrations transport works in. Zero would divide by zero at the
        ## first node, which is how this fails in practice rather than by raising.
        V_liq = ustrip(uconvert(us"m^3", equilibrated.V_phases[].liquid))
        @test V_liq > 0
        @test V_liq < φ * V_rev * 1.5      # a pore volume, not the whole REV
    end

    @testset "the answer satisfies the mass-action law" begin
        ## The control, and the reason the threshold below can be believed: calcite + CO₂
        ## in water, the system ChemistryLab validates against Reaktoro. If the
        ## certificate is sound anywhere it is sound here, and it measures 9e-4 log-units.
        @testset "a well-posed system passes it" begin
            aqueous_only = speciation(
                substances, split("Cal H2O@ CO2");
                aggregate_state = [AS_AQUEOUS], exclude_species = split("H2@ O2@ CH4@"),
            )
            control = ChemicalSystem(
                collect(values(Dict(symbol(s) => s for s in aqueous_only))),
                ["H2O@", "H+", "CO3-2", "Ca+2"],
            )

            names = String.(symbol.(control.species))
            n = Any[fill(0.0us"mol", length(names))...]
            n[findfirst(==("H2O@"), names)] = 55.5us"mol"
            n[findfirst(==("Cal"), names)] = 0.05us"mol"
            n[findfirst(==("CO2@"), names)] = 0.01us"mol"

            control_eq = equilibrate(
                ChemicalState(control, n), OptimaOptimizer(tol = 1.0e-10, verbose = false),
            )
            residual, _, checked = worst_mass_action_residual(control_eq)
            @test checked >= 6            # it measured something
            @test residual < 1.0e-2
        end

        ## And the state the chloride examples start from, which does not.
        ##
        ## Measured on this state: 4 reactions qualify and the worst is 11.8 log-units,
        ## on portlandite; the water autoprotolysis is out by 5.9, so the H⁺ and OH⁻ that
        ## come back are six orders of magnitude from satisfying Kw. Lowering the floor
        ## does not rescue it — at 1e-12, 24 reactions qualify and the worst is 179.
        ##
        ## It is not a regression from the version bump: ChemistryLab 0.3.1 with
        ## OptimaSolver 0.2.7 fails it the same way and merely had no convergence check
        ## to say so. Nor is it Friedel's salt: dropping `C4AClH10` from the species list
        ## moves the numbers without fixing them. The lead worth following is that
        ## `OptimaSolver` refuses the log variable space on this system outright — "the
        ## conservation matrix has rank 8 for 9 rows" — so one conservation law is a
        ## combination of the others and the multipliers are not unique, which is exactly
        ## the conditioning an interior-point method stalls on.
        ##
        ## Marked broken rather than deleted: this is the acceptance criterion for the
        ## chloride models, and the day it starts passing, the suite should say so.
        @testset "the OPC initial state does not" begin
            residual, _, checked = worst_mass_action_residual(equilibrated)
            @test checked >= 1
            @test_broken residual < 1.0e-2
        end
    end

end
