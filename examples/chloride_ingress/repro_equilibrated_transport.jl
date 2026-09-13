# Reproduce and check the local equilibrium used by reduced-Newton transport.
# Run from the repository root:
# julia +1.12 --project=examples examples/chloride_ingress/repro_equilibrated_transport.jl
using LinearAlgebra
include("equilibrated_transport.jl")
include("element_balance.jl")

function opc_equilibrated_case(; chloride = 1.0)
    substances = build_species(joinpath(pkgdir(ChemistryLab), "data", "cemdata18-thermofun.json"))
    by_name = Dict(symbol(s) => s for s in substances)
    seeds = split("Portlandite ettringite monosulphate12 H2O@ Ca+2 OH- Cl- Na+ K+ H+ Al+3 SO4-2")
    aq = speciation(substances, seeds; aggregate_state = [AS_AQUEOUS])
    solids = [by_name[n] for n in ("Portlandite", "ettringite", "monosulphate12", "C4AClH10")]
    system = ChemicalSystem(unique_species(vcat(collect(aq), solids)), CEMDATA_PRIMARIES)
    state = ChemicalState(system; T = 293.15us"K")
    # The original oxide recipe of run_4, expressed per m³ of medium. Add NaCl
    # together so a finite chloride probe preserves the recipe's charge balance.
    na2o = 0.0016 * 350.0 / 61.98e-3
    k2o = 0.0049 * 350.0 / 94.2e-3
    for (name, amount) in (
            "H2O@" => 0.121 * 55_500.0 - na2o - k2o,
            "Na+" => 2na2o + chloride, "K+" => 2k2o,
            "OH-" => 2na2o + 2k2o, "Cl-" => chloride,
            "Portlandite" => 1640.0, "monosulphate12" => 100.0,
        )
        set_quantity!(state, name, amount * us"mol")
    end
    prim = string.(symbol.(system.SM.primaries))
    b = system.SM.A * ustrip.(us"mol", state.n)
    names = ("Cl-", "Na+", "K+", "Ca+2", "SO4-2", "H+")
    fixed = NamedTuple{(Symbol("H2O@"), Symbol("AlO2-"))}(
        Tuple(b[findfirst(==(name), prim)] for name in ("H2O@", "AlO2-"))
    )
    components = ComponentSet(;
        names, z = (-1, 1, 1, 2, -2, 1),
        D = (2.032e-9, 1.334e-9, 1.957e-9, 0.792e-9, 1.065e-9, 9.31e-9), fixed
    )
    model = equilibrated_transport(components, system; initial_state = state)
    return model, collect(b[model.rows.transported])
end

function check_equilibrated_transport()
    model, totals = opc_equilibrated_case()
    eq, cert = equilibrium_state(model, totals)
    cert.optimal || error("uncertified OPC equilibrium: $cert")
    b = component_totals(model, totals)
    balance = norm(model.system.SM.A * ustrip.(us"mol", eq.n) - b, Inf)
    c, ok = speciate(model, totals)
    ok || error("speciation rejected")
    J = ForwardDiff.jacobian(t -> collect(first(speciate(model, t))), totals)
    println("ChemistryLab ", pkgversion(ChemistryLab), "; certified = ", cert.optimal)
    println("Component balance error [mol/m³]: ", balance)
    println("Aqueous component concentrations [mol/m³ water]: ", c)
    println("Jacobian diagonal: ", diag(J))
    return model, totals, J
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_equilibrated_transport()
end
