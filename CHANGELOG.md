# Changelog

## v0.1.0 — a physics model is a struct, and nothing else

First release.

### The interface

A model is a plain struct holding its material parameters, subtyping
`AbstractPoroModel`. Multiple dispatch on that struct selects the constitutive behavior,
so a model file describes its own equations and knows nothing about time stepping or
assembly.

- Finite volume callbacks: `storage!`, `flux!`, `bcondition!`, `reaction!`.
- Finite element callbacks: `assemble_element!`, `element_matrices!`, `facet_load!`.
- Introspection: `nspecies`, `species_names`.

The stubs throw rather than return silently, so a model that forgets one fails loudly
instead of solving the wrong equation. `bcondition!` is the exception: no boundary term is
a legitimate default.

No Jacobian is written anywhere. `VoronoiFVM.solve` differentiates the finite volume
callbacks with `ForwardDiff.jl` and owns the Newton loop and the adaptive time stepping,
which is what keeps a model file down to its equations.

### Models

Shipped in `examples/`, each with its governing equations, its material data and the
reference solution it is checked against. They move to `src/Models/` as they are validated.

| Example | Physics | Backend |
| :--- | :--- | :--- |
| `M1_diffusion` | Fick diffusion, saturated medium | VoronoiFVM |
| `darcy_column` | transient single-phase Darcy flow | VoronoiFVM |
| `M1_Richards` | unsaturated flow, Van Genuchten / Mualem | VoronoiFVM |
| `Richard_2D` | unsaturated drainage of a composite column | VoronoiFVM |
| `M6_drying` | non-isothermal drying: liquid, dry air, heat | VoronoiFVM |
| `M7_Biot` | Biot poroelasticity, two materials | Ferrite |
| `Chloricem` | reactive chloride transport in cementitious materials | VoronoiFVM |

The `Chloricem` family is staged: single-species Langmuir adsorption, then multi-ionic
transport with electroneutrality, then operator splitting against `ChemistryLab.jl`
equilibria, then surface complexation on C-S-H. Only the earlier stages are validated
against reference profiles; the later ones and the ternary binder variants are implemented
but not yet checked against TOUGHREACT/Thermoddem results.

### Requirements

Julia 1.12 or later. The floor comes from `ChemistryLab.jl` and `OptimaSolver.jl`, whose
every registered version declares `julia = "1.12.0-1"`.

`ChemistryLab` is held at 0.3 deliberately. On 0.11 the initial OPC equilibrium moves from
c_OH = 452.2 to 385.2 mol/m³ and n_CH from 1640.2 to 820.0 mol/m³ of concrete — the exact
factor of two on portlandite points at a normalization change, and every reference profile
would have to be re-checked before the bound is widened.
