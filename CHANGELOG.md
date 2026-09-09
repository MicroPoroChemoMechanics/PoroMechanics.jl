# Changelog

## Unreleased

- Bound `MeanFieldHomogenization` to 0.7 in `docs/Project.toml` and `examples/Project.toml`.
  The documentation build resolves its manifest from scratch — `docs/Manifest.toml` is
  gitignored — so an unbounded dependency picked up 0.12, where `RVE(::Symbol)` no longer
  exists, and `benchmarks/mfh_poroelastic.jl` failed with a `MethodError` that does not
  reproduce locally against the installed 0.7. The bound restores the pairing the pages
  were measured under. Raising it is an API migration rather than a version bump: 0.12
  also moves TensND from 0.4 to 0.5, `benchmarks/mfh_thick_cylinder.jl` calls `TensISO`
  directly, and the printed coefficients are compared against closed forms, so the
  numbers have to be revalidated and not merely recompiled.
- Update the examples and test environments to ChemistryLab 0.14.2 and OptimaSolver 0.5
  (resolved to 0.5.1). ChemistryLab 0.14 makes `equilibrate(state)` certified by default;
  calls with an explicit optimizer retain the single-backend path. Add an OPC certificate
  and element-balance check for the default API used by example 2b. Correct the obsolete
  claim that the package was held at ChemistryLab 0.3.
- `newton_solve!` now throws when backtracking or the iteration budget is exhausted.
  Rejected trials are not committed; the residual history includes the initial state and
  every accepted correction, including convergence on the last allowed correction.
  `maxiter` counts corrections. Callers must handle failure before advancing material states.
- Add Newton failure-path tests and require convergence in the BBM continuum-tangent comparison.
- Allow separate `core`, `validation`, `regression`, `chemistry` and `bil` test groups;
  the default still runs all groups.
- Select regression tolerances per case, with an explicit strict mode, and record Julia,
  CPU and BLAS information in newly generated references.
- Initialize OPC in chloride examples 3 and 4 through ChemistryLab’s certified solver,
  checking optimality against the original element totals. Retain the legacy solver’s
  expected failing test; transient chemistry still requires revalidation.
- Document the actual profile regression coverage, the legacy OPC equilibrium failure
  and the limits of parameter differentiation in the homogenization backend.

### ChemistryLab migration checks

The migration compares ChemistryLab 0.13.0 / OptimaSolver 0.4.3 with
ChemistryLab 0.14.2 / OptimaSolver 0.5.1, using Julia 1.12.7 and the same example code.
The latest registered ChemistryLab version was checked in the
[General registry](https://github.com/JuliaRegistries/General/blob/master/C/ChemistryLab/Versions.toml).
The [upstream release notes](https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl/blob/v0.14.2/CHANGELOG.md)
describe the certified default and the required OptimaSolver update.

- The full `Pkg.test()` run under Julia 1.12.7 passes: 620 passed, one expected broken
  test retained for the legacy OPC interior-point equilibrium. No new failures.
- OPC certified initialization remains approximately 385.2 mol/m³ of pore water for OH⁻
  and 1640 mol/m³ of concrete for portlandite. The default `equilibrate(state)` now passes
  the independent certificate and original element-balance checks as well.
- Reduced transport checks use `N=2`, `t_end=1.0`, `n_save=1`, with other defaults.
  Example 2b finishes with a relative L2 signature change of 6.05e-9. Its largest change
  in dissolved calcium is 7.40e-6 mol/m³ of pore water (about 0.016%).
- Example 4 also finishes, including its differentiated chemistry callback, but changes
  more. At its middle node, the final quantities are:

| Quantity | 0.13.0 / 0.4.3 | 0.14.2 / 0.5.1 |
| :--- | ---: | ---: |
| Dissolved Ca²⁺ [mol/m³ pore water] | 0.0668051 | 0.0627926 |
| Free Cl⁻ [mol/m³ pore water] | 7.80909e-6 | 1.70836e-5 |
| Adsorbed Cl⁻ [mol/m³ concrete] | 3.86286e-8 | 8.46175e-8 |
| Friedel's salt [mol/m³ concrete] | 1.17517e-6 | 9.15056e-7 |

These are short execution checks, not validated long-term reference profiles. Both
versions still report `MaxIters` from the legacy interior-point path; the differences
cannot establish which transient result is physically correct. Existing reference files
were not regenerated. The two dependencies move together, so these comparisons do not
isolate ChemistryLab's changes from OptimaSolver's.

The previously reported `ChemicalState` construction failures for the solid-solution
examples `tran2018.jl` and `m100_ternary.jl` were not reproduced under either version
with the current code and dependency snapshots. This only checks construction, not
their complete hydration or transport histories. The reduced-run timings include Julia
compilation and concurrent work and are not performance benchmarks.

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
| `fickian_diffusion` | Fick diffusion, saturated medium | VoronoiFVM |
| `darcy_column` | transient single-phase Darcy flow | VoronoiFVM |
| `richards_1d` | unsaturated flow, Van Genuchten / Mualem | VoronoiFVM |
| `richards_2d` | unsaturated drainage of a composite column | VoronoiFVM |
| `nonisothermal_drying` | non-isothermal drying: liquid, dry air, heat | VoronoiFVM |
| `biot_consolidation` | Biot poroelasticity, two materials | Ferrite |
| `chloride_ingress` | reactive chloride transport in cementitious materials | VoronoiFVM |

The `chloride_ingress` family is staged: single-species Langmuir adsorption, then multi-ionic
transport with electroneutrality, then operator splitting against `ChemistryLab.jl`
equilibria, then surface complexation on C-S-H. Only the earlier stages are validated
against reference profiles; the later ones and the ternary binder variants are implemented
but not yet checked against TOUGHREACT/Thermoddem results.

### Scope

This release is a *chemo*-poro-mechanics code: alongside transport and mechanics it carries
surface complexation on C-S-H, mineral kinetics, and the physico-chemical data tables the
reactive examples read. Thermodynamic equilibrium is already delegated to `ChemistryLab.jl`,
and the rest is intended to follow it upstream — leaving this package to describe transport
and mechanics, and to call `ChemistryLab.jl` for everything chemical.

### Requirements

Julia 1.12 or later. The floor comes from `ChemistryLab.jl` and `OptimaSolver.jl`, whose
every registered version declares `julia = "1.12.0-1"`.

The examples and tests require ChemistryLab 0.14.2 or a compatible 0.14 patch release,
paired with OptimaSolver 0.5. These are not dependencies of the core library.
The earlier claim that ChemistryLab was held at 0.3 was stale: the environments already
used 0.13. The reported factor of two on portlandite came from duplicate species in the
chemical system, not a demonstrated normalization change in ChemistryLab; species are
now deduplicated before constructing the system.
