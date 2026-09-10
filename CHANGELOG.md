# Changelog

## Unreleased

- Add `SurfaceResolvedTransport`: the surface potential as a nodal unknown, with the
  Gouy-Chapman balance as its equation, so nothing is solved inside a callback any more.
  The two formulations of the same physics — nested root-find against nodal unknown — agree
  to `1.9e-16` on every transported species, which is the strongest check either can get.
  The nodal one is **6.6× faster** (1.5 s against 9.9 s for the same 67 steps): removing a
  bisection from a storage term evaluated on `Dual`s, once per node per Newton iteration,
  more than pays for the extra unknown. Its residual is scaled by `F Γ_max`, without which
  it sits four orders below the concentration rows that Newton measures convergence on.
- Note that the algebraic unknown needs a consistent initial value. `β = 0` is not on the
  constraint manifold, and the first step then has to move the whole surface inventory at
  once: the step controller collapses to `Δt_min` and reports `Δu/Δu_opt = 1.8e8`, which
  reads as a physics failure and is an initialization failure.
- **Fix `conservation_defect`**, which was measuring the wrong thing on models with
  algebraic rows. A row with no storage has `rate = 0` while `integrate` returns its
  constraint residual, so the difference is that residual over a scale the transported rows
  set. On the double layer model that reported `1e-8`; restricted to the rows that have a
  storage it is `1.1e-11`. It takes a `species` argument now. Two explanations of the
  `1e-8` were proposed and measured before the cause was found — the bisection tolerance
  and the Newton tolerance — and neither survived: tightening Newton to `1e-12` made it
  marginally worse.
- Add `examples/chloride_ingress/sorbing_transport.jl`: the double layer inside the
  **storage**, so that `T_i = φ c_i + S_i(c)` is what the scheme conserves. `S_i` is a
  function of the unknowns rather than a field refreshed between segments, so VoronoiFVM
  differentiates it and the whole matrix `∂S_i/∂c_j` lands in the Jacobian — exactly, and
  without the lag. The surface potential is still a root-find inside the storage term, once
  per node per Newton iteration; that is affordable (67 steps, 11 s on 41 nodes) and it is
  the reduced-Newton idea applied where it is cheap.
- Measure what the diagonal retardation of `run_4.jl` discards. At the front, the row of
  `∂S_Cl/∂c` reads `+1.64e-3` for chloride, `−2.71e-5` for sodium and potassium, `−6.56e-4`
  for hydroxide and **`−1.51e-2` for calcium** — 9.2 times the diagonal term, and of the
  opposite sign. Calcium competes for the deprotonated silanol sites, so it governs
  chloride binding more strongly than chloride does. Keeping only `∂S_Cl/∂c_Cl` keeps the
  second smallest term of the row.
- Measure the conservation defect on the real double layer: `1.0e-8` for the inventory
  form against `2.7e-2` for a frozen `K_d`, six orders apart. The `1.0e-8` is not
  round-off and its floor is identified: `solve_dlm` brackets β to `1e-9` and
  `dS_Cl/dβ = 0.197`, so `S_Cl` is defined to about `1.1e-9` relative and the transient
  accumulates it. Promoting β to a nodal unknown, which `dlm_residual` exists for, removes
  the root-find and that floor together.
- Add `examples/chloride_ingress/nernst_planck.jl`: multi-ionic transport with a
  zero-current closure, the transport core of the rewrite of `run_4.jl` onto conservative
  balances. Ions of different mobility cannot separate freely, and `run_4.jl` lets them:
  measured on its own solution, the net current it carries reaches
  `|Σ zᵢJᵢ| / Σ|zᵢJᵢ| = 0.46` at the front, and neutrality is then restored inside the
  chemistry pass by letting OH⁻ absorb it. ChemistryLab states the same thing structurally
  — charge is a pseudo-element `:Zz`, one row of the conservation matrix, and on the OPC
  system that row is independent: `A` is 9×34 of rank 9, rank 8 without it.
- Validate it against the closed form. For a binary z:z salt the zero-current condition
  gives `Ψ = r ln(c/c_boundary)` with `r = (D₋−D₊)/(D₋+D₊) = 0.20737`. Measured over
  21…321 nodes the error is −9.87, −3.51, −1.01, −0.26, −0.066 %: second order in `dx`,
  converging to the analytical value.
- Record that a transient parameter sensitivity does **not** follow from making a model's
  coefficients type parameters. `fvm_system` builds a system whose unknowns are `Float64`;
  a `Dual` in a coefficient reaches the callbacks but the solution vector cannot carry its
  partials, and the solve fails converting one back. VoronoiFVM's own mechanism —
  `System(...; nparams)` plus `solve(...; params)` — is the supported route, and it
  requires the model to read the coefficient from `params`. Marked `@test_broken` with the
  reason, in `test/reactive/nernst_planck.jl`.
- Correct four stale claims in `CLAUDE.md`: the DLM is no longer in three copies, the
  `[compat]` pins are 0.14.2/0.5 rather than 0.13/0.4, the solid-solution construction
  failure was not reproduced, and the rank-8 explanation of the broken OPC equilibrium is
  not supported by the current matrices — `cs.SM.A`, `DualEquilibriumSolver.A` and the
  aqueous block are all full rank. The residual stands; its cause is open again.
- Read the molar volumes of the solid phases from the database rather than from a copy.
  `chemistry_step4!` carried four hard-coded values; they had drifted from what `sp[:V⁰]`
  returns by up to 0.48 %, and the two that drifted most, monosulphate and Friedel's salt,
  are the pair whose exchange drives the porosity change. The reduced run moves by 2.5e-8
  in relative L2, below the regression tolerance.
- Instrument the porosity clamp. `clamp(φ, 1e-4, 0.999)` is a guard, not a correction: if
  it bites, the volume closure has produced a porosity outside the physical range and
  everything downstream is meaningless. It now warns instead of absorbing it silently.
- Add `molar_volume` and `water_density` next to the chloride examples, both reading the
  thermodynamic database. `water_density` is documented and **not yet used**: the molar
  density of water is written as `55_500.0` in five files and nine places, and only
  `run_4.jl` has a regression reference, so changing it there alone would shift four
  uncovered examples and break the initial-condition cross-check against `run_3.jl` in
  `test/chemistry_interface.jl`. The database value is 55345.3 mol/m³ at 20 °C, 0.28 %
  below. Clearing that debt waits until the other examples are pinned.
- Add a conservation harness, `test/reactive/conservation.jl`. `element_balance_error`
  answers the right question for a closed cell and the wrong one for a cell transport runs
  through; the general identity is `d/dt ∫T = Σ_Γ ∫J·n`, and VoronoiFVM's test functions
  compute the right-hand side exactly. The harness integrates an inventory defined
  independently of the scheme, which is the only way it can contradict one: applied to a
  scheme's own storage the identity holds by construction and measures nothing.
  It falsifies, which was the condition for it to prove anything — on a Langmuir tracer,
  the inventory form `T = φc + S(c)` violates the balance by 7e-14 and the frozen
  retardation `T = (φ + K_d)c` by 0.79.
- Add `chloride_ingress` to the regression cases. It is a **non-regression** reference and
  is labeled as one: the transient chemistry still reports `MaxIters` from the legacy
  interior-point path, so the pinned profile records what the code does rather than what is
  physically right. Its tolerance is 1e-6 rather than the 1e-10 of the pure-transport
  cases, because the interior-point solve is not reproducible in its last digits. It has to
  be regenerated from the `examples` environment, the root one having no chemistry stack.
- Merge the three copies of the C-S-H double layer model into
  `examples/chloride_ingress/dlm.jl`. `run_4.jl`, `tran2018.jl` and `chloride_ternary.jl`
  each carried their own; they differed only in whether magnesium was present, in the
  chloride binding mechanism, and in whether the site density was interpolated. Magnesium
  at zero concentration and an interpolation between equal endpoints are special cases, so
  only the mechanism is a real branch and it is now carried by dispatch on `OuterSphere`
  or `TernaryNeutral`. 485 lines removed for 415 added, one of which is a file the three
  examples share.
- The double layer coefficients are type parameters, so a `ForwardDiff.Dual` can enter
  `K_Cl` or `Gamma_max`. The three replaced implementations declared every coefficient
  `::Float64`, which allowed differentiation with respect to a concentration but never
  with respect to a surface constant — the derivative an inverse identification needs.
- Split the model into `dlm_residual`, `dlm_loadings` and `solve_dlm`. The residual has no
  root finding inside, which is what a globally implicit scheme needs in order to carry the
  surface potential as an unknown; the solver is built on the same expression, so the two
  cannot disagree.
- Add `test/dlm.jl` to the `core` group: 26 tests on values, on derivatives against central
  differences, on differentiation with respect to a parameter, and on the `n_csh ≤ 0` guard.
  The pinned numbers were measured on the three implementations before the merge, so they
  guard the refactor rather than the physics.

  The merge is exact where it can be. `solve_dlm_marks` is reproduced bit for bit on all
  288 measured quantities, `solve_dlm_ternary` on 287 of 288 — the exception is one
  derivative differing by one unit in the last place. `run_4.jl` end to end is unchanged
  on all 252 values of a reduced run. Two intended differences remain, both in `run_4.jl`:
  it now sees the `n_csh ≤ 0` guard the other two already had, and it would account for
  magnesium if it transported any, which it does not.
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
