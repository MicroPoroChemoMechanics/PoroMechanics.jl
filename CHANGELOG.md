# Changelog

## v0.1.0 — Initial release

### Core

- `AbstractPoroModel` / `AbstractPoroSolver` — a physics model is a plain struct holding
  its material parameters; multiple dispatch selects the constitutive behaviour.
- FVM interface: `storage!`, `flux!`, `bcondition!`, `reaction!`.
- FEM interface: `assemble_element!`, `element_matrices!`, `facet_load!`.
- No hand-written Jacobians: `VoronoiFVM.jl` differentiates the FVM callbacks with
  `ForwardDiff.jl` and owns the Newton loop and the adaptive time stepping.

### Models (in `examples/`, moving to `src/Models/` as they are validated)

- **M1 diffusion** — Fick diffusion of a solute in a saturated porous medium (1D).
- **Darcy column** — transient single-phase Darcy flow (1D).
- **M1 Richards** — unsaturated flow, Van Genuchten retention and Mualem relative
  permeability (1D).
- **Richards 2D** — unsaturated drainage of a composite column, unstructured grid.
- **M6 drying** — non-isothermal drying, three coupled unknowns (liquid pressure, dry air
  pressure, temperature), two materials.
- **M7 Biot** — Biot poroelasticity on P1/P1 triangles (Ferrite.jl), Ternay dam test case.
- **Chloricem** — reactive chloride transport in cementitious materials, from single-species
  Langmuir adsorption up to multi-ionic transport coupled to `ChemistryLab.jl` thermodynamic
  equilibrium (cemdata18) and surface complexation on C-S-H.

### Requirements

- Julia 1.12 or later — the floor comes from `ChemistryLab.jl` and `OptimaSolver.jl`.
