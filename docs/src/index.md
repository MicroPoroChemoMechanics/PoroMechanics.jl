# PoroMechanics.jl

*Reactive transport and poromechanics of porous media.*

PoroMechanics.jl simulates coupled phenomena in porous media — flow, solute and reactive
transport, cement chemistry, and poromechanics — on two numerical backends: finite volumes
for transport, finite elements for coupled mechanics.

A physics model is a plain Julia struct holding its material parameters. Multiple dispatch
on that struct selects the constitutive behavior, so a model file stays a description of
its own equations and knows nothing about time stepping or assembly. Jacobians are never
written by hand: the finite volume callbacks are differentiated automatically with
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

## Key features

- **Unsaturated flow** — Richards' equation [richards1931](@cite) with Van Genuchten
  retention [vangenuchten1980](@cite) and Mualem relative permeability [mualem1976](@cite).
- **Solute and multi-ionic transport** — Fick diffusion, Nernst-Planck transport with an
  electroneutrality constraint, effective diffusivity from the Oh-Jang tortuosity model
  [ohjang2004](@cite).
- **Non-isothermal drying** — liquid water, dry air and heat, coupled through a modified
  Kelvin equation and an entropy balance, with latent-heat transport by vapor
  [philip1957](@cite).
- **Poromechanics** — Biot poroelasticity [biot1941](@cite) on unstructured meshes.
- **Reactive transport in cementitious materials** — operator splitting in the style of
  TOUGHREACT [xu2004](@cite), with thermodynamic equilibrium from cemdata18
  [lothenbach2019](@cite) through [ChemistryLab.jl](https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl),
  Friedel's salt binding, and surface complexation on C-S-H [tran2018](@cite).

## Validation limits

The profile regression suite covers Fick, Darcy, Richards 1D, non-isothermal drying and
Biot consolidation. Analytical benchmarks additionally check poroelasticity and Gardner
flow, including mesh or time refinement. Richards 2D and the chloride transport profiles
are not covered by that regression suite.

Reactive transport remains experimental. Examples 3 and 4 use a certified OPC initial
equilibrium, checked against the original element totals. The legacy interior-point
solver retains an expected failing mass-action test and is still used during transient
chemistry steps. A certified initial condition does not validate those transient results. The solid-solution examples
The previously reported solid-solution state-construction failures in `tran2018.jl` and
`m100_ternary.jl` were not reproduced in the 0.13.0 versus 0.14.2 migration checks with
the current examples. This checks construction only; their full hydration and transport
histories still need numerical validation.

Parameter sensitivities are tested for constitutive laws and selected solves. The 2D
homogenization backend uses Float64 assembly and a finite-difference macroscopic tangent;
it is not currently differentiable end to end with respect to material parameters.

## Scope, and where the chemistry belongs

PoroMechanics.jl is today a *chemo*-poro-mechanics code: next to transport and mechanics it
carries chemistry of its own — surface complexation on C-S-H (double layer model), mineral
dissolution and precipitation kinetics, and the physico-chemical data tables the reactive
examples read.

That is a transitional state, not a design choice. Thermodynamic equilibrium is already
delegated to [ChemistryLab.jl](https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl),
which owns the databases, the speciation and the Gibbs minimization. The rest is meant to
follow it upstream, leaving this package to describe transport and mechanics and to call
ChemistryLab.jl for everything chemical.

The clearest sign that the code currently sits in the wrong repository: the double layer
model exists here in three near-identical variants, one per example family. A single
implementation, in ChemistryLab.jl, is where it should live.

## Backends

| Problem class | Library |
| :--- | :--- |
| Transport, diffusion, flow | [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl) |
| Coupled mechanics | [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) |

## Installation

```julia
using Pkg
Pkg.add("PoroMechanics")
```

The examples additionally need `ExtendableGrids` for their grids, and the reactive ones
need `ChemistryLab.jl` and `OptimaSolver.jl`; none of the three is a dependency of the
package itself.

## Where to go next

| Section | For |
| :--- | :--- |
| [Getting Started](quickstart.md) | writing and running a first model, end to end |
| [Examples](examples/fickian_diffusion.md) | one worked problem per physics, with its equations, data and reference solution |
| [API](api.md) | every exported name |
| [References](references.md) | the literature the models are built from |
