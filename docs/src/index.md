```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "PoroMechanics.jl"
  text: "Porous media, coupled"
  tagline: Flow, solute and reactive transport, cement chemistry and poromechanics — on finite volume and finite element backends, differentiable with respect to material parameters.
  image:
    src: /logo.png
    alt: PoroMechanics
  actions:
    - theme: brand
      text: Get started
      link: /quickstart
    - theme: alt
      text: Examples
      link: /examples/fickian_diffusion
    - theme: alt
      text: Validation
      link: /validation/terzaghi
    - theme: alt
      text: View on GitHub
      link: https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl

features:
  - icon: 🚀
    title: Getting started
    details: Install the package, then write and run a first transport model from its governing equation to its profile.
    link: /quickstart
  - icon: 💧
    title: Examples
    details: One worked problem per physics — Fick diffusion, Darcy flow, Richards infiltration, non-isothermal drying, Biot consolidation — each with its equations, its data and its reference solution.
    link: /examples/fickian_diffusion
  - icon: 📐
    title: Validation
    details: Terzaghi, Mandel, Cryer, De Leeuw and Gardner against their closed forms, and the Bil reference cases against published numerics.
    link: /validation/terzaghi
  - icon: 🧩
    title: Writing a model
    details: What a physics model is here — a struct of material parameters, and the five callbacks multiple dispatch selects on it.
    link: /demos/writing_a_model
  - icon: 🧪
    title: Reactive transport
    details: Nernst-Planck transport under electroneutrality, and equilibrium chemistry delegated to ChemistryLab.jl.
    link: /demos/reactive_transport
  - icon: 📖
    title: API reference
    details: Every exported name — interfaces and backends, physics models, constitutive laws, plasticity.
    link: /api
---
```

## What it does

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

The transport models `NernstPlanck` and `EquilibratedTransport` now live in the package.
The optional ChemistryLab extension translates conserved totals and certified equilibrium
results; it implements no equilibrium solver. The double-layer and AFm prototypes remain
in the examples pending migration of their chemical laws to ChemistryLab.
See [Reactive transport](demos/reactive_transport.md).

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
that solve chemical equilibria need `ChemistryLab.jl`, `DynamicQuantities.jl` and
`OptimaSolver.jl`. ChemistryLab and DynamicQuantities are optional dependencies;
loading them activates the equilibrium adapter. `using OptimaSolver` enables
ChemistryLab's certified solver. Pure ionic transport requires none of them.
