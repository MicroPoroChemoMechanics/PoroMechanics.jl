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

## Backends

| Problem class | Library |
| :--- | :--- |
| Transport, diffusion, flow | [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl) |
| Coupled mechanics | [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl) |

## Installation

PoroMechanics.jl and two of its dependencies are distributed through the
MicroPoroChemoMechanics registry:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url = "https://github.com/MicroPoroChemoMechanics/MPCM-Registry"))
Pkg.add("PoroMechanics")
```

## Where to go next

| Section | For |
| :--- | :--- |
| [Getting Started](quickstart.md) | writing and running a first model, end to end |
| [Examples](examples/M1_diffusion.md) | one worked problem per physics, with its equations, data and reference solution |
| [API](api.md) | every exported name |
| [References](references.md) | the literature the models are built from |
