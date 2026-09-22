# Physics models and solvers

## Models

Physics models that the package ships, as opposed to those a script defines for itself.
What earns a model its place here is that someone else would write it again: the equation
is fixed and only the data changes, so shipping it turns the next script into a case rather
than a re-implementation.

That is not the same as being complicated. Fick diffusion is a storage term and a flux, and
[Writing a model](../demos/writing_a_model.md) measures that writing such a model through the
package costs three lines more than writing it directly against `VoronoiFVM` — the
abstraction pays for itself in what it *shares*, not in what it saves per model. A model
whose equation is particular to one study is still better written where it is used.

Every model carries its boundary data in a `dirichlet` field rather than in a method,
because an imposed value is what distinguishes one case from another and not one model from
another. A value may be a number or a function of time.

```@docs
PoroMechanics.dirichlet_value
PoroMechanics.apply_dirichlet!
```

### Fickian diffusion

```@docs
FickModel
diffusivity
```

### Darcy flow

```@docs
DarcyModel
storativity
mobility
```

### Richards' equation

```@docs
RichardsModel
liquid_conductivity
```

### Saturation from a solution

For Richards flow, saturation follows from the liquid pressure. For drying, it
also depends on the air pressure, temperature and material region.

```@docs
liquid_saturation
```

## Non-isothermal drying

Configure the material laws, shared fluid properties and boundary values, then
pass the model and your grid to `fvm_system`. Each cell region indexes `materials`.
The three rows contain liquid pressure, dry-air pressure and temperature; the
balances store water mass, dry-air mass and entropy.

```@docs
DryingMaterial
DryingParameters
DryingModel
drying_material
vapor_pressure
```

The [drying tutorial](../examples/nonisothermal_drying.md) supplies a clay/rock case.
Its `drying_case` factory keeps geometry, initial conditions and heating history
explicit, and `run_drying(case = custom_case)` runs a modified experiment without
redefining any balance callback.

## Reactive transport

[`NernstPlanck`](@ref) transports ion concentrations and an electric potential.
[`EquilibratedTransport`](@ref) transports conserved component totals and obtains
local aqueous concentrations from ChemistryLab. These are distinct formulations:
the equilibrium-coupled model currently omits electromigration.
See [Reactive transport](../demos/reactive_transport.md) for units, boundary data and
optional chemistry setup.

```@docs
NernstPlanck
nions
ipot
effective_diffusivity
net_charge
edge_current
ComponentSet
ncomp
EquilibratedTransport
equilibrated_transport
component_totals
equilibrium_state
speciate
LocalEquilibriumError
```

## Linear Biot assembly and time integration

Keep the Ferrite mesh, spaces and constraints explicit, select a material for each
cell, then assemble once and integrate at the requested times. A step callback
can record fields or diagnostics without implementing the time loop.

```@docs
assemble_biot_matrices
assemble_biot_load
solve_biot
```

## Poroplasticity

One-dimensional axisymmetric poroplasticity: a Richards-like liquid balance coupled to a
skeleton that may yield. The state carried between steps is the material's own, so any
`AbstractMaterial` can be the skeleton.

```@docs
PoroplastModel
PoroplastState
poroplast_initial_states
poroplast_element_residual
poroplast_step!
fluid_density
liquid_mass
intrinsic_permeability
axisymmetric_strain_1d
```

## Computational homogenization

A periodic cell solved under an imposed macroscopic strain, or under an imposed macroscopic
stress by an outer Newton loop on the strain that produces it. The cells may be plastic, in
which case the tangent is the algorithmic one and the state is carried between macroscopic
steps.

```@docs
PeriodicCell
periodic_cell
cell_states
homogenize_stress
homogenize_to_stress
homogenized_stiffness
homogenized_tangent
plane_strain
```

