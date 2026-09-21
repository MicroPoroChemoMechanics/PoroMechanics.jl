# API

Every exported name of `PoroMechanics`.

```@docs
PoroMechanics
```

## Model and solver types

```@docs
AbstractPoroModel
AbstractPoroSolver
```

## Finite volume interface

Callbacks consumed by [VoronoiFVM.jl](https://github.com/j-fu/VoronoiFVM.jl). A model
implements the ones its physics needs; the Jacobian is obtained from them by automatic
differentiation, never written by hand.

```@docs
storage!
flux!
bcondition!
reaction!
```

## Finite element interface

Callbacks consumed by [Ferrite.jl](https://github.com/Ferrite-FEM/Ferrite.jl).

```@docs
assemble_element!
element_matrices!
facet_load!
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

The [drying tutorial](examples/nonisothermal_drying.md) supplies a clay/rock case.
Its `drying_case` factory keeps geometry, initial conditions and heating history
explicit, and `run_drying(case = custom_case)` runs a modified experiment without
redefining any balance callback.

## Model introspection

```@docs
PoroMechanics.nspecies
PoroMechanics.species_names
```

## Constitutive layer

Material laws owned by the package. Every law carries its coefficients as *type
parameters*, so a result can be differentiated with respect to the parameters themselves
and not only with respect to the unknowns — the property that makes inverse calibration
and sensitivity analysis possible.

### Retention curves

```@docs
AbstractRetention
VanGenuchten
Gardner
ExponentialCutoff
saturation
dsaturation_dpc
```

### Relative permeability

```@docs
AbstractRelativePermeability
Mualem
GardnerKrl
PowerLawKrl
relative_permeability
gas_relative_permeability
```

### Tortuosity

```@docs
AbstractTortuosity
OhJang
tortuosity
```

### Poroelasticity

```@docs
AbstractPoroelastic
BiotPoroelastic
lame
shear_modulus
bulk_modulus
oedometric_modulus
biot_modulus
compaction_coefficient
storage_coefficient
consolidation_coefficient
hydraulic_conductivity
skempton
undrained_poisson
undrained_bulk_modulus
```

### The stress–strain interface

The mechanical layer deliberately knows nothing about pore pressure: its signature is the
one the Ferrite ecosystem already uses, so a constitutive model written against it travels
beyond this package. The poromechanical coupling sits on top.

```@docs
AbstractMaterial
AbstractMaterialState
NoState
initial_state
material_response
stress_controlled_response
LinearElastic
elastic_stiffness
skeleton
total_stress
poro_response
```

### Pressure-dependent elasticity

```@docs
LogarithmicElastic
LogarithmicElasticState
tangent_moduli
mean_compressive_stress
```

### Unsaturated effective stress

When two fluid phases share the pore space there is no single pore pressure, and the
question of what replaces it is a modeling choice rather than a formula — so the Bishop
coefficient is a model of its own.

```@docs
AbstractBishop
SaturationBishop
PowerBishop
bishop_coefficient
equivalent_pore_pressure
unsaturated_total_stress
suction_stress
```

## Backends

Glue to the solver packages. The physics lives in the constitutive layer; these only wire
it up.

### Finite volumes

```@docs
fvm_system
```

### Finite elements

```@docs
biot_element_matrices!
radial_element_matrices!
node_dof_maps
combine!
```

#### Axisymmetric elastoplasticity

Ferrite v1 has no axisymmetric element, so the kinematics are written out: the hoop strain
``\varepsilon_{\theta\theta} = u_r/r`` makes the strain a genuine 3D tensor even though the
mesh is 2D, and a constitutive model that reads ``-\mathrm{tr}(\sigma)/3`` gets the wrong
mean stress without it.

```@docs
axisymmetric_strain
axisymmetric_shape_strain
assemble_axisymmetric!
newton_solve!
```

## Models

Physics models that the package ships, as opposed to those a script defines for itself.
What earns a model its place here is that someone else would write it again: the equation
is fixed and only the data changes, so shipping it turns the next script into a case rather
than a re-implementation.

That is not the same as being complicated. Fick diffusion is a storage term and a flux, and
[Writing a model](demos/writing_a_model.md) measures that writing such a model through the
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

## Materials

### Barcelona Basic Model

Elastoplasticity for unsaturated soils, in which suction is a second loading variable:
drying expands the yield surface, wetting shrinks it, and a soil wetted under constant
stress can therefore be pushed to yield and collapse.

```@docs
BBM
BBMState
compression_index
preconsolidation
yield_function
mean_pressure
equivalent_stress
bbm_moduli
suction_stress_increment
log_mean
step_shear_modulus
trial_stress
hardening_modulus
PoroMechanics.dyield_dp
PoroMechanics.dyield_dq
PoroMechanics.deviator
deviatoric_tolerance
PoroMechanics.return_residual
PoroMechanics.solve_return_map
elastoplastic_tangent
algorithmic_tangent
ContinuumTangent
ExplicitPredictor
```

### Drucker-Prager

Perfect plasticity with a non-associated flow rule: friction sets the yield cone, dilatancy
sets the plastic flow direction, and keeping the two apart is what lets the model shear
without inventing volume.

```@docs
DruckerPrager
DruckerPragerState
drucker_prager_return
friction_coefficient
cohesion_intercept
dilatancy_coefficient
apex_pressure
```

### A Biot medium with an arbitrary skeleton

```@docs
BiotPlastic
porosity
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

## Tabulated material curves

A retention or relative-permeability curve given as data rather than as a formula —
measured points, or a curve digitized from a reference. Interpolation is linear in the
tabulated variable and the coefficients are parameterized by their own type, so a table can
carry `ForwardDiff.Dual` values like any closed-form law.

```@docs
Tabulated
TabulatedKrl
interpolate_table
```

