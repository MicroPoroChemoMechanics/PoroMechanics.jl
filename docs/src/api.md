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
ExponentialCutoff
saturation
dsaturation_dpc
```

### Relative permeability

```@docs
AbstractRelativePermeability
Mualem
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
question of what replaces it is a modelling choice rather than a formula — so the Bishop
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
