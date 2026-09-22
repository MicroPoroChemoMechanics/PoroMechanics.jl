# Constitutive laws

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

