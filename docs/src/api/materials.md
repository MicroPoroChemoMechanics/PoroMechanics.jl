# Plasticity and material models

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

