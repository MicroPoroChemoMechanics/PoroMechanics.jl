# API

The API reference is organized by responsibility:

- [Interfaces and numerical backends](#Model-and-solver-types) on this page.
- [Physics models and solvers](api/models.md): transport, drying, reactive transport,
  Biot coupling, poroplasticity and homogenization.
- [Constitutive laws](api/constitutive.md): retention, permeability, tortuosity,
  elasticity and effective stress.
- [Plasticity and material models](api/materials.md): BBM, Drucker-Prager and
  coupling to a Biot medium.

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

