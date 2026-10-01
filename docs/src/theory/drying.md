# 7. Drying and temperature

## Evaporation couples water and heat

A wet specimen can lose water as liquid or as vapor. Evaporation transfers water
between phases but does not destroy it. It also consumes latent heat, so drying can
cool the specimen. Conversely, heating changes vapor pressure and therefore transport.
These couplings make a single isothermal Richards equation insufficient when gas and
temperature variations matter.

The package's [`DryingModel`](@ref) assumes a rigid porous skeleton and uses three
unknowns: liquid pressure $p_l$, dry-air partial pressure $p_a$, and absolute
temperature $T$. Gas pressure is the sum $p_g=p_a+p_v$, where $p_v$ is water-vapor
partial pressure. The vapor pressure is determined by liquid–vapor equilibrium.
This model omits gravity and mechanical deformation.

## Why suction changes vapor pressure

At equilibrium, liquid water and vapor have equal specific Gibbs free energy.
That quantity measures the thermodynamic tendency of water to transfer between phases.
At a fixed temperature and for approximately incompressible liquid and ideal vapor,
integrating their pressure dependence gives

```math
\frac{p_v}{p_{v,\mathrm{ref}}(T)}
=\exp\left[\frac{M_w(p_l-p_{l,\mathrm{ref}})}{\rho_l RT}\right].
```

Here $M_w$ is the molar mass of water, $R$ the gas constant, and
$p_{v,\mathrm{ref}}(T)$ the vapor pressure in equilibrium with liquid at the reference
pressure $p_{l,\mathrm{ref}}$ and the same temperature. The exponent is dimensionless.
A lower liquid pressure lowers equilibrium vapor pressure. This is the isothermal
Kelvin relation; the reference state must be specified before interpreting the ratio
as relative humidity.

The implementation uses a temperature-dependent extension, not just this fixed-temperature
formula. With $\theta=T-T_0$, it evaluates

```math
p_v=p_{v0}\exp\left\{\frac{M_w}{RT}\left[
\frac{p_l-p_{l0}}{\rho_l}+\frac{L_0\theta}{T_0}
+(C_{pl}-C_{pv})\left(\theta-T\ln\frac{T}{T_0}\right)
\right]\right\}.
```

Here $L_0$ is latent heat per mass at the reference temperature, and $C_{pl}$ and
$C_{pv}$ are specific heat capacities of liquid and vapor. [`DryingParameters`](@ref)
stores these constants; [`vapor_pressure`](@ref) evaluates the relation. At
$p_l=p_{l0}$ and $T=T_0$, it returns $p_{v0}$. Pressures used for ideal-gas densities
are absolute. Do not insert an excess pressure into a gas law.
Dangla §7.8 gives the equilibrium argument [dangla_notes](@cite).

## What is conserved?

For fixed porosity $n$, total water mass per bulk volume is

```math
m_w=n\left[\rho_l S_l+\rho_v(1-S_l)\right], \qquad
\rho_v=\frac{M_w p_v}{RT}.
```

Its balance uses the sum of liquid and vapor mass fluxes. Evaporation would enter separate
phase balances with opposite signs, so it cancels when those balances are added.
Dry air has a separate total-mass balance, including dissolved air in this implementation.
Liquid and gas move through Darcy-type laws; vapor also diffuses through the gas mixture.

The third stored quantity is **volumetric entropy**, not simply $T$ or heat capacity
times temperature. It includes the contributions of the phases and a capillary term.
The thermal transport combines conduction and entropy carried by moving matter,
including the effect of latent heat [philip1957](@cite).

The implemented entropy balance follows the reference model's approximation without a
separate volumetric entropy-production term. It should not be presented as the unrestricted
exact energy balance. Dissolved air is assigned gaseous-air entropy, neglecting heat of
dissolution. These assumptions delimit the model's physical scope.

## Connecting to a simulation

| Physical ingredient | Code or example |
|:---|:---|
| Reference pressure, temperature, heat capacities, latent heat | `DryingParameters` |
| Regional porosity, retention, transport properties | `DryingMaterial` |
| Water, air, and entropy storage and fluxes | `DryingModel` |
| Boundary heating | `heat_flux`, specified as incoming heat flux in W/m² |
| Complete initial and boundary data | [Non-isothermal drying example](../examples/nonisothermal_drying.md) |

An imposed incoming heat flux $Q$ is converted to an entropy flux $Q/T$.
The implementation floors the denominator at 200 K during trial evaluations; this is
a numerical guard, not a validation of the model at such temperatures. Temperature
Dirichlet and heat-flux conditions should be assigned to distinct boundaries.

This is heat and moisture transport in a rigid material. Thermoporoelastic deformation
would require additional mechanical coupling and thermal constitutive terms; it does
not follow automatically from solving the drying model.

**Check your understanding.** In a sealed specimen, heating can change saturation and
pressure without changing total water mass: redistribution between liquid and vapor
is compatible with conservation.

Continue with [the numerical formulation](numerics.md).
