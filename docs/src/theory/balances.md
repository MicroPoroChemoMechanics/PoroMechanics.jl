# 2. Balance laws and the role of thermodynamics

## Start with a container

Water stored in a container changes because water enters, leaves, or is supplied inside.
A porous volume obeys the same accounting rule. For a fixed volume $\Omega$ in a
rigid saturated medium,

```math
\frac{\mathrm d}{\mathrm dt}\int_\Omega \rho_l n\,\mathrm dV
+\int_{\partial\Omega}\mathbf w\cdot\mathbf n\,\mathrm dA
=\int_\Omega r_m\,\mathrm dV.
```

The first term is the rate of change of stored mass. The second is the **outward**
mass flow through the boundary. The last is an internal supply $r_m$ in kg/(m³·s).
A positive outward flow reduces storage when there is no supply. Here $\mathbf w$
is mass flux relative to the stationary skeleton.

Dividing a one-dimensional balance by the volume of a short segment and shrinking
that segment gives the local form

```math
\frac{\partial(\rho_l n)}{\partial t}+\nabla\cdot\mathbf w=r_m.
```

The divergence measures net outward flux per unit volume. In one dimension it is
simply $\partial w_x/\partial x$. It is not the flux itself: equal inflow and
outflow give zero divergence even if water moves rapidly.

## A deforming skeleton

When the skeleton moves, we follow its reference volume and measure fluid motion
relative to it. Define fluid mass $m_f$ per unit reference volume and

```math
\zeta=\frac{m_f-m_{f0}}{\rho_{l0}}, \qquad
\dot\zeta+\nabla\cdot\mathbf q=s.
```

A dot means a time derivative. The second equation is the linearized, small-deformation
balance: fluxes and derivatives are evaluated to first order about the reference state,
and $s=r_m/\rho_{l0}$ has units s⁻¹. At finite deformation one needs transformed
reference fluxes; this simplified equation is not the finite-deformation balance.
For constant fluid density, $\zeta$ is the change in pore volume per reference volume.
For a compressible fluid, density changes contribute too.

## Force balance

Slow loading permits us to neglect acceleration. Each small volume then satisfies

```math
\nabla\cdot\boldsymbol\sigma+\mathbf f=\mathbf0,
```

where $\mathbf f$ is body force per unit volume, in N/m³. For example, gravity gives
$\mathbf f=\rho_{\mathrm{bulk}}\mathbf g$. With no body force, axial stress in a
uniform one-dimensional column is constant along the column. This does not imply
that pressure or strain is spatially uniform.

A balance says what must be conserved. It does not tell us how easily water flows or
how stiff the skeleton is. Those questions require **constitutive laws**, such as
Darcy's law and an elastic stress–strain relation.

## What thermodynamics adds

The first law accounts for energy: stored energy changes through mechanical work,
heat exchange, and energy carried by matter. The second law requires nonnegative
entropy production. Together they restrict constitutive laws. A passive material cannot
continually generate useful energy from nothing.

For an isothermal saturated elastic medium, a useful stored energy per reference volume is

```math
\psi(\boldsymbol\varepsilon,\zeta)
=\tfrac12\boldsymbol\varepsilon:\mathbb C:\boldsymbol\varepsilon
+\tfrac M2\left(\zeta-b\varepsilon_v\right)^2.
```

The first term is elastic energy; $\mathbb C$ is the drained stiffness tensor,
the linear mapping from strain to effective stress. For a simple uniaxial elastic
bar, elastic energy per volume is $E\varepsilon^2/2$: the area under the straight
stress–strain line. The tensor expression extends that idea to several dimensions.
A colon means summation over tensor components, like a dot product for matrices.
The second term penalizes a fluid-content change that cannot be accommodated by
skeleton expansion. Both terms have units Pa, equivalently J/m³.
Differentiating with respect to the two state variables gives

```math
p=\frac{\partial\psi}{\partial\zeta}=M(\zeta-b\varepsilon_v),
\qquad
\boldsymbol\sigma=\frac{\partial\psi}{\partial\boldsymbol\varepsilon}
=\mathbb C:\boldsymbol\varepsilon-bp\mathbf I.
```

This calculation explains why the same coupling coefficient $b$ appears in both
relations. Chapter 4 develops their physical meaning. For $M>0$ and positive elastic
moduli, this energy is nonnegative about the reference state.

Flow dissipates energy. With driving force $\mathbf d=-\nabla p+\rho_l\mathbf g$
and Darcy flux $\mathbf q=\mathcal K\mathbf d$, its dissipation per volume is
$\mathbf q\cdot\mathbf d=\mathcal K|\mathbf d|^2\ge0$ if $\mathcal K\ge0$.
Likewise Fourier's law $\mathbf j_h=-\lambda_T\nabla T$ transports heat toward
lower temperature when thermal conductivity $\lambda_T$ is positive. Here $T$
is absolute temperature and $\mathbf j_h$ is heat flux in W/m².

## Closing a model

For the saturated problems ahead, we need a mass balance, force balance, a strain–
displacement relation, an elastic storage law, and a flow law. We must also specify
initial and boundary conditions. Omitting any of these leaves an incomplete problem.

PoroMechanics.jl separates these roles: constitutive functions provide material behavior,
models define storage and transport, and numerical backends assemble their equations.
The [numerical chapter](numerics.md) makes this connection explicit. More general
thermodynamic arguments are given in Dangla, chapter 4 [dangla_notes](@cite).

**Check your understanding.** In a sealed specimen with no sources, the integral of
fluid-content change is zero. It does not follow that pressure is zero, or that internal
redistribution is impossible.

Continue with [transport laws](transport.md).
