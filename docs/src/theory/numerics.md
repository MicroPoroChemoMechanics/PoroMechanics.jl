# 8. From equations to a numerical solution

## Replace a field by finitely many unknowns

An equation such as $\partial_t p=D_h\nabla^2p$ describes a value at every point.
A computer instead stores values at finitely many locations. A mesh divides the domain
into small pieces, and a discretization translates the continuum equations into algebraic
relations between their unknowns. Refining the mesh should improve the approximation
when the method and problem are well posed.

PoroMechanics.jl uses VoronoiFVM for transport and Ferrite for mechanics. Their roles differ,
but both start from the balances introduced in chapter 2.

## Finite volumes: account for each exchange

For a stored quantity $a(U)$ and flux $\mathbf j(U,\nabla U)$, consider
$\partial_t a+\nabla\cdot\mathbf j=0$. Here $U$ denotes the unknown or vector
of unknowns. Integrating over control volume $i$ gives

```math
V_i\frac{\mathrm d a(U_i)}{\mathrm dt}+\sum_j F_{ij}=0.
```

Here $V_i$ is its volume, and $F_{ij}$ is the rate leaving it toward neighbor $j$.
An interior exchange enters one volume and leaves the other: $F_{ij}=-F_{ji}$.
When all balances are added, internal exchanges cancel. This is the core conservation
property of a finite-volume method.

For homogeneous Darcy flow on a one-dimensional edge of length $h$ and face area
$A$, $F_{ij}=A\mathcal K(p_i-p_j)/h$. The model's `flux!` callback supplies
$\mathcal K(p_i-p_j)$; VoronoiFVM supplies the geometric factors. Inserting another
factor $1/h$ into this callback would count the distance twice.

| Continuum term | Callback | Darcy example |
|:---|:---|:---|
| Stored quantity $a$ | `storage!` | $Sp$ |
| Transport coefficient times value difference | `flux!` | $\mathcal K(p_i-p_j)$ |
| Boundary data | `bcondition!` | Prescribed pressures from model data |

The [writing-a-model tutorial](../demos/writing_a_model.md) develops a runnable example.
For nonlinear laws, the residual depends nonlinearly on $U$. A Newton iteration
corrects a trial solution using its Jacobian, the matrix of residual derivatives.
Automatic differentiation supplies derivatives for the finite-volume callbacks;
it does not remove the need for convergence checks.

## Finite elements: balance against test functions

For mechanics, approximate displacement and pressure by combinations of basis functions
within each element. Instead of enforcing equilibrium at every point, require it to hold
when averaged against a family of **test functions**. A displacement test function
$\mathbf v$ can be viewed as a small virtual displacement; it is zero where displacement
is prescribed. Multiplying equilibrium by $\mathbf v$ and integrating by parts gives

```math
\int_\Omega\boldsymbol\varepsilon(\mathbf v):\boldsymbol\sigma\,\mathrm dV
=\int_\Omega\mathbf v\cdot\mathbf f\,\mathrm dV
+\int_{\Gamma_t}\mathbf v\cdot\overline{\mathbf t}\,\mathrm dA.
```

The left side is internal virtual work. The right side is work from body force and
prescribed traction on boundary $\Gamma_t$. Integration by parts reduces the order
of spatial derivatives and exposes the traction boundary condition.

For a pressure test function $r$ (zero where pressure is prescribed), the gravity-free
fluid balance becomes

```math
\int_\Omega r\left(b\nabla\cdot\dot{\mathbf u}+\frac{\dot p}{M}\right)\,\mathrm dV
+\int_\Omega\nabla r\cdot\mathcal K\nabla p\,\mathrm dV
=\int_\Omega rs\,\mathrm dV-\int_{\Gamma_q}r\overline q_n\,\mathrm dA.
```

The quantity $\overline q_n$ is prescribed **outward** volume flux. Its minus sign reflects loss
of stored fluid. These general weak forms explain the terms that an implementation must
assemble; supported loading interfaces must still be checked for the chosen solver.

## The coupled matrix system

Substituting basis functions and collecting coefficients gives the linear Biot system

```math
\begin{bmatrix}\mathbf K&-\mathbf Q\\0&\mathbf H\end{bmatrix}
\begin{bmatrix}\mathbf u\\\mathbf p\end{bmatrix}
+\begin{bmatrix}0&0\\\mathbf Q^{\mathsf T}&\mathbf C_p\end{bmatrix}
\begin{bmatrix}\dot{\mathbf u}\\\dot{\mathbf p}\end{bmatrix}
=\begin{bmatrix}\mathbf f_u\\\mathbf f_p\end{bmatrix}.
```

Here $\mathbf K$ is elastic stiffness, $\mathbf Q$ couples pressure and displacement,
$\mathbf H$ represents Darcy transport, and $\mathbf C_p$ contains storage $1/M$.
The two block matrices are called `K1` and `K2` by [`assemble_biot_matrices`](@ref).
The opposite coupling signs agree with the stress and storage laws in chapter 4.

Backward Euler replaces a time derivative by
$\dot{\mathbf x}\simeq(\mathbf x^{n+1}-\mathbf x^n)/\Delta t$, giving

```math
(\mathbf K_1+\mathbf K_2/\Delta t)\mathbf x^{n+1}
=\mathbf f^{n+1}+\mathbf K_2\mathbf x^n/\Delta t.
```

This is an implicit step: the new state appears on both the storage and transport terms.
[`solve_biot`](@ref) uses this formulation. The worked Cartesian cases use plane strain;
radial elements separately account for circumferential strain and geometric integration
weights. Reducing a problem to one spatial coordinate does not remove these geometric terms.
Displacement and pressure interpolation must also be chosen compatibly; arbitrary choices
can give spurious pressure patterns, particularly near the undrained limit.

## What to check before trusting a result

First check units, pressure reference, and boundary signs. Then check that mass changes
match integrated sources and boundary flows. Compare to a limiting case or analytical
solution, and reduce mesh size and time step separately to identify spatial and temporal
error. A converged nonlinear solve only means that the discrete equations were solved;
it does not show that the mesh is accurate or the material assumptions appropriate.

The [validation pages](../validation/terzaghi.md) provide concrete examples of these
comparisons. To connect parameter changes to output changes, continue with
[parameter identification](../demos/parameter_identification.md) and
[differentiating a solve](../demos/solver_sensitivity.md). Differentiability is tested
for selected paths; it should not be assumed for every backend and coupled model.
