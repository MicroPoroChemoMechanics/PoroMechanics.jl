# Biot Poroelasticity 2D — Ternay Dam (M7)

> **Source:** [`examples/M7_Biot/run.jl`](https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl/blob/main/examples/M7_Biot/run.jl)
>
> **Reference solution:** Biot poroelasticity on a gravity dam.
>
> **Reference test case:** Ternay gravity dam — hydromechanical loading

---

## Physical problem

We simulate the hydromechanical response of the Ternay gravity dam (concrete + rock
foundation) to a sudden reservoir filling.
The solid displacement $\mathbf{u}$ [m] and the pore pressure $p_l$ [Pa] are the unknowns.

**Governing equations (plane strain, zero gravity, saturated medium $S_l = 1$):**

*Mechanical equilibrium:*
```math
\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}, \qquad
  \boldsymbol{\sigma} = \lambda \operatorname{tr}(\boldsymbol{\varepsilon})\,\mathbf{I}
  + 2\mu\,\boldsymbol{\varepsilon} - b\,p_l\,\mathbf{I}
```

*Liquid mass conservation (Biot):*
```math
\frac{\partial}{\partial t}\!\left(b\,\nabla\cdot\mathbf{u} + N p_l\right)
  - \nabla \cdot \left(\frac{k_\text{int}}{\mu_l}\,\nabla p_l\right) = 0
```

where $\boldsymbol{\varepsilon} = \tfrac{1}{2}(\nabla\mathbf{u} + \nabla^\top\mathbf{u})$ is
the infinitesimal strain tensor, $b$ the Biot coefficient, and $N$ [Pa⁻¹] the storage
modulus at constant strain.

**Boundary conditions:**

| Region | Location | Condition |
|---|---|---|
| 101–105, 121 | Upstream face | $p_l = \rho_l g (H - y)$ (hydrostatic) |
| 106–112, 125 | Downstream face | $p_l = 0$ (drainage) |
| 101–105, 121 | Upstream face | traction $\mathbf{t} = -p_\text{hydro}(y)\,\mathbf{n}$ |
| 122, 124 | Lateral rock faces | $u_1 = 0$ |
| 123 | Base of foundation ($y = 455$ m) | $u_1 = u_2 = 0$ (clamped) |

**Initial condition:** $\mathbf{u} = \mathbf{0}$, $p_l = 0$ (dry dam, instantaneous filling at $t=0$).

---

## Geometry and mesh

The geometry represents the Ternay dam cross-section (Loire valley, France).
The mesh `ternay.msh` is in GMSH 2.2 format: 479 nodes, 860 triangular elements,
physical surfaces `"1"` (concrete) and `"2"` (rock foundation),
boundary lines `"101"`–`"125"`.

| Physical group | Type | Description |
|---|---|---|
| `"1"` | Surface | Concrete dam body |
| `"2"` | Surface | Rock foundation |
| `"101"`–`"105"`, `"121"` | Line | Upstream face (amont) |
| `"106"`–`"112"`, `"125"` | Line | Downstream face (aval) |
| `"122"`, `"124"` | Line | Lateral rock boundaries |
| `"123"` | Line | Foundation base ($y = 455$ m NGF) |

The reservoir surface is at $H = 517$ m NGF; the upstream base is at $y \approx 476$ m,
giving a maximum hydrostatic pressure $p_\text{max} \approx 10^4 \times (517 - 476) = 0.41$ MPa.

---

## Material parameters

| Symbol | Concrete | Rock | Unit | Description |
|---|---|---|---|---|
| $E$ | $1.4 \times 10^{10}$ | $1.8 \times 10^{10}$ | Pa | Young's modulus |
| $\nu$ | $0.15$ | $0.15$ | — | Poisson's ratio |
| $k_\text{int}$ | $10^{-14}$ | $10^{-11}$ | m² | Intrinsic permeability |
| $\mu_l$ | $10^{-3}$ | $10^{-3}$ | Pa·s | Dynamic viscosity |
| $b$ | $0.4$ | $0.2$ | — | Biot coefficient |
| $N$ | $10^{-10}$ | $10^{-10}$ | Pa⁻¹ | Storage modulus |

**Characteristic consolidation times** ($c_v = \frac{k_\text{int}/\mu_l}{b^2/(\lambda+2\mu) + N}$, $T_c = L^2/c_v$):

| Material | $c_v$ [m²/s] | Reference length $L$ | $T_c$ |
|---|---|---|---|
| Concrete | $\approx 2 \times 10^{-4}$ | 10 m (dam thickness) | $\approx 8$ days |
| Rock | $\approx 2 \times 10^{-1}$ | 40 m (foundation depth) | $\approx 9$ s |

The rock consolidates in seconds; the concrete is the long-time bottleneck.

---

## PoroMechanics.jl implementation

### Model definition

All material and hydraulic parameters are stored in a single `M7Model` struct.

```julia
Base.@kwdef struct M7Model <: AbstractPoroModel
    # concrete (physical surface "1")
    E_b  :: Float64 = 1.4e10;  nu_b :: Float64 = 0.15
    k_b  :: Float64 = 1.0e-14; b_b  :: Float64 = 0.4;  N_b :: Float64 = 1.0e-10
    # rock (physical surface "2")
    E_r  :: Float64 = 1.8e10;  nu_r :: Float64 = 0.15
    k_r  :: Float64 = 1.0e-11; b_r  :: Float64 = 0.2;  N_r :: Float64 = 1.0e-10
    # fluid (shared)
    mu_l :: Float64 = 1.0e-3
    # hydraulic reference: p_l(y) = ρ_l·g·(H−y)
    H    :: Float64 = 517.0    # water surface elevation [m NGF]
    rho_g :: Float64 = 10_000.0 # ρ_l·g [Pa/m]
end

PoroMechanics.nspecies(::M7Model)      = 3   # u₁, u₂, p_l
PoroMechanics.species_names(::M7Model) = [:u1, :u2, :p]
```

### Discretisation: P1/P1 mixed elements

The problem is discretised on triangular elements with **P1 interpolation for both
displacement and pressure** (equal-order approximation). The `DofHandler` orders the
displacement DOFs first, then pressure:

```julia
ip_u = Lagrange{RefTriangle, 1}()^2   # 2D vector field
ip_p = Lagrange{RefTriangle, 1}()     # scalar field

dh = DofHandler(grid)
add!(dh, :u, ip_u)    # DOFs 1–6 per element (3 nodes × 2 components)
add!(dh, :p, ip_p)    # DOFs 7–9 per element
close!(dh)
```

### Two-matrix assembly

The linear Biot problem separates into a **stationary matrix** $K_1$ and a **storage
matrix** $K_2$.  At each time step $n$ the system is:

```math
\left(K_1 + \frac{K_2}{\Delta t}\right)\mathbf{x}^{n+1}
  = \mathbf{f}_\text{ext} + \frac{K_2}{\Delta t}\,\mathbf{x}^n
```

The element contributions are:

| Block | Matrix | Expression |
|---|---|---|
| $K_1[\mathbf{u},\mathbf{u}]$ | Elastic stiffness | $\int_\Omega \boldsymbol{\varepsilon}(\delta\mathbf{u}) : \mathbf{C} : \boldsymbol{\varepsilon}(\mathbf{u})\,d\Omega$ |
| $K_1[\mathbf{u},p]$ | Biot coupling (mech.) | $-b\int_\Omega (\nabla\cdot\delta\mathbf{u})\,p\,d\Omega$ |
| $K_1[p,p]$ | Darcy conductivity | $\frac{k_\text{int}}{\mu_l}\int_\Omega \nabla\delta p \cdot \nabla p\,d\Omega$ |
| $K_2[p,\mathbf{u}]$ | Biot coupling (hydr.) | $b\int_\Omega (\nabla\cdot\mathbf{u})\,\delta p\,d\Omega$ |
| $K_2[p,p]$ | Compressibility storage | $N\int_\Omega \delta p\,p\,d\Omega$ |

```julia
function PoroMechanics.element_matrices!(ke1, ke2, is_beton::Bool, m::M7Model, cv_u, cv_p)
    # select material parameters
    λ, μ = lame_coeffs(is_beton ? m.E_b : m.E_r, is_beton ? m.nu_b : m.nu_r)
    K_l  = (is_beton ? m.k_b : m.k_r) / m.mu_l
    b    = is_beton ? m.b_b : m.b_r
    N    = is_beton ? m.N_b : m.N_r

    for q in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, q)
        # ... K_uu, K_up, K_pp, K_pu, M_pp assembled at each quadrature point
    end
end
```

### Hydrostatic surface load

The upstream face receives a pressure traction $\mathbf{t} = -p_\text{hydro}(y)\,\mathbf{n}$,
assembled via `FacetIterator`:

```julia
function PoroMechanics.facet_load!(fe, facet, m::M7Model, fv_u)
    coords = getcoordinates(facet)
    for q in 1:getnquadpoints(fv_u)
        x  = spatial_coordinate(fv_u, q, coords)
        n  = getnormal(fv_u, q)
        dΓ = getdetJdV(fv_u, q)
        t  = -p_hydro(m, x[2]) * n   # inward push on the concrete face
        for i in 1:getnbasefunctions(fv_u)
            fe[i] += (shape_value(fv_u, q, i) ⋅ t) * dΓ
        end
    end
end
```

### Time loop

Backward Euler integration; the system matrix $A = K_1 + K_2/\Delta t$ is constant
(material linearity) and can be factored once per simulation:

```julia
res = run_M7(dt = 86400.0, n_steps = 30)  # 30 × 1-day steps
# res.x  → solution vector (1437 DOFs)
# res.dh → DofHandler for post-processing
```

---

## Expected results

### Short-time transient ($\Delta t = 100$ s, 20 steps → $t \approx 0.02$ days)

The rock reaches steady state almost instantly ($T_c^\text{rock} \approx 9$ s).
Displacements converge to their elastic limit within the first few steps.
The pore pressure maximum in the concrete equals the upstream boundary value
($p_\text{hydro}(y = 476\,\text{m}) = 0.41$ MPa) from step 1 — the Dirichlet
condition is imposed immediately.

```
 Step |    t [days] | p_max concrete [MPa] | u₁_max [mm] | u₂_max [mm]
──────────────────────────────────────────────────────────────────────
    1 |      0.0012 |              +0.4100 |    +14.67   |     +2.51
   10 |      0.0116 |              +0.4100 |    +14.84   |     +2.57
   20 |      0.0231 |              +0.4100 |    +14.84   |     +2.58   ← steady
```

### Long-time consolidation ($\Delta t = 1$ day, 30 steps → $t = 30$ days $\approx 4 T_c^\text{concrete}$)

The pore pressure field inside the concrete transitions from the initial
step-function profile to the steady-state Laplace solution (linear pressure
distribution from upstream to downstream). Displacements increase slightly as
the effective stress evolves.

```julia
res = run_M7(dt = 86400.0, n_steps = 30)
```

At $t \approx 30$ days ($\gg T_c^\text{concrete} \approx 8$ days), the pressure
field is close to the harmonic steady state $\Delta p_l = 0$ with upstream and
downstream Dirichlet values.

---

## Key points

- **Two-matrix split** — the linear Biot system separates cleanly into a stationary
  part $K_1$ (assembled once) and a storage part $K_2$ (assembled once). Only the
  right-hand side changes at each step. The system matrix $A = K_1 + K_2/\Delta t$
  can be factored once if $\Delta t$ is constant.
- **P1/P1 mixed elements** — equal-order interpolation for $\mathbf{u}$ and $p_l$ is
  not inf-sup stable in general, but works well here because the Biot storage term
  $N > 0$ regularises the pressure block (no spurious pressure modes).
- **`is_beton::Bool` dispatch** — the material type is determined from the Ferrite
  `cellset` at assembly time and passed directly to `PoroMechanics.element_matrices!` as a
  `Bool` flag, avoiding any runtime dictionary look-up.
- **FEM backend** — unlike the VoronoiFVM examples, this model uses
  [Ferrite.jl](https://ferrite-fem.github.io/Ferrite.jl/stable/) (FEM) instead of
  VoronoiFVM (FVM). The `PoroMechanics.element_matrices!` / `PoroMechanics.facet_load!` interface replaces
  `storage!` / `flux!` / `bcondition!`.
- **Verification** — at $t \to \infty$, the pore pressure satisfies the Laplace
  equation $\Delta p_l = 0$ with the upstream/downstream Dirichlet values. The
  displacement field then represents the fully drained elastic response of the dam.
