# Richards Equation 1D — Barrier Imbibition

> **Source:** [`examples/M1_Richards/run.jl`](https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl/blob/main/examples/M1_Richards/run.jl)
>
> **Reference solution:** Richards 1D with Van Genuchten retention.
>
> **Reference test case:** material "bo", horizontal imbibition

---

## Physical problem

We simulate the progressive imbibition of an initially dry containment barrier.
The liquid pressure $p_l$ [Pa] is the unknown.

**PDE:**

$$\rho_l \phi \frac{\partial S_l(p_c)}{\partial t} + \nabla \cdot W_l = 0$$

with the unsaturated Darcy flux:

$$W_l = -K_l \nabla p_l + K_l \rho_l g, \qquad K_l = \frac{\rho_l k_\text{int} k_{rl}(p_c)}{\mu_l}$$

and the capillary pressure $p_c = p_g - p_l$.

**Boundary conditions:**

| Boundary | Condition |
|---|---|
| $x = 0$ (left) | Zero Neumann — impermeable boundary |
| $x = L$ (right) | Dirichlet $p_l = p_g$ — full saturation imposed |

**Initial condition:** $p_l = -7.611\,930 \times 10^7$ Pa (dry state).

---

## Van Genuchten / Mualem retention curves

**Liquid saturation $S_l(p_c)$:**

$$S_l(p_c) = \begin{cases} 1 & \text{if } p_c \leq 0 \\ \left(1 + \left(\dfrac{p_c}{a_{S_l}}\right)^n\right)^{-m_{S_l}} & \text{if } p_c > 0 \end{cases}, \quad n = \frac{1}{1 - m_{S_l}}$$

**Relative permeability $k_{rl}(p_c)$ — Mualem:**

$$k_{rl}(p_c) = \sqrt{S_e} \left(1 - \left(1 - S_e^{1/m_{krl}}\right)^{m_{krl}}\right)^2, \quad S_e = \left(1 + \left(\frac{p_c}{a_{krl}}\right)^n\right)^{-m_{krl}}$$

---

## Parameters (material "bo")

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $\phi$ | $0.30$ | — | Porosity |
| $\rho_l$ | $10^3$ | kg/m³ | Liquid density |
| $k_\text{int}$ | $10^{-20}$ | m² | Intrinsic permeability |
| $\mu_l$ | $10^{-3}$ | Pa·s | Dynamic viscosity |
| $p_g$ | $10^5$ | Pa | Gas pressure |
| $a_{S_l}$ | $1.5 \times 10^6$ | Pa | Van Genuchten parameter ($S_l$ curve) |
| $m_{S_l}$ | $0.06$ | — | Van Genuchten exponent ($S_l$ curve) |
| $a_{krl}$ | $3.0 \times 10^6$ | Pa | Van Genuchten parameter ($k_{rl}$ curve) |
| $m_{krl}$ | $0.5$ | — | Mualem exponent ($k_{rl}$ curve) |

---

## PoroMechanics.jl implementation

```julia
Base.@kwdef struct M1RichardsModel <: AbstractPoroModel
    phi   :: Float64 = 0.30
    rho_l :: Float64 = 1.0e3
    k_int :: Float64 = 1.0e-20
    mu_l  :: Float64 = 1.0e-3
    p_g   :: Float64 = 1.0e5
    gravity :: Float64 = 0.0    # horizontal → zero gravity
    a_Sl  :: Float64 = 1.5e6;   m_Sl  :: Float64 = 0.06
    a_krl :: Float64 = 3.0e6;   m_krl :: Float64 = 0.5
end
```

### Type-stable constitutive laws (ForwardDiff)

VoronoiFVM computes the Jacobian via `ForwardDiff.jl`: the constitutive functions
must be **type-stable** for `Dual` types (dual numbers).
The `zero(pc)` pattern preserves the type of `pc` (Float64 or Dual).

```julia
function _Sl(pc, a, m)
    pc ≤ 0 && return 1.0 + zero(pc)   # saturated: S_l = 1 + 0·Dual
    n = 1.0 / (1.0 - m)
    return (1.0 + (pc / a)^n)^(-m)
end

function _krl(pc, a, m)
    pc ≤ 0 && return 1.0 + zero(pc)
    n  = 1.0 / (1.0 - m)
    Se = clamp((1.0 + (pc / a)^n)^(-m), 0.0, 1.0)
    # ForwardDiff guard: d/dSe[√Se] = 1/(2√Se) → ∞ as Se → 0.
    # Dry zone (Se ≈ 0): k_rl ≈ 0 physically → return 0 directly.
    Se < 1e-14 && return zero(pc)
    arg = clamp(1.0 - Se^(1.0 / m), 0.0, 1.0)
    return sqrt(Se) * (1.0 - arg^m)^2
end
```

!!! warning "ForwardDiff pitfall — `sqrt(Se)` near zero"
    `d/dSe[√Se] = 1/(2√Se)` diverges as `Se → 0`.
    Without the `Se < 1e-14` guard, ForwardDiff computes `NaN` in very dry zones,
    which terminates Newton with the error *"trying to assemble NaN"*.

### Callbacks

```julia
# Richards flux (with optional gravity)
function PoroMechanics.flux!(f, u, edge, m::M1RichardsModel, ::Any)
    pl1, pl2 = u[1, 1], u[1, 2]
    pc_avg   = m.p_g - (pl1 + pl2) / 2
    Kl_avg   = Kl(m, pc_avg)
    dx       = edge.coord[1, 2] - edge.coord[1, 1]
    f[1]     = Kl_avg * (pl1 - pl2) + Kl_avg * m.rho_l * m.gravity * dx
end

# Storage: ρ_l φ S_l(p_c)
function PoroMechanics.storage!(f, u, ::Any, m::M1RichardsModel, ::Any)
    f[1] = m.rho_l * m.phi * Sl(m, m.p_g - u[1])
end

# Right BC: Dirichlet p_l = p_g (saturation)
function PoroMechanics.bcondition!(f, u, bnode, m::M1RichardsModel, ::Any)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 2, value = m.p_g)
end
```

---

## Time-step control

This problem is **very stiff** ($k_\text{int} = 10^{-20}$ m² — compacted clay):
the hydraulic diffusivity is extremely low and the imbibition front advances slowly.

The `Δu_opt` parameter of `SolverControl` controls the maximum variation of $p_l$ per
time step and drives automatic adaptation:

| `Δu_opt` | % of the $p_l$ range (7.7×10⁷ Pa) | Time steps for 10 years |
|---|---|---|
| $10^5$ Pa | 0.13 % | ~27 000 (too slow) |
| **$10^6$ Pa** | **1.3 %** | **~2 700 (accuracy/speed trade-off)** |

```julia
ctrl = VoronoiFVM.SolverControl(;
    Δt      = 1.0e6,   # initial step [s]
    Δt_max  = 3.154e7, # 1 year [s]
    Δt_min  = 1.0,
    Δu_opt  = 1.0e6,   # max variation of p_l per step [Pa]
    reltol  = 1.0e-4,
    abstol  = 1.0e-8,
)
```

---

## Expected results (10 years, 101-node mesh)

The imbibition front progresses from right to left.
The liquid pressure $p_l$ increases (becomes less negative) and saturation $S_l$ grows.

```
t [years]      | p_l[x=0] [Pa]   | p_l[mid] [Pa]   | S_l[mid] [-]
────────────────────────────────────────────────────────────────────
0.0000         | -7.6119e+07     | -7.6119e+07     | 0.780000
1.0            | -7.6119e+07     | -7.6119e+07     | ≈ 0.780
...
10.0           | -7.6119e+07     | < -7.6119e+07   | > 0.780

Physical check (x = midpoint):
  p_l : -7.6119e+07 → -X.XXXe+07 Pa    (increasing)
  S_l : 0.780000 → Y.YYYYYY            (increasing)
  ✓ p_l increases (imbibition confirmed)
```

---

## Key points

- **Extreme stiffness** — $k_\text{int} = 10^{-20}$ m² imposes very long characteristic
  times (decadal scale) but very steep local gradients near the front.
- **ForwardDiff type-stability** — essential for `_Sl` and `_krl`: use
  `zero(pc)` in early returns and protect `sqrt(Se)` against zero values.
- **`Δu_opt`** — set to ~1 % of the unknown's variation range; too restrictive
  unnecessarily multiplies the number of time steps.
- **No gravity** — the test case is horizontal ($g = 0$), which isolates the pure
  capillary physics.
