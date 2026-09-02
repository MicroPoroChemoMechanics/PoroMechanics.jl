# Getting Started

A model is a struct plus the callbacks its physics needs. Nothing else is registered or
declared: dispatch on the struct is what ties the two together.

This page builds the simplest one — diffusion of a tracer through a saturated porous
medium — and solves it.

The package already ships this equation as [`FickModel`](@ref), with per-region
coefficients and its boundary data in a field; it is written out here from nothing because
the point of the page is the mechanism, not the equation. `TracerModel` is the name used
below so that the two do not collide.

## The equation

```math
\varphi \frac{\partial c}{\partial t} = \nabla \cdot (D \varphi \nabla c)
\qquad \text{on } [0, L]
```

with a concentration imposed at ``x = 0`` and a sealed face at ``x = L``.

## The model

```@example quickstart
using PoroMechanics
using VoronoiFVM
using ExtendableGrids

Base.@kwdef struct TracerModel <: AbstractPoroModel
    φ::Float64    = 0.30     # porosity [-]
    D::Float64    = 1e-10    # effective diffusion coefficient [m²/s]
    c_in::Float64 = 1.0      # concentration imposed at x = 0 [mol/m³]
end
nothing # hide
```

Three callbacks describe it. `storage!` is the accumulation term, `flux!` the flux between
two neighboring control volumes, `bcondition!` the boundary conditions:

```@example quickstart
PoroMechanics.storage!(f, u, node, m::TracerModel, data) = (f[1] = m.φ * u[1])

PoroMechanics.flux!(f, u, edge, m::TracerModel, data) = (f[1] = m.D * m.φ * (u[1, 1] - u[1, 2]))

function PoroMechanics.bcondition!(f, u, bnode, m::TracerModel, data)
    boundary_dirichlet!(f, u, bnode; species = 1, region = 1, value = m.c_in)
end

PoroMechanics.nspecies(::TracerModel) = 1
PoroMechanics.species_names(::TracerModel) = [:c]
nothing # hide
```

The last two are how the model says how many unknowns it carries and what they are called;
[`fvm_system`](@ref) reads the species count off `nspecies` rather than being told again.

`flux!` returns the difference between the two node values, not a gradient: VoronoiFVM
divides by the edge length itself. The sealed face needs no code — a zero Neumann
condition is what happens when nothing is imposed.

## Solving

```@example quickstart
m    = TracerModel()
grid = simplexgrid(range(0, 1.0; length = 101))

sys = fvm_system(m, grid)

inival = unknowns(sys; inival = 0.0)
inival[1, 1] = m.c_in     # keep the initial state consistent with the boundary value

control = VoronoiFVM.SolverControl(; Δt = 1.0e4, Δt_max = 1.0e7, Δu_opt = 0.1)
tsol = solve(sys; inival, times = (0.0, 1.0e8), control)

(length(tsol.t), maximum(tsol[1, :, end]))
```

That initial assignment matters. Without it the Dirichlet node jumps from `0` to `c_in`
over the first step; the step-size controller sees a change it cannot reduce by shrinking
`Δt`, and halves the step down to `Δt_min` before giving up.

`Δu_opt` is worth setting too. It is the concentration change the controller aims for per
step, in the units of the unknown — leave it at its default and a problem whose time scale
is 10⁸ s is integrated in steps sized for something else entirely.

[`fvm_system`](@ref) is what ties the three methods to the solver: it wraps them in the
closures `VoronoiFVM.System` expects and reads the species count off the model, so the
three-line adapter never has to be written twice.

## Where the Jacobian went

Nowhere — there is none to write. `VoronoiFVM.solve` differentiates the callbacks above
with `ForwardDiff.jl` and runs its own Newton loop and adaptive time stepping. Keeping the
callbacks type-generic is what makes that work: annotate arguments as `Real` rather than
`Float64`, and return `zero(x)` rather than a bare `0.0` when short-circuiting, so dual
numbers pass through.

## Next

The [examples](examples/fickian_diffusion.md) carry one worked problem per physics — each with its
governing equations, its material data, and the reference solution it is checked against.
They use the models the package ships rather than defining their own, which is the split
this page exists to explain: [Writing a model](demos/writing_a_model.md) measures what that
choice costs and what it buys.
