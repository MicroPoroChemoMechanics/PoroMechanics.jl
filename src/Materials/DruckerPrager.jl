"""
Drucker-Prager perfect plasticity with a non-associated flow rule.

The cone criterion for a frictional, cohesive material:

```math
f(\\boldsymbol{\\sigma}) = q + f_f\\, p - c_c, \\qquad
p = \\tfrac{1}{3}\\mathrm{tr}\\,\\boldsymbol{\\sigma}, \\qquad
q = \\sqrt{3 J_2}
```

with the coefficients that match the Mohr-Coulomb cone through the compressive meridian,

```math
f_f = \\frac{6\\sin\\varphi}{3 - \\sin\\varphi}, \\qquad
d_d = \\frac{6\\sin\\psi}{3 - \\sin\\psi}, \\qquad
c_c = \\frac{6\\cos\\varphi}{3 - \\sin\\varphi}\\, c
```

``\\varphi`` is the friction angle, ``\\psi`` the dilatancy angle and ``c`` the cohesion.
Taking ``\\psi < \\varphi`` makes the flow non-associated, which is what a real soil does:
it shears without dilating as much as an associated rule would predict.

**The mean stress is tension-positive here**, unlike [`BBM`](@ref), which counts a
compression positive. Drucker-Prager is universally written this way, and so is the code
this package is checked against, so the convention follows the criterion rather than its
neighbor in `src/Materials/`. A compressive state has ``p < 0`` and sits comfortably inside
the cone; the apex is at ``p = c_c / f_f`` on the tensile side.
"""

"""
    DruckerPrager(; E, nu, cohesion, friction, dilatancy)

Perfect Drucker-Prager plasticity on isotropic linear elasticity.

| field | meaning |
|---|---|
| `elastic` | the elastic law, a [`LinearElastic`](@ref) |
| `cohesion` | ``c`` [Pa] |
| `friction` | ``\\varphi`` [rad] |
| `dilatancy` | ``\\psi`` [rad] |

Angles are stored in **radians**. Deck files usually quote degrees — convert at the call
site rather than inside the model, so the units of a stored parameter are never in doubt.

Every coefficient is type-parameterized, so a `ForwardDiff.Dual` can enter the cohesion or
the friction angle and a result can be differentiated with respect to them.
"""
struct DruckerPrager{T, E <: LinearElastic} <: AbstractMaterial
    elastic::E
    cohesion::T
    friction::T
    dilatancy::T
end

function DruckerPrager(elastic::LinearElastic, cohesion, friction, dilatancy)
    c, φ, ψ = promote(cohesion, friction, dilatancy)
    return DruckerPrager{typeof(c), typeof(elastic)}(elastic, c, φ, ψ)
end

function DruckerPrager(; E, nu, cohesion, friction, dilatancy)
    return DruckerPrager(LinearElastic(; E = E, nu = nu), cohesion, friction, dilatancy)
end

"""
    DruckerPragerState(σ, ε, εp, γp)

Stress, total strain, plastic strain and the cumulative plastic shear strain
``\\gamma_p = \\int \\dot\\lambda``.

`γp` is carried even though perfect plasticity does not use it: it is the measure a
softening or hardening cohesion is a function of, and a state that cannot report how much
plastic straining it has been through is useless for anything but the perfect case.

Each tensor carries its **own** type parameter. That is not pedantry: differentiating with
respect to a material parameter — the cohesion, say — makes the stress and the plastic
strain `Dual` while the imposed strain stays `Float64`, and a state that demanded one shared
type would reject exactly the case this package exists to support.
"""
struct DruckerPragerState{S, E, P, T, Z} <: AbstractMaterialState
    σ::S
    ε::E
    εp::P
    γp::T
    σ0::Z
end

## A state built without a prestress starts from zero, which is what a laboratory path does.
DruckerPragerState(σ, ε, εp, γp) = DruckerPragerState(σ, ε, εp, γp, zero(σ))

function initial_state(m::DruckerPrager, σ0::Tensors.SymmetricTensor{2, 3, T}) where {T}
    z = zero(σ0)
    ## The initial stress is both the current stress and the prestress: at zero strain the
    ## predictor must reproduce it exactly.
    return DruckerPragerState(σ0, z, z, zero(T), σ0)
end

initial_state(m::DruckerPrager) = initial_state(m, zero(Tensors.SymmetricTensor{2, 3, eltype(m.elastic)}))

"""
    friction_coefficient(m::DruckerPrager) -> f_f

``6\\sin\\varphi / (3 - \\sin\\varphi)``, the slope of the cone in the ``(p, q)`` plane.
"""
friction_coefficient(m::DruckerPrager) = 6 * sin(m.friction) / (3 - sin(m.friction))

"""
    dilatancy_coefficient(m::DruckerPrager) -> d_d

``6\\sin\\psi / (3 - \\sin\\psi)``, the volumetric part of the flow direction. Equal to
`friction_coefficient` exactly when the rule is associated.
"""
dilatancy_coefficient(m::DruckerPrager) = 6 * sin(m.dilatancy) / (3 - sin(m.dilatancy))

"""
    cohesion_intercept(m::DruckerPrager) -> c_c

``6\\cos\\varphi\\,c / (3 - \\sin\\varphi)``, the value of ``q`` at which the cone crosses
``p = 0``.
"""
cohesion_intercept(m::DruckerPrager) = 6 * cos(m.friction) * m.cohesion / (3 - sin(m.friction))

"""
    apex_pressure(m::DruckerPrager) -> p_apex

The mean stress at which the cone closes, ``c_c / f_f``. A stress path that would leave the
cone through its tip returns here instead of to its side; see [`material_response`](@ref).

Infinite for a frictionless material, which is the von Mises limit and has no apex.
"""
function apex_pressure(m::DruckerPrager)
    ff = friction_coefficient(m)
    return cohesion_intercept(m) / ff
end

"""
    yield_function(m::DruckerPrager, σ) -> f

``f = q + f_f p - c_c``. Negative inside the cone, zero on it.
"""
function yield_function(m::DruckerPrager, σ::Tensors.SymmetricTensor{2})
    p = Tensors.tr(σ) / 3
    s = Tensors.dev(σ)
    q = sqrt(3 * (s ⊡ s) / 2)
    return q + friction_coefficient(m) * p - cohesion_intercept(m)
end

"""
    drucker_prager_return(m, ε, εp_n, σ0 = zero(ε)) -> (σ, εp, Δγ, at_apex)

The elastic predictor and its plastic corrector, as a pure function of the strain.

`σ0` is a **prestress**: the stress the material already carries at zero strain. Geotechnical
cases are posed this way — `base/Poroplast` starts at −11.5 MPa isotropic with the mesh
undeformed — and folding it into the predictor is the only place it belongs, because the
yield check has to see the *total* stress and not the increment.

Perfect plasticity with a linear elastic predictor makes the smooth return closed-form:
consistency on ``f`` after the return gives

```math
\\Delta\\gamma = \\frac{f^{tr}}{3G + K f_f d_d}
```

so the returned state follows in one step. No inner Newton is needed, which is the whole
difference in cost between this and [`BBM`](@ref).

Two returns exist and both are needed. The **smooth** one slides the trial stress back onto
the side of the cone; it is valid only while the corrected ``q`` stays positive. When the
trial state is so tensile that ``q^{tr} - 3G\\Delta\\gamma`` would go negative, the correct
projection is onto the **apex**, where the cone has no unique normal and the deviator
vanishes entirely. Omitting the apex return is the classic way to get a Drucker-Prager
implementation that passes every test until a tensile corner appears, and then returns a
negative ``q``.
"""
function drucker_prager_return(
        m::DruckerPrager, ε::Tensors.SymmetricTensor{2, 3}, εp_n, σ0 = zero(ε)
    )
    C = elastic_stiffness(m.elastic, Val(3))
    K = m.elastic.λ + 2 * m.elastic.μ / 3
    G = m.elastic.μ

    ff = friction_coefficient(m)
    dd = dilatancy_coefficient(m)
    cc = cohesion_intercept(m)

    ## Predict from the previous *plastic* strain rather than the previous stress: the two
    ## agree only while nothing has drifted, and the plastic strain is the state variable.
    σ_tr = σ0 + C ⊡ (ε - εp_n)
    p_tr = Tensors.tr(σ_tr) / 3
    s_tr = Tensors.dev(σ_tr)
    q_tr = sqrt(3 * (s_tr ⊡ s_tr) / 2)
    f_tr = q_tr + ff * p_tr - cc
    T = typeof(f_tr)

    f_tr <= zero(T) && return σ_tr, εp_n, zero(T), false

    Δγ = f_tr / (3 * G + K * ff * dd)
    q_new = q_tr - 3 * G * Δγ

    if q_new >= zero(T)
        n = s_tr / q_tr                       # unit direction of the deviator
        p_new = p_tr - K * dd * Δγ
        σ_new = q_new * n + p_new * one(σ_tr)
        εp_new = εp_n + Δγ * (3 * n / 2 + dd * one(σ_tr) / 3)
        return σ_new, εp_new, Δγ, false
    end

    ## Apex return: the whole deviator is shed and the stress collapses onto the tip.
    ##
    ## The elastic strain that carries a purely spherical stress is spherical too, so it is
    ## written down rather than solved for. Inverting `C` here would also break the
    ## differentiation: `C` is built from the material's own scalars while `ε` may carry a
    ## `Dual`, and `C \\ σ` has no promoting method for that pair.
    p_apex = cc / ff
    σ_new = p_apex * one(σ_tr)
    εp_new = ε - ((p_apex - Tensors.tr(σ0) / 3) / (3 * K)) * one(σ_tr)
    Δεp_dev = Tensors.dev(εp_new - εp_n)
    return σ_new, εp_new, sqrt(2 * (Δεp_dev ⊡ Δεp_dev) / 3), true
end

"""
    material_response(m::DruckerPrager, ε, state, Δt) -> (σ, ∂σ∂ε, state)

The stress, the consistent tangent and the updated state.

The tangent is **not written out by hand**: it is `Tensors.gradient` of
[`drucker_prager_return`](@ref), so it is the derivative of the return actually performed
rather than of the one that was meant to be. That is the same rule the finite volume models
follow — the algorithmic tangent of a non-associated cone is exactly the kind of expression
whose sign errors survive every test that only checks the stress.

One branch is treated differently, on purpose. At the apex the returned stress is a
constant, so the true algorithmic tangent is **zero** and a global stiffness assembled from
it is singular. The elastic stiffness stands in there. It costs Newton its quadratic rate on
steps that hit the tip; saying so is better than shipping a matrix that cannot be factored.
"""
function material_response(
        m::DruckerPrager, ε::Tensors.SymmetricTensor{2, 3}, state::DruckerPragerState, Δt
    )
    εp_n, σ0 = state.εp, state.σ0
    ∂σ∂ε, σ = Tensors.gradient(e -> first(drucker_prager_return(m, e, εp_n, σ0)), ε, :all)
    _, εp_new, Δγ, at_apex = drucker_prager_return(m, ε, εp_n, σ0)
    C = at_apex ? elastic_stiffness(m.elastic, Val(3)) : ∂σ∂ε
    return σ, C, DruckerPragerState(σ, ε, εp_new, state.γp + Δγ, σ0)
end

# ── Pairing a plastic skeleton with Biot's coupling ───────────────────────────

"""
    BiotPlastic(; skeleton, b, beta, N, k, mu_l)

A Biot medium whose skeleton is an arbitrary elastoplastic material.

[`BiotPoroelastic`](@ref) hard-wires a `LinearElastic` skeleton, which is what makes
`poro_response` short and what stops it short of `base/Poroplast`: that case is a
Drucker-Prager skeleton under a Biot coupling, and no shipped date of it is elastic. This
struct is the same hinge with the skeleton left open.

| field | meaning |
|---|---|
| `skeleton` | the drained stress–strain law — any `AbstractMaterial` |
| `b` | Biot coefficient of the stress coupling and of the elastic porosity change [-] |
| `beta` | coefficient with which the **plastic** volume change enters the porosity [-] |
| `N` | storage modulus at constant strain [Pa⁻¹] |
| `k` | intrinsic permeability [m²] |
| `mu_l` | dynamic viscosity [Pa·s] |

`b` and `beta` are separate because Bil's `Poroplast` keeps them separate: its porosity is

```math
\\phi = \\phi_0 + b\\,(\\mathrm{tr}\\,\\varepsilon - \\mathrm{tr}\\,\\varepsilon^p)
      + N (p_l - p_{l0}) + \\beta\\,\\mathrm{tr}\\,\\varepsilon^p
```

so a plastic volume change opens pore space at a different rate than an elastic one. They
coincide in `base/Poroplast`, where both are 0.8, and a comparison that assumed they always
do would pass there and be wrong everywhere else.
"""
Base.@kwdef struct BiotPlastic{M <: AbstractMaterial, T} <: AbstractMaterial
    skeleton::M
    b::T = 1.0
    beta::T = 1.0
    N::T = 0.0
    k::T = 1.0e-19
    mu_l::T = 1.0e-3
end

function BiotPlastic(skeleton::AbstractMaterial, b, beta, N, k, mu_l)
    return BiotPlastic(skeleton, promote(b, beta, N, k, mu_l)...)
end

skeleton(m::BiotPlastic) = m.skeleton
Base.eltype(::BiotPlastic{M, T}) where {M, T} = T

initial_state(m::BiotPlastic, args...; kwargs...) =
    initial_state(m.skeleton, args...; kwargs...)

"""
    poro_response(m::BiotPlastic, ε, p, state, Δt) -> (σ, ∂σ∂ε, ∂σ∂p, state)

Total stress of the medium and its tangents, from the skeleton's own response.

Identical in shape to the [`BiotPoroelastic`](@ref) method — ask the skeleton, then add the
pressure coupling — which is the point: the poromechanical layer does not care whether the
skeleton yielded, and a plastic skeleton needs no new coupling code.
"""
function poro_response(m::BiotPlastic, ε::Tensors.SymmetricTensor{2}, p, state, Δt)
    σ_eff, ∂σ∂ε, state_new = material_response(m.skeleton, ε, state, Δt)
    ## `beta`, not `b`. Bil converts between the two stresses with the coefficient it calls
    ## beta — `sig += beta*pl` before its return mapping and `sig -= beta*pl` after — while
    ## `b` appears in its incremental update and in the elastic part of the porosity. The two
    ## are equal in `base/Poroplast`, so the choice is invisible there; it is written down
    ## because the deck that separates them is the one that would expose a wrong guess.
    σ = total_stress(m.beta, σ_eff, p)
    ∂σ∂p = -m.beta * one(σ_eff)
    return σ, ∂σ∂ε, ∂σ∂p, state_new
end

"""
    porosity(m::BiotPlastic, phi0, ε, εp, p, p0) -> φ

Lagrangian porosity, following Bil's `Poroplast`:

```math
\\phi = \\phi_0 + b\\,(\\mathrm{tr}\\,\\varepsilon - \\mathrm{tr}\\,\\varepsilon^p)
      + N (p_l - p_{l0}) + \\beta\\,\\mathrm{tr}\\,\\varepsilon^p
```

Splitting the volumetric strain into its elastic and plastic parts, each with its own
coefficient, is what `b` and `beta` are for. With `beta == b` this collapses to the usual
``\\phi_0 + b\\,\\mathrm{tr}\\,\\varepsilon + N\\,\\Delta p``.
"""
function porosity(m::BiotPlastic, phi0, ε, εp, p, p0)
    tr_e = Tensors.tr(ε)
    tr_ep = Tensors.tr(εp)
    return phi0 + m.b * (tr_e - tr_ep) + m.N * (p - p0) + m.beta * tr_ep
end
