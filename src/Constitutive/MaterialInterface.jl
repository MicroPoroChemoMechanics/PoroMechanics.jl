"""
The stress–strain interface: what a material must answer when asked for its response.

Two layers on purpose.

The **mechanical** layer is deliberately poromechanics-free. Its signature is the one the
Ferrite ecosystem already uses (`MaterialModelsBase.jl`),

```julia
material_response(mat, ε, state, Δt) -> (σ, ∂σ∂ε, state_new)
```

so a Barcelona Basic Model written against it is usable from *any* Ferrite code, not only
from this package. Nothing in it knows about pore pressure.

The **poromechanical** layer sits on top: the total stress of a Biot medium is the skeleton
response less the pressure term, ``\\sigma = \\sigma' - b\\,p\\,\\mathbf{I}``, and the
coupling is a known, separate contribution. Keeping the two apart is what lets the
constitutive models travel.
"""

"""
    AbstractMaterial

Supertype of the stress–strain models. A concrete material implements
[`material_response`](@ref) and [`initial_state`](@ref).
"""
abstract type AbstractMaterial end

"""
    AbstractMaterialState

Supertype of the internal-variable containers. States are immutable: a material returns a
*new* state rather than mutating the one it was given, which keeps the response a pure
function of its arguments and therefore differentiable.
"""
abstract type AbstractMaterialState end

"""
    NoState()

The state of a material that has none — linear elasticity, for instance.
"""
struct NoState <: AbstractMaterialState end

"""
    initial_state(mat) -> state

The internal variables a material starts from. Called once per quadrature point.
"""
initial_state(::AbstractMaterial) = NoState()

"""
    material_response(mat, ε, state, Δt) -> (σ, ∂σ∂ε, state_new)

Stress, consistent tangent and updated internal variables for the strain increment ending
at `ε` over a step `Δt`.

- `ε` is a `SymmetricTensor{2,dim}`, `σ` likewise, and `∂σ∂ε` a `SymmetricTensor{4,dim}`.
- `state` is whatever [`initial_state`](@ref) returned, and `state_new` replaces it.

For a rate-independent material `Δt` is unused but kept in the signature: it is what a
viscoplastic or a creeping material needs, and changing the signature later would break
every model written against it.
"""
function material_response(mat::AbstractMaterial, ε, state, Δt)
    return error("material_response not implemented for $(typeof(mat))")
end

# ── Linear elasticity ─────────────────────────────────────────────────────────

"""
    LinearElastic(λ, μ)
    LinearElastic(; E, nu)

Isotropic linear elasticity, ``\\sigma = \\lambda\\,\\mathrm{tr}(\\varepsilon)\\mathbf{I}
+ 2\\mu\\,\\varepsilon``.

Type-parameterized like every other law here, so a response can be differentiated with
respect to `λ` or `μ` and not only with respect to the strain.
"""
struct LinearElastic{T} <: AbstractMaterial
    λ::T
    μ::T
end

LinearElastic(λ, μ) = LinearElastic(promote(λ, μ)...)
LinearElastic(; E, nu) = LinearElastic(
    E * nu / ((1 + nu) * (1 - 2nu)),
    E / (2 * (1 + nu)),
)

Base.eltype(::LinearElastic{T}) where {T} = T

"""
    elastic_stiffness(mat, ::Val{dim}) -> SymmetricTensor{4,dim}

The isotropic stiffness ``C_{ijkl} = \\lambda\\delta_{ij}\\delta_{kl}
+ \\mu(\\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk})``.
"""
function elastic_stiffness(mat::LinearElastic, ::Val{dim}) where {dim}
    λ, μ = mat.λ, mat.μ
    δ(i, j) = i == j ? one(λ) : zero(λ)
    return Tensors.SymmetricTensor{4, dim}(
        (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    )
end

function material_response(
        mat::LinearElastic, ε::Tensors.SymmetricTensor{2, dim}, state, Δt
    ) where {dim}
    σ = mat.λ * Tensors.tr(ε) * one(ε) + 2 * mat.μ * ε
    return σ, elastic_stiffness(mat, Val(dim)), state
end

# ── The poromechanical layer ──────────────────────────────────────────────────

"""
    skeleton(m::BiotPoroelastic) -> LinearElastic

The drained skeleton of a Biot material, as a stress–strain model in its own right. This is
the hinge between the two layers: everything below it is ordinary solid mechanics, and
everything above it is poromechanics.
"""
function skeleton(m::BiotPoroelastic)
    λ, μ = lame(m)
    return LinearElastic(λ, μ)
end

"""
    total_stress(b, σ_eff, p) -> σ

Total Cauchy stress of a poroelastic medium, ``\\sigma = \\sigma' - b\\,p\\,\\mathbf{I}``,
from the effective (skeleton) stress and the pore pressure. `b` is the Biot coefficient.

Sign convention: tension positive, so a positive pore pressure relieves the skeleton.
"""
total_stress(b, σ_eff::Tensors.SymmetricTensor{2}, p) = σ_eff - b * p * one(σ_eff)

"""
    poro_response(m::BiotPoroelastic, ε, p, state, Δt) -> (σ, ∂σ∂ε, ∂σ∂p, state_new)

Total stress of a Biot medium and its tangents, obtained by asking the skeleton for its
mechanical response and adding the pressure coupling.

`∂σ∂p = -b I` is constant for linear Biot, but it is returned rather than assumed so that a
model with a stress-dependent Biot coefficient can be dropped in without changing the
callers.
"""
function poro_response(
        m::BiotPoroelastic, ε::Tensors.SymmetricTensor{2, dim}, p, state, Δt
    ) where {dim}
    σ_eff, ∂σ∂ε, state_new = material_response(skeleton(m), ε, state, Δt)
    σ = total_stress(m.b, σ_eff, p)
    ∂σ∂p = -m.b * one(σ_eff)
    return σ, ∂σ∂ε, ∂σ∂p, state_new
end

# ── Pressure-dependent elasticity ─────────────────────────────────────────────

"""
    LogarithmicElasticState(σ, ε)

Internal variables of [`LogarithmicElastic`](@ref): the stress and strain the material
carries from the previous step. A hypoelastic law needs them because its stiffness depends
on where it currently sits, so the response is an increment rather than a function of the
total strain.
"""
struct LogarithmicElasticState{S, E} <: AbstractMaterialState
    σ::S
    ε::E
end

"""
    LogarithmicElastic(; κ, nu, e0, p_min)

Elasticity with a bulk modulus proportional to the mean effective compressive stress,

```math
K = \\frac{\\bar p\\,(1+e_0)}{\\kappa}, \\qquad
G = \\frac{3K(1-2\\nu)}{2(1+\\nu)}, \\qquad
\\bar p = -\\tfrac{1}{3}\\mathrm{tr}(\\sigma)
```

the elastic law of critical-state soil mechanics, and the elastic part of the Barcelona
Basic Model. `κ` is the swelling index, `e0` the initial void ratio.

Tension positive throughout, so ``\\bar p > 0`` in compression. Under tension the modulus
would go to zero or negative, which is unphysical and would break the solve, so `p_min`
floors it: a soil that is genuinely in tension is outside this law's range and the floor
says so rather than producing a plausible number.

Unlike [`LinearElastic`](@ref), this material has a state and its tangent changes with it —
which is what makes it the honest rehearsal for a return mapping.
"""
Base.@kwdef struct LogarithmicElastic{T} <: AbstractMaterial
    κ::T = 0.02          # swelling index [-]
    nu::T = 0.3          # Poisson's ratio [-]
    e0::T = 0.9          # initial void ratio [-]
    p_min::T = 1.0e3     # floor on the mean compressive stress [Pa]
end

function LogarithmicElastic(κ, nu, e0, p_min)
    return LogarithmicElastic(promote(κ, nu, e0, p_min)...)
end

Base.eltype(::LogarithmicElastic{T}) where {T} = T

## A pressure-dependent modulus has no meaning without a stress to evaluate it at, so the
## zero-argument form says so rather than handing back an empty state that would fail later
## and further away.
initial_state(::LogarithmicElastic) = error(
    "LogarithmicElastic has no default state: its stiffness depends on the current " *
        "stress. Use `initial_state(m, σ0)` with the in-situ stress."
)

"""
    initial_state(m::LogarithmicElastic, σ0) -> LogarithmicElasticState

Start the material from a known in-situ stress. A pressure-dependent modulus is undefined
without one, which is why this material cannot use the zero-argument form.
"""
initial_state(::LogarithmicElastic, σ0::Tensors.SymmetricTensor{2}) =
    LogarithmicElasticState(σ0, zero(σ0))

"""
    mean_compressive_stress(m, σ) -> p̄ [Pa]

``\\bar p = -\\mathrm{tr}(\\sigma)/3``, floored at `m.p_min`.
"""
mean_compressive_stress(m::LogarithmicElastic, σ) =
    max(-Tensors.tr(σ) / 3, m.p_min)

"""
    tangent_moduli(m, σ) -> (K, G)

Bulk and shear moduli at the current stress.
"""
function tangent_moduli(m::LogarithmicElastic, σ)
    K = mean_compressive_stress(m, σ) * (1 + m.e0) / m.κ
    G = 3K * (1 - 2m.nu) / (2 * (1 + m.nu))
    return K, G
end

function material_response(
        m::LogarithmicElastic, ε::Tensors.SymmetricTensor{2, dim},
        state::LogarithmicElasticState, Δt
    ) where {dim}
    K, G = tangent_moduli(m, state.σ)
    λ = K - 2G / 3
    C = elastic_stiffness(LinearElastic(λ, G), Val(dim))

    ## Explicit integration over the step: the moduli are evaluated at the stress the
    ## material came in with, so the tangent returned is exact *for this scheme*, which is
    ## what the consistent-tangent test checks.
    Δε = ε - state.ε
    σ = state.σ + C ⊡ Δε
    return σ, C, LogarithmicElasticState(σ, ε)
end

# ── Stress control ────────────────────────────────────────────────────────────

"""
    stress_controlled_response(mat, σ_target, [s,] state, Δt; tol, maxiter) -> (ε, σ, state_new, iters)

The strain that produces a prescribed stress, by Newton on the material response.

Laboratory paths are prescribed in stress, not in strain: an isotropic compression test
holds a cell pressure and measures the volume change. A model whose interface is
``\\varepsilon \\mapsto \\sigma`` cannot follow one directly, so this inverts it, using the
same consistent tangent the global solve uses — which is the cheapest possible check that
the tangent is right, since a wrong one shows up immediately as a failure to converge.

The optional `s` is the second loading variable of a model that has one, the suction of the
Barcelona Basic Model in particular.

`tol` is relative to the target stress, so it means the same thing at 1 kPa and at 1 MPa.
"""
function stress_controlled_response(
        mat::AbstractMaterial, σ_target::Tensors.SymmetricTensor{2}, state, Δt;
        kwargs...
    )
    return _stress_control(ε -> material_response(mat, ε, state, Δt), σ_target, state; kwargs...)
end

function stress_controlled_response(
        mat::AbstractMaterial, σ_target::Tensors.SymmetricTensor{2}, s::Real, state, Δt;
        kwargs...
    )
    return _stress_control(
        ε -> material_response(mat, ε, s, state, Δt), σ_target, state; kwargs...
    )
end

function _stress_control(
        respond, σ_target, state; tol = 1.0e-10, maxiter = 50, maxhalve = 12
    )
    ## The residual is measured squared, and the tolerance squared with it. Taking a norm
    ## would put a `sqrt` on a quantity that goes to zero at convergence, and `sqrt` has an
    ## infinite slope there: the value would be right and any derivative carried through it
    ## would be `NaN`. Nothing here needs the norm itself, only comparisons.
    scale2 = max(σ_target ⊡ σ_target, one(eltype(σ_target)))
    ε = state.ε
    σ, C, st = respond(ε)
    res = (σ - σ_target) ⊡ (σ - σ_target)
    iters = 0
    while res > tol^2 * scale2 && iters < maxiter
        iters += 1
        Δε = Tensors.inv(C) ⊡ (σ - σ_target)
        α = one(res)
        for _ in 1:maxhalve
            σt, Ct, stt = respond(ε - α * Δε)
            rt = (σt - σ_target) ⊡ (σt - σ_target)
            if rt < res
                ε, σ, C, st, res = ε - α * Δε, σt, Ct, stt, rt
                break
            end
            α /= 2
        end
    end
    return ε, σ, st, iters
end
