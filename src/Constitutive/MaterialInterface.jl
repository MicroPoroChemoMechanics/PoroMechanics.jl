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

Type-parameterised like every other law here, so a response can be differentiated with
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
