"""
    NernstPlanck(; phi, D, z, tortuosity=nothing, q_background=0, dirichlet=())

Multi-ionic transport with a zero-current closure.

| field | meaning |
|---|---|
| `phi` | porosity, scalar |
| `D` | one diffusivity per ion; effective in bulk unless multiplied by `tortuosity` |
| `z` | charge numbers, one per ion, in a tuple |
| `tortuosity` | an `AbstractTortuosity`; `τ` here is `D_eff/D⁰`, **not** a geometric factor |
| `dirichlet` | one boundary tuple per unknown, ions first and the potential last |
| `q_background` | prescribed `Σ zᵢ cᵢ` [mol/m³ of solution], opposite to the untransported charge |

The unknowns are the `n` ion concentrations followed by the potential `Ψ`, so
`nspecies == length(D) + 1`.

Coefficients retain their numeric types, so a `ForwardDiff.Dual` can enter a parameter.
Use `fvm_system(model, grid; reaction = true)` to assemble electroneutrality.
Initial and boundary concentrations must satisfy that constraint. Set a Dirichlet
reference for the potential to remove its constant nullspace.

!!! note "The tortuosity convention"
    `OhJang` returns `D_eff/D⁰`, porosity included. The flux carries **no** extra `φ`, while
    the storage does. Mixing that up with `FickModel`, whose `D` is a pore diffusivity,
    costs a factor `1/φ`.
"""
Base.@kwdef struct NernstPlanck{T, DD, ZZ, TT, Q, B} <: AbstractPoroModel
    phi::T
    D::DD
    z::ZZ
    tortuosity::TT = nothing
    q_background::Q = 0.0
    dirichlet::B = ()

    function NernstPlanck(phi::T, D::DD, z::ZZ, tortuosity::TT, q_background::Q, dirichlet::B) where {T, DD, ZZ, TT, Q, B}
        !isempty(D) && length(D) == length(z) ||
            throw(DimensionMismatch("provide one diffusivity and charge per ion"))
        all(d -> isfinite(d) && d >= 0, D) ||
            throw(ArgumentError("diffusivities must be finite and nonnegative"))
        isfinite(phi) && 0 < phi <= 1 || throw(ArgumentError("porosity must lie in (0, 1]"))
        all(isfinite, z) && isfinite(q_background) || throw(ArgumentError("charges must be finite"))
        isempty(dirichlet) || length(dirichlet) == length(D) + 1 ||
            throw(DimensionMismatch("provide one boundary tuple per ion and potential"))
        return new{T, DD, ZZ, TT, Q, B}(phi, D, z, tortuosity, q_background, dirichlet)
    end
end

"""Number of transported ions, excluding the electric potential."""
nions(m::NernstPlanck) = length(m.D)
nspecies(m::NernstPlanck) = nions(m) + 1
"""Index of the dimensionless electric potential in the unknown vector."""
ipot(m::NernstPlanck) = nions(m) + 1

species_names(m::NernstPlanck) =
    vcat([Symbol("c_", i) for i in 1:nions(m)], [:Ψ])

"""
    effective_diffusivity(m, i)

`Dᵢ τ(φ)`, the coefficient the flux actually carries.
"""
effective_diffusivity(m::NernstPlanck, i) =
    m.tortuosity === nothing ? m.D[i] : m.D[i] * tortuosity(m.tortuosity, m.phi, 1)

# ── The interface ─────────────────────────────────────────────────────────────

"""
    storage!(f, u, node, m::NernstPlanck, data)

`φ cᵢ` for each ion, and **zero** for the potential, which makes its row algebraic: the
constraint in `reaction!` holds at every instant rather than accumulating.
"""
function storage!(f, u, node, m::NernstPlanck, ::Any)
    for i in 1:nions(m)
        f[i] = m.phi * u[i]
    end
    f[ipot(m)] = zero(eltype(f))
    return nothing
end

"""
    reaction!(f, u, node, m::NernstPlanck, data)

The electroneutrality constraint `Σᵢ zᵢ cᵢ = q_background`, which is what determines `Ψ`.
Zero for the ions: they have no volumetric source here.
"""
function reaction!(f, u, node, m::NernstPlanck, ::Any)
    for i in 1:nions(m)
        f[i] = zero(eltype(f))
    end
    f[ipot(m)] = sum(m.z[i] * u[i] for i in 1:nions(m)) - m.q_background
    return nothing
end

"""
    flux!(f, u, edge, m::NernstPlanck, data)

Scharfetter-Gummel for each ion. The potential carries no flux of its own.

`VoronoiFVM.fbernoulli_pm(x)` returns `(B(x), B(-x))` with `B(x) = x/(eˣ−1)`, and the fitted flux of
`−D(∇c + z c ∇Ψ)` is `D (B(−δ) c₁ − B(δ) c₂)` with `δ = z(Ψ₁ − Ψ₂)`. At `δ = 0` both
Bernoulli factors are one and this is the plain difference.
"""
function flux!(f, u, edge, m::NernstPlanck, ::Any)
    iψ = ipot(m)
    δψ = u[iψ, 1] - u[iψ, 2]
    for i in 1:nions(m)
        bp, bm = VoronoiFVM.fbernoulli_pm(m.z[i] * δψ)          # (B(δ), B(−δ))
        f[i] = effective_diffusivity(m, i) * (bm * u[i, 1] - bp * u[i, 2])
    end
    ## No flux of its own: the potential is fixed by the constraint in `reaction!`.
    f[iψ] = zero(eltype(f))
    return nothing
end

"""
    bcondition!(f, u, bnode, m::NernstPlanck, data)

One boundary tuple per unknown. The potential needs at least one Dirichlet value: only its
gradient enters the fluxes, so without a reference the system is singular by a constant.
"""
function bcondition!(f, u, bnode, m::NernstPlanck, ::Any)
    return _transport_dirichlet!(f, u, bnode, m.dirichlet, 1)
end

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""
    net_charge(m, u) -> Vector

`Σᵢ zᵢ cᵢ` at every node, to be compared with `q_background`. The algebraic constraint
preserves this value when initial and boundary data are consistent.
"""
net_charge(m::NernstPlanck, u) =
    [sum(m.z[i] * u[i, k] for i in 1:nions(m)) for k in axes(u, 2)]

"""
    edge_current(m, u, dx) -> (current, scale)

`Σᵢ zᵢ Jᵢ` on each edge of a uniformly spaced one-dimensional grid, and the sum of
`|zᵢ Jᵢ|` to scale it by. `dx` is the node spacing. At zero-current conditions the
first vector should vanish to solver accuracy.
"""
function edge_current(m::NernstPlanck, u, dx)
    n = size(u, 2)
    T = promote_type(eltype(u), typeof(dx), (typeof(effective_diffusivity(m, i) * m.z[i]) for i in 1:nions(m))...)
    cur = zeros(T, n - 1)
    scale = zeros(T, n - 1)
    for k in 1:(n - 1)
        δψ = u[ipot(m), k] - u[ipot(m), k + 1]
        for i in 1:nions(m)
            bp, bm = VoronoiFVM.fbernoulli_pm(m.z[i] * δψ)
            J = effective_diffusivity(m, i) * (bm * u[i, k] - bp * u[i, k + 1]) / dx
            cur[k] += m.z[i] * J
            scale[k] += abs(m.z[i] * J)
        end
    end
    return cur, scale
end
