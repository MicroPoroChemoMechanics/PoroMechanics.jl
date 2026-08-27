# Shared by the poroelasticity benchmarks: a homogeneous Biot material and its element
# matrices. Terzaghi and Mandel differ in geometry, boundary conditions and reference
# solution — not in constitutive behaviour — so the assembly is written once here.
#
# This file is the natural candidate to move into `src/Constitutive/Poroelasticity.jl` at
# milestone J3, when poroelasticity becomes a first-class part of the library.

using PoroMechanics
using Ferrite
using LinearAlgebra
using SparseArrays
using Printf

"""
    HomogeneousBiot(; E, nu, k, mu_l, b, N)

Linear isotropic Biot poroelasticity, one material.

| Field | Meaning |
|---|---|
| `E`, `nu` | drained Young's modulus [Pa] and Poisson's ratio [-] |
| `k` | intrinsic permeability [m²] |
| `mu_l` | dynamic viscosity of the fluid [Pa·s] |
| `b` | Biot coefficient [-] |
| `N` | storage modulus at constant strain [Pa⁻¹], the inverse of the Biot modulus |
"""
Base.@kwdef struct HomogeneousBiot <: AbstractPoroModel
    E::Float64 = 1.0e8
    nu::Float64 = 0.2
    k::Float64 = 1.0e-13
    mu_l::Float64 = 1.0e-3
    b::Float64 = 1.0
    N::Float64 = 7.2e-9
end

PoroMechanics.nspecies(::HomogeneousBiot) = 3
PoroMechanics.species_names(::HomogeneousBiot) = [:u1, :u2, :p]

"""Lamé coefficients `(λ, μ)` [Pa]."""
lame(m::HomogeneousBiot) = (
    m.E * m.nu / ((1 + m.nu) * (1 - 2m.nu)),
    m.E / (2 * (1 + m.nu)),
)

"""Drained bulk modulus `K = λ + 2μ/3` [Pa]."""
function bulk_modulus(m::HomogeneousBiot)
    λ, μ = lame(m)
    return λ + 2μ / 3
end

"""Oedometric modulus `M_o = λ + 2μ = K + 4μ/3` [Pa]."""
function oedometric_modulus(m::HomogeneousBiot)
    λ, μ = lame(m)
    return λ + 2μ
end

"""
    consolidation_coefficient(m) -> c [m²/s]

Uniaxial diffusivity ``c = (k/\\mu_l) / (N + b^2/M_o)``, obtained by eliminating the strain
between the constitutive law and the storage equation under uniaxial conditions.
"""
consolidation_coefficient(m::HomogeneousBiot) =
    (m.k / m.mu_l) / (m.N + m.b^2 / oedometric_modulus(m))

"""Skempton coefficient ``B = b / (N K + b^2)`` [-]."""
skempton(m::HomogeneousBiot) = m.b / (m.N * bulk_modulus(m) + m.b^2)

"""
    undrained_poisson(m) -> ν_u

``\\nu_u = \\dfrac{3\\nu + bB(1-2\\nu)}{3 - bB(1-2\\nu)}``, which tends to 1/2 as the fluid
and grains become incompressible (``B \\to 1``).
"""
function undrained_poisson(m::HomogeneousBiot)
    q = m.b * skempton(m) * (1 - 2m.nu)
    return (3m.nu + q) / (3 - q)
end

"""
    element_matrices!(ke1, ke2, ::Nothing, m::HomogeneousBiot, cv_u, cv_p)

`ke1` holds the elastic stiffness, the Biot coupling and the Darcy conductivity; `ke2` the
coupling and the storage. Each step solves
``(K_1 + K_2/\\Delta t)\\, x^{n+1} = f_\\text{ext} + K_2 x^n / \\Delta t``.
"""
function PoroMechanics.element_matrices!(ke1, ke2, ::Nothing, m::HomogeneousBiot, cv_u, cv_p)
    fill!(ke1, 0.0)
    fill!(ke2, 0.0)

    λ, μ = lame(m)
    K_l = m.k / m.mu_l          # hydraulic conductivity [m²/(Pa·s)]

    nu_l = getnbasefunctions(cv_u)
    np_l = getnbasefunctions(cv_p)

    for q in 1:getnquadpoints(cv_u)
        dΩ = getdetJdV(cv_u, q)

        ## Elastic stiffness ∫ ε(δu) : C : ε(u) dΩ
        for i in 1:nu_l
            εᵢ = symmetric(shape_gradient(cv_u, q, i))
            for j in 1:nu_l
                εⱼ = symmetric(shape_gradient(cv_u, q, j))
                σⱼ = λ * tr(εⱼ) * one(εⱼ) + 2μ * εⱼ
                ke1[i, j] += (εᵢ ⊡ σⱼ) * dΩ
            end
        end

        ## Biot coupling
        for i in 1:nu_l
            div_δu = tr(shape_gradient(cv_u, q, i))
            for j in 1:np_l
                Np = shape_value(cv_p, q, j)
                val = m.b * div_δu * Np * dΩ
                ke1[i, nu_l + j] -= val
                ke2[nu_l + j, i] += val
            end
        end

        ## Darcy conductivity and storage
        for i in 1:np_l
            ∇Npi = shape_gradient(cv_p, q, i)
            Npi = shape_value(cv_p, q, i)
            for j in 1:np_l
                ∇Npj = shape_gradient(cv_p, q, j)
                Npj = shape_value(cv_p, q, j)
                ke1[nu_l + i, nu_l + j] += K_l * (∇Npi ⋅ ∇Npj) * dΩ
                ke2[nu_l + i, nu_l + j] += m.N * Npi * Npj * dΩ
            end
        end
    end
    return nothing
end

# ── Helpers shared by the benchmark drivers ───────────────────────────────────

"""
    node_dof_maps(dh, grid) -> (uy_dof, p_dof)

For each node, the global dof of the vertical displacement and of the pressure. Ferrite
interleaves the components of a vector field, so component 2 of local node `loc` sits at
`dof_range(dh, :u)[2loc]`.
"""
function node_dof_maps(dh, grid)
    uy_dof = zeros(Int, getnnodes(grid))
    p_dof = zeros(Int, getnnodes(grid))
    u_range = dof_range(dh, :u)
    p_range = dof_range(dh, :p)
    for cell in CellIterator(dh)
        d = celldofs(cell)
        for (loc, node) in enumerate(cell.nodes)
            uy_dof[node] = d[u_range[2loc]]
            p_dof[node] = d[p_range[loc]]
        end
    end
    return uy_dof, p_dof
end

"""
    uniform_schedule(T_probe; T_start, dT) -> Vector

Dimensionless times to march through: one short step to `T_start`, which captures the
undrained response, then a uniform grid of spacing `dT` merged with the probe times so that
each probe is landed on exactly.
"""
function uniform_schedule(T_probe; T_start, dT)
    T_max = maximum(T_probe)
    grid = T_start .+ dT .* (1:ceil(Int, (T_max - T_start) / dT))
    schedule = sort(unique(vcat(T_start, grid, T_probe)))
    return filter(<=(T_max + 1.0e-12), schedule)
end

"""
    combine!(A, K1, K2, inv_dt)

Fill `A` with `K1 + K2/Δt`, in place and without touching the sparsity pattern.

Writing `K1 + inv_dt .* K2` instead would be wrong in the presence of affine constraints:
sparse addition prunes entries that are numerically zero, and the master–slave couplings
are exactly that until `apply!` condenses them — so the pattern would lose the very slots
the condensation needs. All three matrices come from the same `allocate_matrix(dh, ch)`,
hence share a pattern, and the arithmetic can go straight through `nzval`.
"""
function combine!(A, K1, K2, inv_dt)
    @assert A.colptr == K1.colptr == K2.colptr "matrices must share a sparsity pattern"
    @. A.nzval = K1.nzval + inv_dt * K2.nzval
    return A
end
