"""
Finite element backend — Biot assembly on `Ferrite.jl`.

The linear Biot system splits into a stationary part and a storage part,

```math
(K_1 + K_2/\\Delta t)\\, x^{n+1} = f_\\text{ext} + K_2 x^n / \\Delta t
```

so both matrices are assembled once and only the right-hand side moves. `ke1` collects the
elastic stiffness, the Biot coupling and the Darcy conductivity; `ke2` the coupling and the
storage.

Two element families are provided: the Cartesian one, and a radially symmetric one covering
both the cylinder and the sphere.
"""

# ── Cartesian ─────────────────────────────────────────────────────────────────

"""
    biot_element_matrices!(ke1, ke2, m::BiotPoroelastic, cv_u, cv_p)

Element contributions for plane-strain Biot poroelasticity, with `cv_u` a vector-valued
`CellValues` for the displacement and `cv_p` a scalar one for the pressure.
"""
function biot_element_matrices!(ke1, ke2, m::BiotPoroelastic, cv_u, cv_p)
    fill!(ke1, 0.0)
    fill!(ke2, 0.0)

    λ, μ = lame(m)
    K_l = hydraulic_conductivity(m)

    nu_l = Ferrite.getnbasefunctions(cv_u)
    np_l = Ferrite.getnbasefunctions(cv_p)

    for q in 1:Ferrite.getnquadpoints(cv_u)
        dΩ = Ferrite.getdetJdV(cv_u, q)

        ## Elastic stiffness ∫ ε(δu) : C : ε(u) dΩ
        for i in 1:nu_l
            εᵢ = Tensors.symmetric(Ferrite.shape_gradient(cv_u, q, i))
            for j in 1:nu_l
                εⱼ = Tensors.symmetric(Ferrite.shape_gradient(cv_u, q, j))
                σⱼ = λ * Tensors.tr(εⱼ) * one(εⱼ) + 2μ * εⱼ
                ke1[i, j] += (εᵢ ⊡ σⱼ) * dΩ
            end
        end

        ## Biot coupling
        for i in 1:nu_l
            div_δu = Tensors.tr(Ferrite.shape_gradient(cv_u, q, i))
            for j in 1:np_l
                val = m.b * div_δu * Ferrite.shape_value(cv_p, q, j) * dΩ
                ke1[i, nu_l + j] -= val
                ke2[nu_l + j, i] += val
            end
        end

        ## Darcy conductivity and storage
        for i in 1:np_l
            ∇Npi = Ferrite.shape_gradient(cv_p, q, i)
            Npi = Ferrite.shape_value(cv_p, q, i)
            for j in 1:np_l
                ∇Npj = Ferrite.shape_gradient(cv_p, q, j)
                Npj = Ferrite.shape_value(cv_p, q, j)
                ke1[nu_l + i, nu_l + j] += K_l * (∇Npi ⋅ ∇Npj) * dΩ
                ke2[nu_l + i, nu_l + j] += m.N * Npi * Npj * dΩ
            end
        end
    end
    return nothing
end

# ── Radially symmetric ────────────────────────────────────────────────────────

"""
    radial_element_matrices!(ke1, ke2, m::BiotPoroelastic, cv_u, cv_p, coords; nhoop)

Element contributions for a radially symmetric Biot problem on a 1D mesh in `r`, where the
displacement is the scalar ``u_r``.

`nhoop` is the number of hoop directions, and it is the only thing that separates the two
geometries:

| `nhoop` | geometry | strains | volume weight |
|---|---|---|---|
| 1 | long cylinder, plane strain | ``\\varepsilon_{rr} = u'``, ``\\varepsilon_{\\theta\\theta} = u/r``, ``\\varepsilon_{zz} = 0`` | ``r\\,\\mathrm{d}r`` |
| 2 | sphere | ``\\varepsilon_{rr} = u'``, ``\\varepsilon_{\\theta\\theta} = \\varepsilon_{\\varphi\\varphi} = u/r`` | ``r^2\\,\\mathrm{d}r`` |

The discrete strain operator carries `nhoop` copies of ``N_i/r`` alongside ``N_i'``, so
contracting with the isotropic stiffness gives

```math
K^{uu}_{ij} = \\int \\left[ \\lambda\\,\\hat N_i \\hat N_j
  + 2\\mu\\left(N_i'N_j' + n\\,\\frac{N_iN_j}{r^2}\\right)\\right] r^{\\,n}\\,\\mathrm{d}r,
\\qquad \\hat N_i = N_i' + n\\,\\frac{N_i}{r}
```

The hoop terms are what no Cartesian element produces. The ``N_i/r`` factors are why the
quadrature points must stay strictly inside the elements — they do — and the ``r^n`` weight
suppresses what is left near the axis.

This is the kinematics an axisymmetric Barcelona Basic Model will need, written by hand.
"""
function radial_element_matrices!(
        ke1, ke2, m::BiotPoroelastic, cv_u, cv_p, coords; nhoop::Int
    )
    fill!(ke1, 0.0)
    fill!(ke2, 0.0)

    λ, μ = lame(m)
    K_l = hydraulic_conductivity(m)
    n = nhoop

    nu_l = Ferrite.getnbasefunctions(cv_u)
    np_l = Ferrite.getnbasefunctions(cv_p)

    for q in 1:Ferrite.getnquadpoints(cv_u)
        r = Ferrite.spatial_coordinate(cv_u, q, coords)[1]
        dΩ = Ferrite.getdetJdV(cv_u, q) * r^n

        Ngrad = [Ferrite.shape_gradient(cv_u, q, i)[1] for i in 1:nu_l]
        Nval = [Ferrite.shape_value(cv_u, q, i) for i in 1:nu_l]
        Ndiv = [Ngrad[i] + n * Nval[i] / r for i in 1:nu_l]

        for i in 1:nu_l, j in 1:nu_l
            ke1[i, j] += (
                λ * Ndiv[i] * Ndiv[j] +
                    2μ * (Ngrad[i] * Ngrad[j] + n * Nval[i] * Nval[j] / r^2)
            ) * dΩ
        end

        for i in 1:nu_l, j in 1:np_l
            val = m.b * Ndiv[i] * Ferrite.shape_value(cv_p, q, j) * dΩ
            ke1[i, nu_l + j] -= val
            ke2[nu_l + j, i] += val
        end

        for i in 1:np_l, j in 1:np_l
            ke1[nu_l + i, nu_l + j] += K_l *
                Ferrite.shape_gradient(cv_p, q, i)[1] *
                Ferrite.shape_gradient(cv_p, q, j)[1] * dΩ
            ke2[nu_l + i, nu_l + j] += m.N *
                Ferrite.shape_value(cv_p, q, i) *
                Ferrite.shape_value(cv_p, q, j) * dΩ
        end
    end
    return nothing
end

# ── Assembly helpers ──────────────────────────────────────────────────────────

"""
    node_dof_maps(dh, grid, fields...) -> NamedTuple

For each requested field, a vector giving the global dof of that field at each node.

Scalar fields get one dof per node. For a vector field, pass `(:u, component)` to select a
component: Ferrite interleaves the components, so component `c` of local node `loc` sits at
`dof_range(dh, :u)[dim*(loc-1) + c]`.
"""
function node_dof_maps(dh, grid, fields...)
    maps = map(fields) do f
        (f isa Symbol) ? (f => zeros(Int, Ferrite.getnnodes(grid))) :
            (first(f) => zeros(Int, Ferrite.getnnodes(grid)))
    end
    for cell in Ferrite.CellIterator(dh)
        d = Ferrite.celldofs(cell)
        for (spec, (_, target)) in zip(fields, maps)
            if spec isa Symbol
                rng = Ferrite.dof_range(dh, spec)
                for (loc, node) in enumerate(cell.nodes)
                    target[node] = d[rng[loc]]
                end
            else
                name, comp = spec
                rng = Ferrite.dof_range(dh, name)
                dim = length(rng) ÷ length(cell.nodes)
                for (loc, node) in enumerate(cell.nodes)
                    target[node] = d[rng[dim * (loc - 1) + comp]]
                end
            end
        end
    end
    return NamedTuple(maps)
end

"""
    combine!(A, K1, K2, inv_dt)

Fill `A` with `K1 + K2/Δt`, in place and without touching the sparsity pattern.

Writing `K1 + inv_dt .* K2` instead would be wrong in the presence of affine constraints:
sparse addition prunes entries that are numerically zero, and the master–slave couplings are
exactly that until `apply!` condenses them — so the pattern would lose the very slots the
condensation needs. All three matrices must come from the same `allocate_matrix`, and the
arithmetic then goes straight through `nzval`.
"""
function combine!(A, K1, K2, inv_dt)
    @assert A.colptr == K1.colptr == K2.colptr "matrices must share a sparsity pattern"
    @. A.nzval = K1.nzval + inv_dt * K2.nzval
    return A
end
