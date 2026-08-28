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

    K_l = hydraulic_conductivity(m)

    nu_l = Ferrite.getnbasefunctions(cv_u)
    np_l = Ferrite.getnbasefunctions(cv_p)

    ## The tangent comes from the material interface rather than from λ and μ inline, so
    ## that a different skeleton can be substituted without touching the assembly. It is
    ## queried once because linear elasticity has a constant tangent and no state; a
    ## nonlinear material must be asked at every quadrature point, with the strain and the
    ## state of that point.
    mat = skeleton(m)
    ε_probe = Tensors.symmetric(Ferrite.shape_gradient(cv_u, 1, 1))
    _, C, _ = material_response(mat, zero(ε_probe), initial_state(mat), NaN)

    for q in 1:Ferrite.getnquadpoints(cv_u)
        dΩ = Ferrite.getdetJdV(cv_u, q)

        ## Elastic stiffness ∫ ε(δu) : C : ε(u) dΩ
        for i in 1:nu_l
            εᵢ = Tensors.symmetric(Ferrite.shape_gradient(cv_u, q, i))
            for j in 1:nu_l
                εⱼ = Tensors.symmetric(Ferrite.shape_gradient(cv_u, q, j))
                ke1[i, j] += (εᵢ ⊡ (C ⊡ εⱼ)) * dΩ
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

# ── Axisymmetric elastoplasticity ─────────────────────────────────────────────

"""
    axisymmetric_strain(cv, q, ue, coords) -> SymmetricTensor{2,3}

Strain at a quadrature point of an axisymmetric ``(r,z)`` element, as a genuine **3D**
tensor.

The hoop strain ``\\varepsilon_{\\theta\\theta} = u_r/r`` is not a bookkeeping detail: it is
a real component, and a plasticity model that reads the mean stress as
``-\\mathrm{tr}(\\sigma)/3`` gets the wrong answer if it is missing. Working in 3D from the
start is what stops that class of error.

Index order is ``(r, z, \\theta)``, with ``\\varepsilon_{r\\theta}`` and
``\\varepsilon_{z\\theta}`` zero by symmetry.
"""
function axisymmetric_strain(cv, q, ue, coords)
    ∇u = Ferrite.function_gradient(cv, q, ue)
    u = Ferrite.function_value(cv, q, ue)
    r = Ferrite.spatial_coordinate(cv, q, coords)[1]
    εrr, εzz = ∇u[1, 1], ∇u[2, 2]
    εrz = (∇u[1, 2] + ∇u[2, 1]) / 2
    εθθ = u[1] / r
    return Tensors.SymmetricTensor{2, 3}(
        (i, j) -> i == 1 && j == 1 ? εrr :
            i == 2 && j == 2 ? εzz :
            i == 3 && j == 3 ? εθθ :
            (i == 1 && j == 2) || (i == 2 && j == 1) ? εrz : zero(εrr)
    )
end

"""
    axisymmetric_shape_strain(cv, q, i, coords) -> SymmetricTensor{2,3}

The virtual strain of shape function `i`, in the same 3D form.
"""
function axisymmetric_shape_strain(cv, q, i, coords)
    ∇N = Ferrite.shape_gradient(cv, q, i)
    N = Ferrite.shape_value(cv, q, i)
    r = Ferrite.spatial_coordinate(cv, q, coords)[1]
    εrr, εzz = ∇N[1, 1], ∇N[2, 2]
    εrz = (∇N[1, 2] + ∇N[2, 1]) / 2
    εθθ = N[1] / r
    return Tensors.SymmetricTensor{2, 3}(
        (a, b) -> a == 1 && b == 1 ? εrr :
            a == 2 && b == 2 ? εzz :
            a == 3 && b == 3 ? εθθ :
            (a == 1 && b == 2) || (a == 2 && b == 1) ? εrz : zero(εrr)
    )
end

"""
    assemble_axisymmetric!(K, f, dh, cv, mat, states, states_old, u, Δt)

Assemble the tangent `K` and the internal force `f` of an axisymmetric mechanical problem,
asking `mat` for its response at every quadrature point.

`states_old` holds the converged state of the previous step and is read; `states` is written
with the trial state of the current iterate. Keeping them apart is what makes a Newton
iteration repeatable: an iteration that overwrote the history could not be taken twice from
the same starting point, and a line search or a rejected step would corrupt the material.

Returns the assembled pair; the volume element carries the axisymmetric weight ``r``.
"""
function assemble_axisymmetric!(K, f, dh, cv, mat, states, states_old, u, Δt)
    assembler = Ferrite.start_assemble(K, f)
    ## Element arrays take their type from the global ones, not from `Float64`. That is
    ## what lets a `ForwardDiff.Dual` in a material parameter reach the assembled system:
    ## hard-coding the element type here would silently confine differentiability to the
    ## constitutive layer and stop it at the mesh.
    n = Ferrite.ndofs_per_cell(dh)
    T = eltype(K)
    ke = zeros(T, n, n)
    fe = zeros(T, n)

    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cell)
        coords = Ferrite.getcoordinates(cell)
        ue = u[Ferrite.celldofs(cell)]
        cid = Ferrite.cellid(cell)
        fill!(ke, zero(T))
        fill!(fe, zero(T))

        for q in 1:Ferrite.getnquadpoints(cv)
            r = Ferrite.spatial_coordinate(cv, q, coords)[1]
            dΩ = Ferrite.getdetJdV(cv, q) * r

            ε = axisymmetric_strain(cv, q, ue, coords)
            σ, C, st = material_response(mat, ε, states_old[cid][q], Δt)
            states[cid][q] = st

            for i in 1:n
                δε = axisymmetric_shape_strain(cv, q, i, coords)
                fe[i] += (δε ⊡ σ) * dΩ
                for j in 1:n
                    δεj = axisymmetric_shape_strain(cv, q, j, coords)
                    ke[i, j] += (δε ⊡ (C ⊡ δεj)) * dΩ
                end
            end
        end
        Ferrite.assemble!(assembler, Ferrite.celldofs(cell), ke, fe)
    end
    return K, f
end

"""
    newton_solve!(u, K, f, dh, cv, mat, states, states_old, ch, Δt; tol, maxiter, maxhalve, linsolve)

Newton–Raphson on the equilibrium residual, with backtracking, returning the residual norm
of every iteration.

**Why the backtracking is not optional.** An exact tangent buys a quadratic rate *near* the
solution; it says nothing about getting there. The step on which a material first yields is
where undamped Newton fails: the increment is finite, the tangent switches from elastic to
elastoplastic between one iterate and the next, and the full step overshoots. Measured on
the Barcelona Basic Model under isotropic compression, the first plastic step diverges
outright — the residual climbs from 4.8·10⁴ and wanders for twenty-five iterations without
ever descending, while the well-conditioned tangent (``\\mathrm{cond}(K)\\approx 6``) and the
purely elastic steps that precede it converge in a single iteration. The failure is
globalisation, not linearisation.

Halving the step until the residual decreases fixes it, and costs nothing where it is not
needed: away from the transition the full step is accepted at once, so the quadratic rate
survives intact. `maxhalve` bounds the search; exhausting it means the direction is not a
descent direction, which is a modelling problem rather than one a line search can repair.
"""
function newton_solve!(
        u, K, f, dh, cv, mat, states, states_old, ch, Δt;
        tol = 1.0e-8, maxiter = 25, maxhalve = 12, linsolve = \
    )
    norms = Vector{eltype(f)}()
    Ferrite.apply!(u, ch)
    utrial = similar(u)

    ## Squared, like the stress-control loop: a norm puts a `sqrt` on a quantity that goes
    ## to zero at convergence, and nothing here needs anything but comparisons. The values
    ## reported are still norms.
    residual_sq!(uu) = begin
        assemble_axisymmetric!(K, f, dh, cv, mat, states, states_old, uu, Δt)
        Ferrite.apply_zero!(K, f, ch)
        f ⋅ f
    end

    nrm = residual_sq!(u)
    for _ in 1:maxiter
        push!(norms, sqrt(nrm))
        nrm < tol^2 && break
        Δu = linsolve(K, f)
        α = 1.0
        for _ in 1:maxhalve
            @. utrial = u - α * Δu
            trial = residual_sq!(utrial)
            if trial < nrm
                nrm = trial
                break
            end
            α /= 2
        end
        copyto!(u, utrial)
    end
    ## The states left behind must be those of the accepted iterate, not of the last trial
    ## the line search happened to reject.
    residual_sq!(u)
    return norms
end
