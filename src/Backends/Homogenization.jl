"""
Computational homogenization: a material whose response is a finite element solve on a
periodic unit cell.

This is the brick the two-scale coupling is missing. Rather than a separate FE² driver, the
cell is written as an [`AbstractMaterial`](@ref): given a macroscopic strain it returns a
macroscopic stress, exactly like [`BBM`](@ref) or [`DruckerPrager`](@ref) do. A two-scale
computation is then an ordinary mechanical problem whose material happens to run a nested
solve — no new backend, no new interface.

The kinematics are the classical periodic decomposition,

```math
\\mathbf{u}(\\mathbf{x}) = \\boldsymbol{\\varepsilon}^M \\cdot \\mathbf{x} + \\tilde{\\mathbf{u}}(\\mathbf{x}),
\\qquad \\tilde{\\mathbf{u}} \\text{ periodic}
```

so the fluctuation ``\\tilde{\\mathbf{u}}`` solves

```math
\\int_\\Omega \\delta\\tilde{\\boldsymbol{\\varepsilon}} : \\mathbb{C} :
\\big(\\boldsymbol{\\varepsilon}^M + \\tilde{\\boldsymbol{\\varepsilon}}\\big)\\,\\mathrm{d}\\Omega = 0
```

and the macroscopic stress is the volume average of the microscopic one. Periodicity is
what makes the answer an intrinsic property of the cell rather than of its boundary: uniform
displacement or uniform traction on the edges bracket the periodic answer instead of
reproducing it, which is the classical result and the reason for the extra machinery.
"""

"""
    PeriodicCell(grid, materials; ip_order = 1, qr_order = 2)

A unit cell with one material per cell region, to be homogenized under periodic boundary
conditions.

`materials` maps a cell region index to an `AbstractMaterial`. Regions are the grid's own
`cellsets`-equivalent — with a mesh read from Gmsh, the compacted elementary tags.

Plane strain in two dimensions: the out-of-plane strain is zero, which is what a cell
extruded along a third axis does and what the codes this is checked against assume.
"""
struct PeriodicCell{G, D, C, H, M} <: AbstractMaterial
    grid::G
    dh::D
    cv::C
    ch::H
    materials::M
    region::Vector{Int}
    volume::Float64
end

"""
    periodic_cell(nodes, cells, regions, materials; tol = 1e-8) -> PeriodicCell

Build a cell from raw connectivity: `nodes` is `dim × n`, `cells` is `nodes_per_cell × m`,
`regions` one index per cell.

The periodic facet pairs are collected **geometrically**, from the coordinates, rather than
from named boundary sets. A mesh written by another code carries its own naming convention
— Bil's `composite0.msh` numbers its edges 13, 14, 104, 105, … with no left/right meaning —
and matching on position is the one rule that survives crossing that boundary.
"""
function periodic_cell(nodes, cells, regions, materials; tol = 1.0e-8)
    dim = size(nodes, 1)
    dim == 2 || error("periodic_cell: only two dimensions for now, got $dim")

    ferrite_nodes = [Ferrite.Node(Ferrite.Vec{2}((nodes[1, i], nodes[2, i]))) for i in axes(nodes, 2)]
    ferrite_cells = [Ferrite.Triangle((cells[1, e], cells[2, e], cells[3, e])) for e in axes(cells, 2)]
    grid = Ferrite.Grid(ferrite_cells, ferrite_nodes)

    ## Facet sets by position, so the mesh's own edge numbering is irrelevant.
    x_lo, x_hi = extrema(nodes[1, :])
    y_lo, y_hi = extrema(nodes[2, :])
    on(f) = x -> f(x)
    Ferrite.addfacetset!(grid, "left", x -> abs(x[1] - x_lo) < tol)
    Ferrite.addfacetset!(grid, "right", x -> abs(x[1] - x_hi) < tol)
    Ferrite.addfacetset!(grid, "bottom", x -> abs(x[2] - y_lo) < tol)
    Ferrite.addfacetset!(grid, "top", x -> abs(x[2] - y_hi) < tol)

    ip = Ferrite.Lagrange{Ferrite.RefTriangle, 1}()^2
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)

    qr = Ferrite.QuadratureRule{Ferrite.RefTriangle}(2)
    cv = Ferrite.CellValues(qr, ip, Ferrite.Lagrange{Ferrite.RefTriangle, 1}())

    ## Periodicity on the fluctuation, plus one pinned node.
    ##
    ## The two directions go into a **single** `PeriodicDirichlet`. Adding one constraint
    ## per direction looks equivalent and is not: a corner node belongs to both mappings,
    ## and Ferrite refuses the resulting nested affine constraint.
    ##
    ## The pinned node has to be an **interior** one for the same reason — every boundary
    ## node already carries a periodic constraint. It only removes the rigid translation the
    ## fluctuation is defined up to; without it the system is singular.
    ch = Ferrite.ConstraintHandler(dh)
    pairs = Ferrite.collect_periodic_facets(
        grid, "left", "right", x -> x - Ferrite.Vec{2}((x_hi - x_lo, 0.0))
    )
    Ferrite.collect_periodic_facets!(
        pairs, grid, "bottom", "top", x -> x - Ferrite.Vec{2}((0.0, y_hi - y_lo))
    )
    Ferrite.add!(ch, Ferrite.PeriodicDirichlet(:u, pairs))

    interior = findfirst(axes(nodes, 2)) do i
        abs(nodes[1, i] - x_lo) > tol && abs(nodes[1, i] - x_hi) > tol &&
            abs(nodes[2, i] - y_lo) > tol && abs(nodes[2, i] - y_hi) > tol
    end
    interior === nothing && error(
        "periodic_cell: every node lies on the boundary, so there is none left to pin"
    )
    Ferrite.addnodeset!(grid, "pinned", Set([interior]))
    Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getnodeset(grid, "pinned"), (x, t) -> [0.0, 0.0], [1, 2]))
    Ferrite.close!(ch)

    volume = (x_hi - x_lo) * (y_hi - y_lo)
    return PeriodicCell(grid, dh, cv, ch, materials, Vector{Int}(regions), volume)
end

"""
    plane_strain(ε2::SymmetricTensor{2,2}) -> SymmetricTensor{2,3}

Embed a plane strain tensor in three dimensions, with ``\\varepsilon_{33} = 0``.
"""
function plane_strain(ε2::Tensors.SymmetricTensor{2, 2, T}) where {T}
    return Tensors.SymmetricTensor{2, 3}(
        (i, j) -> (i <= 2 && j <= 2) ? ε2[i, j] : zero(T)
    )
end

"""
    cell_states(cell) -> Matrix

Fresh material states, one per quadrature point of every element: `states[q, e]`.

A path-dependent cell needs them; an elastic one is handed `NoState()` and never looks.
"""
function cell_states(cell::PeriodicCell)
    nq = Ferrite.getnquadpoints(cell.cv)
    ne = Ferrite.getncells(cell.grid)
    return [
        initial_state(cell.materials[cell.region[e]])
            for q in 1:nq, e in 1:ne
    ]
end

initial_state(m::LinearElastic) = NoState()

"""
    homogenize_stress(cell, ε_macro, states_n = nothing; maxiter = 30, rtol = 1e-10)
        -> (σ_macro, u, states)

Solve the cell under the imposed mean strain and return the volume-averaged stress, the
fluctuation field, and the quadrature states it leaves behind.

Newton on the fluctuation, so a path-dependent phase is handled by the same routine as a
linear one — an elastic cell simply converges on the first correction. `states_n` are the
states the cell converged at previously; `nothing` starts from fresh ones, which is only
right for a first step or a rate-independent elastic cell.

Convergence is measured on the residual **relative to the force the macroscopic strain
alone would generate**, not absolutely. The residual of a cell is an integrated stress and
its magnitude follows the loading; an absolute tolerance would be a statement about the
units of the moduli.
"""
function homogenize_stress(
        cell::PeriodicCell, ε_macro::Tensors.SymmetricTensor{2, 2}, states_n = nothing;
        maxiter = 30, rtol = 1.0e-10
    )
    dh, cv, ch = cell.dh, cell.cv, cell.ch
    n_basefuncs = Ferrite.getnbasefunctions(cv)
    ε_M = plane_strain(ε_macro)

    states_prev = states_n === nothing ? cell_states(cell) : states_n
    states = copy(states_prev)

    u = zeros(Ferrite.ndofs(dh))
    ke = zeros(n_basefuncs, n_basefuncs)
    re = zeros(n_basefuncs)
    local scale, residual

    for iter in 1:maxiter
        K = Ferrite.allocate_matrix(dh, ch)
        r = zeros(Ferrite.ndofs(dh))
        assembler = Ferrite.start_assemble(K, r)

        for cc in Ferrite.CellIterator(dh)
            Ferrite.reinit!(cv, cc)
            fill!(ke, 0); fill!(re, 0)
            e = Ferrite.cellid(cc)
            mat = cell.materials[cell.region[e]]
            ue = u[Ferrite.celldofs(cc)]

            for q in 1:Ferrite.getnquadpoints(cv)
                dΩ = Ferrite.getdetJdV(cv, q)
                ε_f = plane_strain(Ferrite.function_symmetric_gradient(cv, q, ue))
                σ, C, st = material_response(mat, ε_M + ε_f, states_prev[q, e], 1.0)
                states[q, e] = st
                for i in 1:n_basefuncs
                    δε = plane_strain(Ferrite.shape_symmetric_gradient(cv, q, i))
                    re[i] += (δε ⊡ σ) * dΩ
                    for j in 1:n_basefuncs
                        δεj = plane_strain(Ferrite.shape_symmetric_gradient(cv, q, j))
                        ke[i, j] += (δε ⊡ C ⊡ δεj) * dΩ
                    end
                end
            end
            Ferrite.assemble!(assembler, Ferrite.celldofs(cc), ke, re)
        end

        ## The scale is the residual *before* condensation, at the first iteration: the
        ## force the imposed macroscopic strain generates. Measuring against the condensed
        ## residual instead would divide by whatever survives the constraints, which for a
        ## cell already in equilibrium is roundoff — and no tolerance is then reachable.
        iter == 1 && (scale = max(norm(r), eps()))

        ## Condense before measuring. On the slave degrees of freedom of a periodic
        ## constraint the raw residual carries a reaction, not an error: it does not vanish
        ## at the solution, and a norm taken over it never converges.
        Ferrite.apply_zero!(K, r, ch)

        residual = norm(r)
        residual <= rtol * scale && break

        ## `apply_zero!` on the correction too — that is what distributes the periodic
        ## coupling onto the increment. `apply!(u, ch)` would set prescribed values, which
        ## is right for a solution and wrong for a Newton correction.
        Δu = K \ r
        Ferrite.apply_zero!(Δu, ch)
        u -= Δu

        iter == maxiter && error(
            "homogenize_stress: the cell did not converge — ‖R‖/‖R₀‖ = $(residual / scale)"
        )
    end

    ## Macroscopic stress: the volume average of the microscopic one.
    σ_sum = zero(Tensors.SymmetricTensor{2, 3, Float64})
    for cc in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cc)
        e = Ferrite.cellid(cc)
        ue = u[Ferrite.celldofs(cc)]
        for q in 1:Ferrite.getnquadpoints(cv)
            σ_sum += states[q, e] isa NoState ?
                begin
                    ε_f = plane_strain(Ferrite.function_symmetric_gradient(cv, q, ue))
                    first(material_response(cell.materials[cell.region[e]], ε_M + ε_f, NoState(), 1.0))
                end * Ferrite.getdetJdV(cv, q) :
                states[q, e].σ * Ferrite.getdetJdV(cv, q)
        end
    end
    return σ_sum / cell.volume, u, states
end

## In-plane components of a symmetric tensor, and back. Deliberately *not* Voigt: the
## macroscopic Newton below perturbs a tensor component and reads a tensor component, so
## the engineering factor of two on the shear term would have to be undone on both sides.
_inplane(σ) = (σ[1, 1], σ[1, 2], σ[2, 2])
_intensor(v) = Tensors.SymmetricTensor{2, 2}((v[1], v[2], v[3]))

"""
    homogenize_to_stress(cell, σ_target, states_n; ε_guess, maxiter = 25, rtol = 1e-8)
        -> (; ε, σ, states, iterations)

Find the macroscopic strain that puts the cell under a prescribed **in-plane stress**, in
plane strain — the control a single-element two-scale problem reduces to.

The three in-plane components are stress-controlled and ``\\varepsilon_{33} = 0``, so
``\\sigma_{33}`` comes out as a result. Newton on the three unknowns, with the homogenized
tangent obtained by perturbing the cell — three extra cell solves per iteration. That is
not a shortcut: once a phase yields, the tangent of the cell is not the volume average of
anything, and Bil homogenizes its own by finite differences for the same reason.

`states_n` are the states at the last converged macroscopic step; every trial re-solves the
cell from *those*, never from the previous trial, so the path stays single-valued.
"""
function homogenize_to_stress(
        cell::PeriodicCell, σ_target::Tensors.SymmetricTensor{2, 2}, states_n = nothing;
        ε_guess = zero(Tensors.SymmetricTensor{2, 2}), maxiter = 25, rtol = 1.0e-8,
        perturbation = 1.0e-8
    )
    states = states_n === nothing ? cell_states(cell) : states_n
    target = _inplane(σ_target)
    scale = max(maximum(abs, target), one(eltype(target)))
    ε = ε_guess
    σ, st = zero(Tensors.SymmetricTensor{2, 3}), states

    for iter in 1:maxiter
        σ, _, st = homogenize_stress(cell, ε, states)
        R = _inplane(σ) .- target
        maximum(abs, R) <= rtol * scale && return (; ε, σ, states = st, iterations = iter)

        C = homogenized_tangent(cell, ε, states; perturbation)
        ε -= _intensor(C \ collect(R))
    end
    error("homogenize_to_stress: no macroscopic equilibrium after $maxiter iterations")
end

"""
    homogenized_tangent(cell, ε_macro, states = nothing; perturbation = 1e-8) -> Matrix

The 3×3 in-plane macroscopic tangent at a given macroscopic strain, by forward differences
on the cell — one column per component of `(ε₁₁, ε₁₂, ε₂₂)`, and the same convention on the
rows, so no engineering factor of two enters anywhere.

Finite differences and not an assembled quantity, deliberately. While the cell is elastic
the tangent could be condensed out of the microscopic stiffness, but the moment a phase
yields there is nothing to condense: the macroscopic tangent depends on which quadrature
points are on their yield surface, and only a perturbation sees that. Bil homogenizes its
own the same way.
"""
function homogenized_tangent(
        cell::PeriodicCell, ε_macro::Tensors.SymmetricTensor{2, 2}, states = nothing;
        perturbation = 1.0e-8
    )
    st = states === nothing ? cell_states(cell) : states
    σ0, _, _ = homogenize_stress(cell, ε_macro, st)
    C = zeros(3, 3)
    for k in 1:3
        dv = ntuple(i -> i == k ? perturbation : 0.0, 3)
        σk, _, _ = homogenize_stress(cell, ε_macro + _intensor(dv), st)
        C[:, k] .= (_inplane(σk) .- _inplane(σ0)) ./ perturbation
    end
    return C
end

"""
    homogenized_stiffness(cell::PeriodicCell) -> Matrix

The effective in-plane stiffness of a linear cell: [`homogenized_tangent`](@ref) evaluated
at zero strain, in the `(11, 12, 22)` convention.

Valid only while every phase is linear. A cell with a plastic phase has no such matrix —
its tangent depends on where it is on its path, and it has to be asked for there.
"""
homogenized_stiffness(cell::PeriodicCell) = homogenized_tangent(cell, zero(Tensors.SymmetricTensor{2, 2}))
