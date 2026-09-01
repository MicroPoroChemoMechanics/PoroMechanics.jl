"""
Computational homogenisation: a material whose response is a finite element solve on a
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

A unit cell with one material per cell region, to be homogenised under periodic boundary
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
    homogenize_stress(cell::PeriodicCell, ε_macro; states = nothing) -> (σ_macro, u)

Solve the cell under the imposed mean strain and return the volume-averaged stress, with
the fluctuation field that produced it.

`ε_macro` is a 2×2 plane strain tensor. Each cell region uses its own material through
[`material_response`](@ref), so the same routine covers an elastic cell and a plastic one —
the difference lives entirely in the materials handed to [`periodic_cell`](@ref).
"""
function homogenize_stress(cell::PeriodicCell, ε_macro::Tensors.SymmetricTensor{2, 2}; states = nothing)
    dh, cv, ch = cell.dh, cell.cv, cell.ch
    n_basefuncs = Ferrite.getnbasefunctions(cv)
    ndofs = Ferrite.ndofs(dh)

    ## A sparse matrix allocated *through* the constraint handler: that is what lets
    ## `apply!` condense the periodic affine constraints, and a dense matrix cannot.
    K = Ferrite.allocate_matrix(dh, ch)
    f = zeros(ndofs)
    assembler = Ferrite.start_assemble(K, f)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ε_M = plane_strain(ε_macro)

    for cc in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cc)
        fill!(ke, 0); fill!(fe, 0)
        mat = cell.materials[cell.region[Ferrite.cellid(cc)]]
        state = states === nothing ? NoState() : states[Ferrite.cellid(cc)]

        for q in 1:Ferrite.getnquadpoints(cv)
            dΩ = Ferrite.getdetJdV(cv, q)
            ## Tangent and the stress the macroscopic strain alone would produce.
            σ0, C, _ = material_response(mat, ε_M, state, 1.0)
            for i in 1:n_basefuncs
                δε = plane_strain(Ferrite.shape_symmetric_gradient(cv, q, i))
                ## Residual of the imposed part: the fluctuation must cancel it.
                fe[i] -= (δε ⊡ σ0) * dΩ
                for j in 1:n_basefuncs
                    δεj = plane_strain(Ferrite.shape_symmetric_gradient(cv, q, j))
                    ke[i, j] += (δε ⊡ C ⊡ δεj) * dΩ
                end
            end
        end
        Ferrite.assemble!(assembler, Ferrite.celldofs(cc), ke, fe)
    end

    Ferrite.apply!(K, f, ch)
    u = K \ f
    Ferrite.apply!(u, ch)

    ## Macroscopic stress: the volume average of the microscopic one, on the *total* strain.
    σ_sum = zero(Tensors.SymmetricTensor{2, 3, Float64})
    for cc in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv, cc)
        ue = u[Ferrite.celldofs(cc)]
        mat = cell.materials[cell.region[Ferrite.cellid(cc)]]
        state = states === nothing ? NoState() : states[Ferrite.cellid(cc)]
        for q in 1:Ferrite.getnquadpoints(cv)
            dΩ = Ferrite.getdetJdV(cv, q)
            ε_fluct = plane_strain(Ferrite.function_symmetric_gradient(cv, q, ue))
            σ, _, _ = material_response(mat, ε_M + ε_fluct, state, 1.0)
            σ_sum += σ * dΩ
        end
    end
    return σ_sum / cell.volume, u
end

"""
    homogenized_stiffness(cell::PeriodicCell) -> SymmetricTensor{4,3}

Effective stiffness, obtained by solving the cell once per independent macroscopic strain.

Three solves in plane strain — ``\\varepsilon_{11}``, ``\\varepsilon_{22}``,
``\\varepsilon_{12}`` — and the columns of the tangent read off the resulting stresses.
Valid while the cell is linear; a plastic cell has no such tensor and must be probed along
the path it actually follows.
"""
function homogenized_stiffness(cell::PeriodicCell)
    δ = 1.0e-6
    probes = (
        Tensors.SymmetricTensor{2, 2}((δ, 0.0, 0.0)),
        Tensors.SymmetricTensor{2, 2}((0.0, 0.0, δ)),
        Tensors.SymmetricTensor{2, 2}((0.0, δ / 2, 0.0)),
    )
    responses = [first(homogenize_stress(cell, ε)) / δ for ε in probes]
    return responses
end
