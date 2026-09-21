"""
    assemble_biot_matrices(dh, cv_u, cv_p, material; constraints=nothing, valuetype=...)
        -> (K1, K2)

Assemble the stationary and storage matrices of linear Cartesian Biot poroelasticity.
`material` is a [`BiotPoroelastic`](@ref), or a function `cell -> material` for a
heterogeneous medium. The closed Ferrite `DofHandler` must contain exactly `:u`
followed by `:p`, matching the supplied displacement and pressure `CellValues`.
Use one cell type and interpolation pair throughout the mesh.

Pass the closed `ConstraintHandler` as `constraints` when using affine constraints:
its master/slave couplings must be present in the sparsity pattern before assembly.
The returned matrices are unconstrained and share that pattern.

`valuetype` defaults to the material's numeric type for a homogeneous medium, and
`Float64` for a material selector. For parameter differentiation through a selector,
pass the promoted coefficient type explicitly. The `CellValues` are reinitialized
in place; the material data and constraints are not changed.
"""
function assemble_biot_matrices(
        dh, cv_u, cv_p, material;
        constraints = nothing,
        valuetype = material isa BiotPoroelastic ? eltype(material) : Float64,
    )
    nu = Ferrite.getnbasefunctions(cv_u)
    np = Ferrite.getnbasefunctions(cv_p)
    n = Ferrite.ndofs_per_cell(dh)
    (
        Ferrite.dof_range(dh, :u) == 1:nu &&
            Ferrite.dof_range(dh, :p) == (nu + 1):(nu + np) && n == nu + np
    ) ||
        throw(ArgumentError("expected exactly the fields :u, :p in that order, matching CellValues"))
    pattern = constraints === nothing ? Ferrite.allocate_matrix(dh) :
        Ferrite.allocate_matrix(dh, constraints)
    K1 = similar(pattern, valuetype)
    K2 = similar(pattern, valuetype)
    as1 = Ferrite.start_assemble(K1)
    as2 = Ferrite.start_assemble(K2)
    ke1 = zeros(valuetype, n, n)
    ke2 = zeros(valuetype, n, n)
    for cell in Ferrite.CellIterator(dh)
        Ferrite.reinit!(cv_u, cell)
        Ferrite.reinit!(cv_p, cell)
        mat = material isa BiotPoroelastic ? material : material(cell)
        biot_element_matrices!(ke1, ke2, mat, cv_u, cv_p)
        dofs = Ferrite.celldofs(cell)
        Ferrite.assemble!(as1, dofs, ke1)
        Ferrite.assemble!(as2, dofs, ke2)
    end
    return K1, K2
end

"""
    assemble_biot_load(dh, facets, fv_u, model; valuetype=Float64) -> Vector

Integrate mechanical surface loads on `facets` using [`facet_load!`](@ref).
The callback receives a zeroed local displacement vector and reinitialized Ferrite
`FacetValues`; it adds the case's traction contributions. They are scattered into
field `:u` of the global load vector, leaving the pressure entries zero.

For several loaded boundaries, sum the returned vectors. A time-dependent load can
be rebuilt inside the `load(t)` callback of [`solve_biot`](@ref). Specify `valuetype`
when the loading carries a numeric type other than `Float64`.
"""
function assemble_biot_load(dh, facets, fv_u, model; valuetype = Float64)
    u_range = Ferrite.dof_range(dh, :u)
    n = Ferrite.getnbasefunctions(fv_u)
    length(u_range) == n || throw(DimensionMismatch("FacetValues must match field :u"))
    f = zeros(valuetype, Ferrite.ndofs(dh))
    fe = zeros(valuetype, n)
    for facet in Ferrite.FacetIterator(dh, facets)
        Ferrite.reinit!(fv_u, facet)
        fill!(fe, zero(valuetype))
        facet_load!(fe, facet, model, fv_u)
        dofs = Ferrite.celldofs(facet)
        for (i, d) in enumerate(u_range)
            f[dofs[d]] += fe[i]
        end
    end
    return f
end
