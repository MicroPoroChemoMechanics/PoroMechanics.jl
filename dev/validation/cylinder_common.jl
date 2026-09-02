# Thick-walled cylinder under internal pressure — the solver, shared between the
# documentation page and the test suite.
#
# Deliberately free of any homogenisation: it takes a Young's modulus and a Poisson ratio
# and knows nothing about where they came from. `benchmarks/mfh_thick_cylinder.jl` feeds it
# moduli upscaled from a microstructure; `test/benchmarks.jl` feeds it the same moduli
# frozen as literals, so the suite checks the page's numbers without needing the other
# package installed.

using PoroMechanics
using Ferrite
using PoroMechanics.Tensors

const RI, RO, PRESSURE = 0.1, 1.0, 20.0e6

"Radial displacement of Lamé's solution, plane strain, internal pressure only."
lame_ur(r, E, ν) = (1 + ν) * PRESSURE * RI^2 / (E * (RO^2 - RI^2)) * ((1 - 2ν) * r + RO^2 / r)

"Hoop stress of the same solution — independent of the moduli."
lame_σθθ(r) = PRESSURE * RI^2 / (RO^2 - RI^2) * (1 + RO^2 / r^2)

"""
    solve_cylinder(E, ν, nr) -> (; err_u, err_σ, residual, ndofs)

Solve the cylinder with `nr` radial elements and return the errors against Lamé, relative
to the peak of each field.

A thick cylinder in plane strain is an axisymmetric problem in disguise: the solution
depends on ``r`` alone, ``\\varepsilon_{zz}`` vanishes and ``\\varepsilon_{\\theta\\theta} =
u_r/r``. So the mesh is a single-element-tall strip in ``(r, z)`` with `u_z` held at zero on
both faces — which imposes plane strain exactly rather than approximating it — and
[`assemble_axisymmetric!`](@ref) supplies the hoop term. One radial row of elements does the
work of a full annulus.
"""
function solve_cylinder(E, ν, nr)
    h = (RO - RI) / nr
    grid = generate_grid(Quadrilateral, (nr, 1), Vec(RI, 0.0), Vec(RO, h))
    ip = Lagrange{RefQuadrilateral, 1}()^2
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    geo = Lagrange{RefQuadrilateral, 1}()
    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), ip, geo)
    fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip, geo)

    ## `u_z = 0` on both z faces is plane strain, and it also removes the axial rigid body
    ## motion. Nothing constrains `u_r`: the pressure equilibrates against the hoop stress.
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0, [2]))
    close!(ch); update!(ch, 0.0)

    ## Internal pressure on r = RI. The outward normal of the domain there is -e_r, so the
    ## traction is +p e_r, and the facet integral carries the axisymmetric weight r.
    fext = zeros(ndofs(dh))
    fe = zeros(ndofs_per_cell(dh))
    for fc in FacetIterator(dh, getfacetset(grid, "left"))
        reinit!(fv, fc)
        fill!(fe, 0)
        coords = getcoordinates(fc)
        for q in 1:getnquadpoints(fv)
            r = spatial_coordinate(fv, q, coords)[1]
            dΓ = getdetJdV(fv, q) * r
            for i in 1:getnbasefunctions(fv)
                fe[i] += (shape_value(fv, q, i) ⋅ Vec(PRESSURE, 0.0)) * dΓ
            end
        end
        fext[celldofs(fc)] .+= fe
    end

    mat = PoroMechanics.LinearElastic(; E = E, nu = ν)
    nq = getnquadpoints(cv)
    mk() = [[PoroMechanics.initial_state(mat) for _ in 1:nq] for _ in 1:getncells(grid)]
    states, old = mk(), mk()
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    u = zeros(ndofs(dh))
    norms = newton_solve!(u, K, f, dh, cv, mat, states, old, ch, 1.0; fext, tol = 1.0e-6)

    ## Displacement at the nodes — never by indexing `u` with a node number, since a
    ## `DofHandler` numbers degrees of freedom by cell traversal.
    un = evaluate_at_grid_nodes(dh, u, :u)
    err_u = ref_u = 0.0
    for (n, x) in enumerate(grid.nodes)
        exact = lame_ur(x.x[1], E, ν)
        err_u = max(err_u, abs(un[n][1] - exact))
        ref_u = max(ref_u, abs(exact))
    end

    ## Stress at the quadrature points, where the finite element stress actually lives.
    ## Index order is (r, z, θ), so the hoop component is the third — not the second.
    err_σ = ref_σ = 0.0
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        coords = getcoordinates(cell)
        ue = u[celldofs(cell)]
        for q in 1:getnquadpoints(cv)
            r = spatial_coordinate(cv, q, coords)[1]
            ε = PoroMechanics.axisymmetric_strain(cv, q, ue, coords)
            σ, _, _ = PoroMechanics.material_response(mat, ε, PoroMechanics.NoState(), 1.0)
            err_σ = max(err_σ, abs(σ[3, 3] - lame_σθθ(r)))
            ref_σ = max(ref_σ, abs(lame_σθθ(r)))
        end
    end

    return (; err_u = err_u / ref_u, err_σ = err_σ / ref_σ,
        residual = last(norms), ndofs = ndofs(dh))
end
