module BiotBackendTests
using Test, PoroMechanics, Ferrite, LinearAlgebra, ForwardDiff

struct SurfaceTraction <: AbstractPoroModel
    magnitude::Float64
end
function PoroMechanics.facet_load!(fe, facet, m::SurfaceTraction, fv)
    for q in 1:getnquadpoints(fv), i in 1:getnbasefunctions(fv)
        fe[i] -= shape_value(fv, q, i)[2] * m.magnitude * getdetJdV(fv, q)
    end
    return nothing
end

# Two layers under uniform compression: the stress is constant and each layer's
# strain is -traction/E. This checks region selection, assembly, scattering of
# surface loads and the global solve against a mechanical solution.
function layered_column(E; check = false)
    grid = generate_grid(Quadrilateral, (2, 2), Vec(0.0, 0.0), Vec(1.0, 1.0))
    ip = Lagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip^2)
    add!(dh, :p, ip)
    close!(dh)
    qr = QuadratureRule{RefQuadrilateral}(2)
    cv_u, cv_p = CellValues(qr, ip^2), CellValues(qr, ip)
    fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip^2)
    ch = ConstraintHandler(dh)
    for side in ("left", "right")
        add!(ch, Dirichlet(:u, getfacetset(grid, side), (x, t) -> 0.0, [1]))
    end
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, [2]))
    addnodeset!(grid, "all", Set(1:getnnodes(grid)))
    add!(ch, Dirichlet(:p, getnodeset(grid, "all"), (x, t) -> 0.0))
    close!(ch)
    lower = BiotPoroelastic(; E, nu = 0.0, b = 0.5, N = 1.0, k = 1.0, mu_l = 1.0)
    upper = BiotPoroelastic(; E = 200.0, nu = 0.0, b = 0.5, N = 1.0, k = 1.0, mu_l = 1.0)
    material_at(cell) = sum(x[2] for x in getcoordinates(cell)) / 4 < 0.5 ? lower : upper
    K1, K2 = assemble_biot_matrices(dh, cv_u, cv_p, material_at; constraints = ch, valuetype = typeof(E))
    f = assemble_biot_load(dh, getfacetset(grid, "top"), fv, SurfaceTraction(2.0); valuetype = typeof(E))
    uy = node_dof_maps(dh, grid, (:u, 2)).u
    pdofs = node_dof_maps(dh, grid, :p).p
    x = solve_biot(K1, K2, ch; inival = zeros(typeof(E), ndofs(dh)), times = [0.0, 0.2, 1.0], load = f, linsolve = (A, b) -> Matrix(A) \ b)
    if check
        @test sum(f[uy]) ≈ -2.0
        @test all(iszero, f[pdofs])
        y = [node.x[2] for node in grid.nodes]
        expected = [-2 * (min(z, 0.5) / E + max(z - 0.5, 0) / 200) for z in y]
        @test x[uy] ≈ expected rtol = 1.0e-12
        @test all(iszero, x[pdofs])
        # The homogeneous overload and the cell-selector overload agree.
        H1, H2 = assemble_biot_matrices(dh, cv_u, cv_p, lower; constraints = ch)
        S1, S2 = assemble_biot_matrices(dh, cv_u, cv_p, _ -> lower; constraints = ch)
        @test H1 == S1
        @test H2 == S2
    end
    top = findfirst(node -> node.x[2] == 1.0, grid.nodes)
    return x[uy[top]]
end

@testset "Linear Biot assembly and evolution" begin
    @testset "Layered column and material sensitivity" begin
        @test layered_column(100.0; check = true) ≈ -0.015
        @test ForwardDiff.derivative(layered_column, 100.0) ≈ 1.0e-4 rtol = 1.0e-10
    end

    @testset "Variable steps, loads and affine constraints" begin
        grid = generate_grid(Line, (4,))
        dh = DofHandler(grid)
        add!(dh, :p, Lagrange{RefLine, 1}())
        close!(dh)
        dofs = node_dof_maps(dh, grid, :p).p
        ch = ConstraintHandler(dh)
        addnodeset!(grid, "first", Set([1]))
        add!(ch, Dirichlet(:p, getnodeset(grid, "first"), (x, t) -> t))
        add!(ch, AffineConstraint(dofs[5], [dofs[2] => 2.0], 1.0))
        close!(ch)
        K1 = allocate_matrix(dh, ch)
        K2 = copy(K1)
        for i in 1:ndofs(dh)
            K1[i, i] = 2.0
            K2[i, i] = 3.0
        end
        # The last cell couples a slave to node 4, creating a master coupling
        # between nodes 2 and 4 that is absent from the unconstrained mesh.
        K1[dofs[4], dofs[5]] = K1[dofs[5], dofs[4]] = -0.25
        # x(t) is linear, so backward Euler is exact on any time partition.
        exact(t) = [t, 2t, 3t, 4t, 4t + 1][invperm(dofs)]
        velocity = [1.0, 2.0, 3.0, 4.0, 4.0][invperm(dofs)]
        forcing(t) = K1 * exact(t) + K2 * velocity
        initial = exact(0.0)
        initial[dofs[1]] = -9.0
        initial[dofs[5]] = -9.0
        saved_initial = copy(initial)
        saved_K1, saved_K2 = copy(K1), copy(K2)
        snapshots = []
        record(x, t, step) = push!(snapshots, (copy(x), t, step))
        times = [0.0, 0.1, 0.4, 1.0]
        result = solve_biot(K1, K2, ch; inival = initial, times, load = forcing, on_step = record)
        @test length(snapshots) == 3
        for (i, (x, t, step)) in enumerate(snapshots)
            @test t == times[i + 1]
            @test step == i
            @test x ≈ exact(t) rtol = 1.0e-12
        end
        @test result ≈ exact(1.0)
        @test initial == saved_initial
        @test K1 == saved_K1
        @test K2 == saved_K2
        @test solve_biot(K1, K2, ch; inival = initial, times = [0.0]) ≈ exact(0.0)
        f = forcing(0.4)
        saved_f = copy(f)
        a = solve_biot(K1, K2, ch; inival = exact(0.0), times = [0.0, 0.4], load = f)
        b = solve_biot(K1, K2, ch; inival = exact(0.0), times = [0.0, 0.4], load = _ -> f)
        @test a == b
        @test f == saved_f
        z = solve_biot(K1, K2, ch; inival = exact(0.0), times = [0.0, 0.4])
        @test z ≈ solve_biot(K1, K2, ch; inival = exact(0.0), times = [0.0, 0.4], load = zeros(5))
        for bad_times in (Float64[], [0.0, 0.0], [1.0, 0.0], [0.0, Inf])
            @test_throws ArgumentError solve_biot(K1, K2, ch; inival = initial, times = bad_times)
        end
        @test_throws DimensionMismatch solve_biot(K1, K2, ch; inival = initial, times, load = zeros(2))
    end
end
end
