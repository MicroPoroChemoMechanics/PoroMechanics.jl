using Ferrite
using LinearAlgebra

@testset "Newton convergence and rejected steps" begin
    grid = Ferrite.generate_grid(Ferrite.Quadrilateral, (2, 1), Ferrite.Vec(1.0, 0.0), Ferrite.Vec(2.0, 1.0))
    ip = Ferrite.Lagrange{Ferrite.RefQuadrilateral, 1}()^2
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip)
    Ferrite.close!(dh)
    cv = Ferrite.CellValues(Ferrite.QuadratureRule{Ferrite.RefQuadrilateral}(2), ip)
    ch = Ferrite.ConstraintHandler(dh)
    for side in ("top", "bottom")
        Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(grid, side), (x, t) -> 0.0, [2]))
    end
    Ferrite.close!(ch)
    Ferrite.update!(ch, 0.0)
    mat = LinearElastic(; E = 1.0e4, nu = 0.25)
    states = [[initial_state(mat) for _ in 1:Ferrite.getnquadpoints(cv)] for _ in 1:Ferrite.getncells(grid)]
    old = deepcopy(states)
    K = Ferrite.allocate_matrix(dh)
    f = zeros(Ferrite.ndofs(dh))
    u = zero(f)
    fext = ones(length(f))
    solve!(; kwargs...) = newton_solve!(u, K, f, dh, cv, mat, states, old, ch, 1.0; fext, kwargs...)

    @test_throws ArgumentError solve!(; maxhalve = 0)
    @test_throws ArgumentError solve!(; tol = 0.0)
    @test_throws ErrorException solve!(; maxiter = 0)
    @test iszero(u)
    initial_residual = copy(f)

    # An uphill direction must fail without accepting its final trial.
    @test_throws ErrorException solve!(; linsolve = (K, f) -> -(K \ f), maxhalve = 3)
    @test iszero(u)
    @test f ≈ initial_residual

    # A downhill correction need not be converged: exhausting the budget must fail.
    @test_throws ErrorException solve!(; linsolve = (K, f) -> (K \ f) / 2, maxiter = 1)
    @test norm(u) > 0
    @test norm(f) ≈ norm(initial_residual) / 2

    # The last allowed correction can converge and must appear in the history.
    fill!(u, 0)
    history = solve!(; maxiter = 1)
    @test length(history) == 2
    @test last(history) < 1.0e-8
    @test last(history) ≈ norm(f)
    @test length(solve!(; maxiter = 0)) == 1
end
