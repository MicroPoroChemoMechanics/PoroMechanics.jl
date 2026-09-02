# Computational homogenization on a periodic cell.
#
# The cells here are synthetic, so the suite needs no Bil: a square split into triangles,
# with the phases assigned by region. The comparison against Bil's own `composite0.msh`
# lives in `benchmarks/bil_mechamic.jl`, where it reproduces `MechaMic` to five figures.

using Tensors: Tensors, SymmetricTensor

@testset "Periodic homogenisation" begin

    """
        square_cell(n) -> (nodes, cells, regions)

    A unit square on an `n × n` grid of squares, each split into two triangles, so a
    two-phase cell is available without a mesh file. `pattern` selects how the phases are
    laid out: `:checkerboard` alternates in both directions, `:layers` stacks bands along
    `y`.
    """
    function square_cell(n; pattern = :checkerboard)
        coords = range(-0.5, 0.5; length = n + 1)
        nodes = zeros(2, (n + 1)^2)
        idx(i, j) = (j - 1) * (n + 1) + i
        for j in 1:(n + 1), i in 1:(n + 1)
            nodes[1, idx(i, j)] = coords[i]
            nodes[2, idx(i, j)] = coords[j]
        end
        cells = zeros(Int, 3, 2n^2)
        regions = zeros(Int, 2n^2)
        e = 0
        for j in 1:n, i in 1:n
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            r = (pattern === :layers ? isodd(j) : isodd(i + j)) ? 1 : 2
            cells[:, e + 1] = [a, b, c];  regions[e + 1] = r
            cells[:, e + 2] = [a, c, d];  regions[e + 2] = r
            e += 2
        end
        return nodes, cells, regions
    end

    @testset "a homogeneous cell returns its own material" begin
        ## The sharpest check available without a reference: if both phases are the same
        ## material, homogenization has nothing to do and must return exactly that
        ## material's stiffness. Anything wrong with the periodic constraints, the volume
        ## average or the fluctuation solve shows up here as a discrepancy from a number
        ## known in closed form.
        mat = LinearElastic(; E = 3.0e9, nu = 0.25)
        nodes, cells, regions = square_cell(4)
        cell = periodic_cell(nodes, cells, regions, Dict(1 => mat, 2 => mat))

        λ, μ = mat.λ, mat.μ
        for (ε2, name) in (
                (SymmetricTensor{2, 2}((1.0e-5, 0.0, 0.0)), "ε11"),
                (SymmetricTensor{2, 2}((0.0, 0.0, 1.0e-5)), "ε22"),
                (SymmetricTensor{2, 2}((0.0, 5.0e-6, 0.0)), "ε12"),
            )
            σ, _ = homogenize_stress(cell, ε2)
            ε3 = plane_strain(ε2)
            σ_exact = λ * Tensors.tr(ε3) * one(ε3) + 2μ * ε3
            @test isapprox(σ, σ_exact; rtol = 1.0e-10) || error("homogeneous cell wrong for $name")
        end
    end

    @testset "a layered cell reproduces Reuss and Voigt exactly" begin
        ## A checkerboard is the wrong cell to check against bounds: by its diagonal
        ## symmetry the uniform strain field is *already* in equilibrium, the fluctuation
        ## is identically zero, and the answer sits exactly on the Voigt bound. A bracket
        ## test on it passes on rounding noise and proves nothing.
        ##
        ## A stack of layers has no such degeneracy, and a closed form on both axes.
        ## Across the layers the stress is uniform and the cell is exactly Reuss. Along
        ## them it is *not* Voigt: plane strain holds ε₂₂ at zero on average, the two
        ## phases then disagree on σ₂₂, and a transverse fluctuation develops. Solving the
        ## laminate for that fluctuation gives a second exact number — a sharper check
        ## than a bound, because it is only reached if the fluctuation is right.
        soft = LinearElastic(; E = 1.0e9, nu = 0.3)
        stiff = LinearElastic(; E = 10.0e9, nu = 0.3)
        nodes, cells, regions = square_cell(6; pattern = :layers)
        cell = periodic_cell(nodes, cells, regions, Dict(1 => soft, 2 => stiff))

        oedometric(m) = m.λ + 2m.μ
        mean(f) = (f(soft) + f(stiff)) / 2          # equal volume fractions
        reuss = 1 / mean(m -> 1 / oedometric(m))

        ## Laminate loaded along the layers, ε₂₂ = 0 on average. σ₂₂ is continuous, so
        ## σ₂₂ = ⟨λ/oed⟩/⟨1/oed⟩ · ε₁₁, and C₁₁ follows from ⟨σ₁₁⟩.
        s = mean(m -> m.λ / oedometric(m)) / mean(m -> 1 / oedometric(m))
        along = mean(oedometric) + s * mean(m -> m.λ / oedometric(m)) -
            mean(m -> m.λ^2 / oedometric(m))

        δ = 1.0e-5
        ## Layers stacked along y: loading along y crosses them, loading along x runs with
        ## them. Only the diagonal term is exact — the transverse stress is not.
        σ_across, _ = homogenize_stress(cell, SymmetricTensor{2, 2}((0.0, 0.0, δ)))
        σ_along, _ = homogenize_stress(cell, SymmetricTensor{2, 2}((δ, 0.0, 0.0)))

        @test σ_across[2, 2] / δ ≈ reuss rtol = 1.0e-12
        @test σ_along[1, 1] / δ ≈ along rtol = 1.0e-12
        ## And the laminate is strictly softer along the layers than the Voigt average,
        ## which is what the transverse fluctuation costs.
        @test σ_along[1, 1] / δ < mean(oedometric)
    end

    @testset "a plastic cell follows the point material it is made of" begin
        ## The path-dependent counterpart of the first test, and the one that exercises the
        ## quadrature-point states: a cell made entirely of one Drucker-Prager material has
        ## no fluctuation to develop, so its averaged stress must equal the stress the point
        ## material reaches along the *same* path — through yield, not only before it.
        mat = DruckerPrager(;
            E = 2713.0e6, nu = 0.339, cohesion = 1.5e6,
            friction = deg2rad(25), dilatancy = deg2rad(25)
        )
        nodes, cells, regions = square_cell(4)
        cell = periodic_cell(nodes, cells, regions, Dict(1 => mat, 2 => mat))

        ## The path has to be deviatoric. Compressing at ε₁₁ = 0 raises the confinement
        ## as fast as the deviator — at ε₂₂ = −2 % the material sits at σ₂₂ = −83 MPa,
        ## σ₁₁ = −43 MPa and is *still* elastic, because a friction angle of 25° puts the
        ## yield surface out of reach on that path. In-plane isochoric shear does not.
        states = cell_states(cell)
        point = initial_state(mat)
        for k in 1:20
            ε_M = SymmetricTensor{2, 2}((5.0e-4 * k, 0.0, -5.0e-4 * k))
            σ_cell, _, states = homogenize_stress(cell, ε_M, states)
            σ_point, _, point = material_response(mat, plane_strain(ε_M), point, 1.0)
            @test σ_cell ≈ σ_point rtol = 1.0e-10
        end
        ## And the path has to have gone plastic, or the test proves nothing new.
        @test point.γp > 0
    end

    @testset "stress control reaches the macroscopic target" begin
        ## `homogenize_to_stress` inverts the cell: it is handed a stress and must find the
        ## strain. Plane strain with σ₁₁ = σ₁₂ = 0 is uniaxial macroscopic loading, and a
        ## dilatant material under vertical compression must expand sideways.
        matrix = DruckerPrager(;
            E = 2713.0e6, nu = 0.339, cohesion = 1.5e6,
            friction = deg2rad(25), dilatancy = deg2rad(25)
        )
        inclusion = LinearElastic(; E = 5000.0e6, nu = 0.49)
        nodes, cells, regions = square_cell(4)

        "Ramp to −16 MPa in eight steps, and return the final macroscopic state."
        function ramp(matrix)
            c = periodic_cell(nodes, cells, regions, Dict(1 => matrix, 2 => inclusion))
            states, ε = cell_states(c), zero(SymmetricTensor{2, 2})
            local out
            for k in 1:8
                out = homogenize_to_stress(
                    c, SymmetricTensor{2, 2}((0.0, 0.0, -2.0e6 * k)), states; ε_guess = ε
                )
                ε, states = out.ε, out.states
            end
            return out
        end

        plastic = ramp(matrix)
        ## The same cell with the plasticity taken out, as the control.
        elastic = ramp(LinearElastic(; E = 2713.0e6, nu = 0.339))

        @test plastic.σ[2, 2] ≈ -16.0e6 rtol = 1.0e-7
        @test abs(plastic.σ[1, 1]) < 1.0e-1     # target is zero; the scale is 16 MPa
        @test abs(plastic.σ[1, 2]) < 1.0e-1
        @test plastic.ε[2, 2] < 0 && plastic.ε[1, 1] > 0

        ## What plasticity does, stated against its own elastic control rather than against
        ## a number: the cell shortens more, and the dilatancy angle makes it spread wider
        ## per unit of shortening than any elastic Poisson effect could.
        @test plastic.ε[2, 2] < elastic.ε[2, 2]
        @test -plastic.ε[1, 1] / plastic.ε[2, 2] > -elastic.ε[1, 1] / elastic.ε[2, 2]
    end

    @testset "the effective stiffness is symmetric" begin
        nodes, cells, regions = square_cell(4)
        cell = periodic_cell(
            nodes, cells, regions,
            Dict(1 => LinearElastic(; E = 2.0e9, nu = 0.2), 2 => LinearElastic(; E = 8.0e9, nu = 0.4)),
        )
        δ = 1.0e-5
        σ1, _ = homogenize_stress(cell, SymmetricTensor{2, 2}((δ, 0.0, 0.0)))
        σ2, _ = homogenize_stress(cell, SymmetricTensor{2, 2}((0.0, 0.0, δ)))
        ## C12 = C21 by construction of an elastic energy; a broken periodic pairing or a
        ## missing transpose in the assembly breaks it.
        @test isapprox(σ1[2, 2], σ2[1, 1]; rtol = 1.0e-8)
        ## And the checkerboard is symmetric under x ↔ y, so the diagonal terms match too.
        @test isapprox(σ1[1, 1], σ2[2, 2]; rtol = 1.0e-8)
    end
end
