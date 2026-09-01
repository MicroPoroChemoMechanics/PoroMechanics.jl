# Computational homogenisation on a periodic cell.
#
# The cells here are synthetic, so the suite needs no Bil: a square split into triangles,
# with the phases assigned by region. The comparison against Bil's own `composite0.msh`
# lives in `benchmarks/bil_mechamic.jl`, where it reproduces `MechaMic` to five figures.

using Tensors: Tensors, SymmetricTensor

@testset "Periodic homogenisation" begin

    """
        square_cell(n) -> (nodes, cells, regions)

    A unit square on an `n × n` grid of squares, each split into two triangles. Regions
    alternate in a checkerboard, so a two-phase cell is available without a mesh file.
    """
    function square_cell(n)
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
            r = isodd(i + j) ? 1 : 2
            cells[:, e + 1] = [a, b, c];  regions[e + 1] = r
            cells[:, e + 2] = [a, c, d];  regions[e + 2] = r
            e += 2
        end
        return nodes, cells, regions
    end

    @testset "a homogeneous cell returns its own material" begin
        ## The sharpest check available without a reference: if both phases are the same
        ## material, homogenisation has nothing to do and must return exactly that
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

    @testset "the effective stiffness lies between Reuss and Voigt" begin
        ## Not a tight check, but one that no sign error survives: a two-phase cell cannot
        ## be stiffer than the arithmetic average of its phases nor softer than the harmonic
        ## one. The checkerboard is half and half by construction.
        soft = LinearElastic(; E = 1.0e9, nu = 0.3)
        stiff = LinearElastic(; E = 10.0e9, nu = 0.3)
        nodes, cells, regions = square_cell(6)
        cell = periodic_cell(nodes, cells, regions, Dict(1 => soft, 2 => stiff))

        δ = 1.0e-5
        σ, _ = homogenize_stress(cell, SymmetricTensor{2, 2}((δ, 0.0, 0.0)))
        C11 = σ[1, 1] / δ

        oedometric(m) = m.λ + 2m.μ
        voigt = (oedometric(soft) + oedometric(stiff)) / 2
        reuss = 2 / (1 / oedometric(soft) + 1 / oedometric(stiff))
        @test reuss < C11 < voigt
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
