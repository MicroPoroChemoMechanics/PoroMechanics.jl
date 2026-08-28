# Shared by the poroelasticity benchmarks.
#
# The material and its derived constants now live in the package itself, as
# `BiotPoroelastic` in `src/Constitutive/Poroelasticity.jl`; the element matrices in
# `src/Backends/FEM.jl`. What remains here is the scheduling helper the benchmarks share and
# an alias that keeps their call sites readable.

using PoroMechanics
using Ferrite
using LinearAlgebra
using SparseArrays
using Printf

include("laplace.jl")

"""
    HomogeneousBiot(; kwargs...)

The benchmarks' name for [`BiotPoroelastic`](@ref), which is what they were using before it
was promoted into the package.
"""
const HomogeneousBiot = BiotPoroelastic

"""
    uniform_schedule(T_probe; T_start, dT) -> Vector

Dimensionless times to march through: one short step to `T_start`, which captures the
undrained response, then a uniform grid of spacing `dT` merged with the probe times so that
each probe is landed on exactly.
"""
function uniform_schedule(T_probe; T_start, dT)
    T_max = maximum(T_probe)
    grid = T_start .+ dT .* (1:ceil(Int, (T_max - T_start) / dT))
    schedule = sort(unique(vcat(T_start, grid, T_probe)))
    return filter(<=(T_max + 1.0e-12), schedule)
end
