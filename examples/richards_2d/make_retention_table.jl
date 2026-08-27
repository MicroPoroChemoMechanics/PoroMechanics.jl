# examples/richards_2d/make_retention_table.jl
#
# Generates `retention_table.dat`, the tabulated retention curve read by run.jl.
#
# The table is SYNTHETIC: a Van Genuchten / Mualem fit for a coarse granular
# medium, not measured data. It exists so that the example runs out of the box.
# Replace it with your own measurements when comparing against an experiment —
# run.jl also accepts a path through the RICHARDS_2D_DATA environment variable.
#
#   S_e(p_c) = [1 + (α·p_c)^n]^(-m),        m = 1 - 1/n
#   S_l      = S_r + (1 - S_r)·S_e
#   k_rl     = √S_e · [1 - (1 - S_e^(1/m))^m]²
#
# Usage:
#   julia --project=examples examples/richards_2d/make_retention_table.jl

using Printf

const α   = 1 / 577.0   # [1/Pa] inverse air-entry pressure
const n   = 4.0         # [-] pore-size distribution index (coarse medium)
const m   = 1 - 1 / n
const S_r = 0.05        # [-] residual saturation

Se(pc) = pc <= 0 ? 1.0 : (1 + (α * pc)^n)^(-m)

function krl(pc)
    se = clamp(Se(pc), 0.0, 1.0)
    se < 1.0e-12 && return 0.0
    return sqrt(se) * (1 - (1 - se^(1 / m))^m)^2
end

out = joinpath(@__DIR__, "retention_table.dat")
open(out, "w") do io
    for pc in range(0.0, 50_000.0; length = 501)
        se = Se(pc)
        @printf(io, "%12.4f  %10.8f  %12.10f\n", pc, S_r + (1 - S_r) * se, krl(pc))
    end
end
println("written: ", out)
