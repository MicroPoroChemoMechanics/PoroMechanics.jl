# Bil reference values for the benchmark pages.
#
# `bbm_bil.jl` used to carry its reference numbers as a `Dict` literal, copied by hand out
# of `base/BBM/BBM.p1`. That is unreviewable — nobody can tell a typo from a result — and
# unregenerable.
#
# This file reads the file instead, and keeps the literals only as a cache for the machines
# that have no Bil: the documentation runner, and anyone who has not cloned it. When Bil
# *is* present the two are compared, so the cache cannot rot silently.

# The harness lives in `test/bil/`, but `docs/make.jl` copies it next to the generated page
# so that a Documenter `@example` block — whose working directory is the built page's folder
# — can still find it. Look beside this file first, fall back to the repository layout.
include(
    let here = joinpath(@__DIR__, "harness.jl")
        isfile(here) ? here : joinpath(@__DIR__, "..", "test", "bil", "harness.jl")
    end
)

"""
    bil_bbm_reference() -> Dict{Float64, NTuple{3, Float64}}

`(tr ε, εv_p, pc*)` at each of the six dates of `base/BBM/BBM`.

Three conversions turn Bil's output into these quantities, and none of them is guessable
from the column names:

* **`tr ε`** comes from `Void_ratio_variation`, which is ``\\Delta e = (1 + e_0)\\,
  \\mathrm{tr}\\,\\varepsilon``. The deck's initial porosity is 0.25, so ``e_0 = 1/3``.
* **`εv_p`** is the trace of `Plastic_strains`, a 3×3 tensor written as nine components,
  **with the sign flipped**: Bil stores a compressive plastic strain as negative, while this
  package and the page count a compaction as positive.
* **`pc*`** is `exp(Hardening_variable)` — Bil hardens on the logarithm. The deck starts at
  40 kPa and the file's first value is 10.59663 = ln(40000).
"""
function bil_bbm_reference()
    frozen = BBM_REFERENCE_CACHE
    bil_root() === nothing && return frozen

    out = read_bil(joinpath(case_dir("BBM"), "BBM.p1"))
    t = times(out)
    plastic = out["Plastic_strains"]
    hardening = column(out, "Hardening_variable")
    dvoid = column(out, "Void_ratio_variation")
    e0 = 1 / 3

    read_values = Dict{Float64, NTuple{3, Float64}}()
    for date in 1.0:6.0
        i = findlast(<=(date + 1.0e-9), t)
        read_values[date] = (
            dvoid[i] / (1 + e0),
            -(plastic[i, 1] + plastic[i, 5] + plastic[i, 9]),
            exp(hardening[i]),
        )
    end

    ## The cache is only allowed to disagree by the rounding it was written with.
    for (date, cached) in frozen
        got = read_values[date]
        for (a, b) in zip(got, cached)
            isapprox(a, b; rtol = 1.0e-5, atol = 1.0e-12) || error(
                "BBM_REFERENCE_CACHE is stale at t = $date: cached $cached, file gives $got"
            )
        end
    end
    return read_values
end

"""
    BBM_REFERENCE_CACHE

The same numbers, for a machine without Bil. Regenerate by running this page with
`BIL_ROOT` set and printing `bil_bbm_reference()`; the check above will refuse a stale copy
before it can mislead anyone.
"""
const BBM_REFERENCE_CACHE = Dict{Float64, NTuple{3, Float64}}(
    1.0 => (-3.058474e-2, 0.0, 39999.81),
    2.0 => (-1.550205e-3, 0.0, 39999.81),
    3.0 => (-5.207435e-2, 1.411818e-2, 56683.02),
    4.0 => (-1.711651e-2, 1.411818e-2, 56683.02),
    5.0 => (-7.440369e-2, 2.917845e-2, 82214.75),
    6.0 => (-3.426270e-2, 2.917845e-2, 82214.75),
)
