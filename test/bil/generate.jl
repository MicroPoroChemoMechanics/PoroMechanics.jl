# Freeze Bil's answers into `references/`, so the test suite never needs Bil.
#
#     julia --project test/bil/generate.jl              # every case
#     julia --project test/bil/generate.jl richards_2d  # one case
#
# Requires Bil's source tree (`BIL_ROOT`) for the decks and their shipped outputs. It does
# **not** require the binary unless `--drift` is passed.
#
# Regenerating is a deliberate act, exactly as in `test/regression/generate.jl`: a reference
# that changes by itself is not a reference. Before regenerating because a comparison
# started failing, check first whether it is *ours* that moved.
#
# ## `--drift`
#
#     julia --project test/bil/generate.jl --drift
#
# Re-runs Bil on each case and compares its fresh output to the one committed under `base/`.
# Do this before trusting a reference, and before blaming this package for a disagreement:
# the files in `base/` were written by earlier versions of Bil (2.9 for `Chloricem`, 3.0.0
# for `BBM`), and a gap that already exists between Bil and its own shipped output is not
# ours to explain.
#
# Measured on 2026-08-30 with Bil 2.11: `BBM` agrees with its 3.0.0 reference to 1e-13 and
# `Richards-2d` with its 2.9 reference to 1e-13. The shipped outputs are trustworthy.

include("cases.jl")

function main(args)
    drift = "--drift" in args
    wanted = filter(a -> !startswith(a, "--"), args)

    if bil_root() === nothing
        println("Bil source tree not found. Set BIL_ROOT to the checkout, e.g.")
        println("  BIL_ROOT=~/Documents/Modelisation/bil-master julia --project test/bil/generate.jl")
        return 1
    end

    cases = isempty(wanted) ? BIL_CASES : filter(c -> c.name in wanted, BIL_CASES)
    if isempty(cases)
        println("no such case: ", join(wanted, ", "))
        println("known cases: ", join((c.name for c in BIL_CASES), ", "))
        return 1
    end

    if drift && bil_executable() === nothing
        println("--drift needs the `bil` binary; set BIL_EXE or put it on the PATH")
        return 1
    end

    for case in cases
        println("── ", case.name, " — base/", case.relative, ", deck ", case.deck)
        if drift
            drift_report(case.relative, case.deck)
            println()
        end
        (; values, version) = case.bil()
        path = write_bil_reference(case, values, version)
        println(
            "  froze ", length(values), " values from Bil ", version,
            " → ", relpath(path, dirname(@__DIR__)),
        )
    end
    return 0
end

exit(main(ARGS))
