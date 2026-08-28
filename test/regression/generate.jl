# Regenerate the regression references.
#
#     julia --project test/regression/generate.jl            # all cases
#     julia --project test/regression/generate.jl nonisothermal_drying  # one case
#
# Run this only when a result is *meant* to change, and say why in the commit
# message. The point of the harness is that references move deliberately.

using Printf

include("cases.jl")

selected = isempty(ARGS) ? CASES : filter(c -> c.name in ARGS, CASES)

if isempty(selected)
    println("No case matched $(ARGS). Known cases: ", join((c.name for c in CASES), ", "))
    exit(1)
end

for case in selected
    print(rpad(case.name, 16))
    elapsed = @elapsed values = run_silently(case.signature)
    path = write_reference(case.name, values)
    @printf("%6d values  %6.1f s  → %s\n", length(values), elapsed, relpath(path, pwd()))
end
