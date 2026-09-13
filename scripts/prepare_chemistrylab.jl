# Prepare the pinned upstream source plus the reviewed sensitivity patch.
# Run before instantiating the examples/test environments on a fresh checkout.
const ROOT = dirname(@__DIR__)
const SOURCE = joinpath(ROOT, "local", "ChemistryLab-0.15.2")
const PATCH = joinpath(ROOT, "patches", "ChemistryLab-0.15.2-sensitivity.patch")
const REVISION = "3e481770f650877830f2279e46d6108f30dbfe92"
if !isdir(SOURCE)
    mkpath(dirname(SOURCE))
    run(`git clone --depth 1 --branch v0.15.2 https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl.git $SOURCE`)
end
readchomp(`git -C $SOURCE rev-parse HEAD`) == REVISION ||
    error("ChemistryLab source has a different revision; preserve it and use a separate checkout")
if success(pipeline(`git -C $SOURCE apply --reverse --check $PATCH`; stdout = devnull, stderr = devnull))
    println("ChemistryLab 0.15.2 sensitivity patch already applied")
else
    run(`git -C $SOURCE apply --check $PATCH`)
    run(`git -C $SOURCE apply $PATCH`)
    println("Applied ChemistryLab 0.15.2 sensitivity patch")
end
println("Development source: ", SOURCE)
