using PoroMechanics
using Documenter
using DocumenterCitations
using Literate

include("pages.jl")

# ── Example pages are generated, never hand-written ───────────────────────────
# Each `examples/<name>/run.jl` is a Literate script: prose in `#` comments, code in
# between. It stays runnable on its own (`julia --project=examples run.jl`) and is the
# single source for `docs/src/examples/<name>.md`, which is regenerated here on every
# build. Editing the generated markdown is pointless — it is overwritten.
#
# `documenter = true` emits `@example` blocks, so Documenter executes the code while
# building; a page listed in `NONEXECUTED` is emitted as plain `julia` fences instead,
# for cases too heavy or too dependency-hungry to run on every doc build.

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const GENERATED_DIR = joinpath(@__DIR__, "src", "examples")

const DEMOS_DIR = joinpath(@__DIR__, "..", "demos")
const DEMOS_OUT = joinpath(@__DIR__, "src", "demos")
const LITERATE_DEMOS = ["writing_a_model", "parameter_identification", "solver_sensitivity"]

const BENCHMARKS_DIR = joinpath(@__DIR__, "..", "benchmarks")
const VALIDATION_DIR = joinpath(@__DIR__, "src", "validation")

const LITERATE_BENCHMARKS = ["terzaghi", "mandel", "cryer", "deleeuw", "gardner_infiltration", "gardner_transient", "bbm_bil", "bil_richards"]

# `bil_richards` reruns Bil to refine its time step, which no documentation runner has
# installed, so its page is emitted as plain `julia` fences. The measured tables are written
# into the prose rather than produced at build time, for exactly that reason.
const NONEXECUTED_BENCHMARKS = ["bil_richards"]

const LITERATE_EXAMPLES = [
    "fickian_diffusion",
    "darcy_column",
    "richards_1d",
    "nonisothermal_drying",
    "biot_consolidation",
]
const NONEXECUTED = ["biot_consolidation"]   # needs Ferrite + a Gmsh mesh

mkpath(VALIDATION_DIR)

# The benchmark scripts include shared files by relative path. Documenter runs an `@example`
# block with the working directory set to the built page's folder, so those files have to
# sit next to the generated markdown; Documenter copies non-markdown files from src to build.

const BENCHMARK_SHARED = ["laplace.jl", "biot_common.jl", "richards_common.jl"]

for f in BENCHMARK_SHARED
    cp(joinpath(BENCHMARKS_DIR, f), joinpath(VALIDATION_DIR, f); force = true)
end

for name in LITERATE_BENCHMARKS
    Literate.markdown(
        joinpath(BENCHMARKS_DIR, name * ".jl"),
        VALIDATION_DIR;
        name = name,
        documenter = true,
        credit = false,
        codefence = name in NONEXECUTED_BENCHMARKS ?
            ("```julia" => "```") : ("```@example $name" => "```"),
    )
end

mkpath(DEMOS_OUT)
for name in LITERATE_DEMOS
    Literate.markdown(
        joinpath(DEMOS_DIR, name * ".jl"),
        DEMOS_OUT;
        name = name,
        documenter = true,
        credit = false,
        codefence = ("```@example $name" => "```"),
    )
end

mkpath(GENERATED_DIR)
for name in LITERATE_EXAMPLES
    Literate.markdown(
        joinpath(EXAMPLES_DIR, name, "run.jl"),
        GENERATED_DIR;
        name = name,
        documenter = true,
        credit = false,
        codefence = name in NONEXECUTED ? ("```julia" => "```") : ("```@example $name" => "```"),
    )
end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

DocMeta.setdocmeta!(
    PoroMechanics,
    :DocTestSetup,
    :(using PoroMechanics);
    recursive = true,
)

ENV["GKSwstype"] = "100"   # headless GR backend — prevents Plots from hanging in doc builds

makedocs(;
    modules = [PoroMechanics],
    authors = "Anthony Soive and Jean-François Barthélémy",
    sitename = "PoroMechanics.jl",
    format = Documenter.HTML(;
        # MathJax3(config) merges only at the top level: any :tex given here replaces
        # Documenter's whole default, so inlineMath and tags have to be repeated.
        mathengine = Documenter.MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/mhchem"]),
                :tex => Dict(
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                    "packages" => ["base", "ams", "autoload", "mhchem"],
                ),
            )
        ),
        canonical = "https://MicroPoroChemoMechanics.github.io/PoroMechanics.jl",
        repolink = "https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/custom.css"],
        prettyurls = (get(ENV, "CI", nothing) == "true"),
        collapselevel = 1,
        size_threshold_warn = 200_000,
    ),
    pages = pages,
    plugins = [bib],
    warnonly = [:docs_block, :missing_docs],
)

deploydocs(;
    repo = "github.com/MicroPoroChemoMechanics/PoroMechanics.jl.git",
    devbranch = "main",
    push_preview = false,
)
