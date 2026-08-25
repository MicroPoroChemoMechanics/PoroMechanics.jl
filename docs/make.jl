using PoroMechanics
using Documenter
using DocumenterCitations

include("pages.jl")

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
    authors = "Jean-François Barthélémy and Anthony Soive",
    sitename = "PoroMechanics.jl",
    format = Documenter.HTML(;
        mathengine = Documenter.MathJax3(
            Dict(
                :loader => Dict("load" => ["[tex]/mhchem"]),
                :tex => Dict("packages" => Dict("[+]" => ["mhchem"])),
            )
        ),
        canonical = "https://MicroPoroChemoMechanics.github.io/PoroMechanics.jl",
        repolink = "https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl",
        edit_link = "main",
        assets = ["assets/custom.css"],
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
