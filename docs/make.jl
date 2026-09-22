using PoroMechanics
using Documenter
using DocumenterCitations
# VitePress renders the site from the Markdown that Documenter emits, and typesets
# every formula at build time into static SVG, so no MathJax bundle is ever fetched
# by the reader's browser. The TeX extensions it loads — mhchem included — are
# declared in `docs/src/.vitepress/mathjax-plugin.ts`, not in a `mathengine` here.
using DocumenterVitepress
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
const LITERATE_DEMOS = ["writing_a_model", "parameter_identification", "solver_sensitivity", "reactive_transport"]

const BENCHMARKS_DIR = joinpath(@__DIR__, "..", "benchmarks")
const VALIDATION_DIR = joinpath(@__DIR__, "src", "validation")

const LITERATE_BENCHMARKS = ["terzaghi", "mandel", "cryer", "deleeuw", "gardner_infiltration", "gardner_transient", "bbm_bil", "bil_richards", "bil_poroplast", "bil_mechamic", "mfh_poroelastic", "mfh_thick_cylinder"]

# `bil_richards` reruns Bil to refine its time step, which no documentation runner has
# installed, so its page is emitted as plain `julia` fences. The measured tables are written
# into the prose rather than produced at build time, for exactly that reason.
const NONEXECUTED_BENCHMARKS = ["bil_richards", "bil_poroplast", "bil_mechamic"]

const LITERATE_EXAMPLES = [
    "fickian_diffusion",
    "fickian_identification",
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

const BENCHMARK_SHARED = ["laplace.jl", "biot_common.jl", "richards_common.jl", "bil_common.jl", "cylinder_common.jl"]

for f in BENCHMARK_SHARED
    cp(joinpath(BENCHMARKS_DIR, f), joinpath(VALIDATION_DIR, f); force = true)
end

# `bil_common.jl` includes the Bil output reader, which lives with the tests rather than the
# benchmarks; it has to travel with it. On a runner without Bil the reader is loaded and
# never used — `bil_bbm_reference` falls back to its cached table.
cp(
    joinpath(@__DIR__, "..", "test", "bil", "harness.jl"),
    joinpath(VALIDATION_DIR, "harness.jl"); force = true,
)

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

# ── Stopgap: citations render as their own struct name ───────────────────────
#
# GUARDED, so one file serves both versions of DocumenterCitations.
#
# 1.5 wraps every expanded citation in a `CitationSiteNode`, whose only purpose
# is to give the citation an HTML anchor the bibliography can link back to. Its
# docstring calls it "transparent in any output format other than HTML", and both
# the LaTeX writer and MDFlatten implement it as "render my children".
#
# DocumenterVitepress 0.3.6 ships a DocumenterCitations extension, but it covers
# `BibliographyNode` only. With no method for `CitationSiteNode` the writer falls
# through to its generic branch, which prints `Markdown.plain(element)` — so every
# citation on the site comes out as the literal text
# `DocumenterCitations.CitationSiteNode("biot1941-cite-1")`.
#
# Its `[compat]` reads `DocumenterCitations = "1.4"`, which Julia expands to
# `^1.4`, so 1.5 is what the docs environment resolves and this method is what
# keeps the bibliography readable. On 1.4 the name does not exist and the block
# is skipped: referring to it unconditionally is an `UndefVarError` raised before
# the first page is built. Remove once DocumenterVitepress covers the node.
if isdefined(DocumenterCitations, :CitationSiteNode)
    function DocumenterVitepress.render(
            io::IO,
            mime::MIME"text/plain",
            node::Documenter.MarkdownAST.Node,
            ::DocumenterCitations.CitationSiteNode,
            page,
            doc;
            kwargs...,
        )
        return DocumenterVitepress.render(
            io, mime, node, node.children, page, doc; kwargs...,
        )
    end
end

# ── Stopgap: heading anchors that contain LaTeX ──────────────────────────────
#
# DocumenterVitepress builds each heading as `## <text> {#<slug>}`, where the slug
# is Documenter's anchor label passed through its own `sanitized_anchor_label` —
# whose comment says "vitepress doesn't like special markdown characters in the id
# slug", but which strips only `[ ] ( ) *`.
#
# A heading such as `### Why a radial balance contains a factor of ``r``` yields
# the slug `Why-a-radial-balance-contains-a-factor-of-r` — harmless — but one
# carrying a LaTeX command does not: VitePress's `{#...}` parser rejects the
# backslash and the braces and treats the whole suffix as *text*, so the heading
# renders with `{#...}` visible, the formula is dropped, and the same garbage
# lands in the "On this page" outline.
#
# `%` is in the set for a harder reason, and it is the one that was measured here.
# Two headings on this site end in a percentage — "approached to 1 %" in
# `bbm_bil` and "where the 3 % went" in `bil_richards` — and a slug ending in a
# bare `%` is not a valid URI escape. VitePress calls `decodeURI` on every link
# target it renders, so the whole build dies with `[vitepress] URI malformed`,
# naming the file but not the line, and nothing is emitted at all.
#
# Stripping those characters from the slug is safe here: nothing links to those
# anchors, and this narrows to headings, leaving docstring anchors — which
# legitimately carry braces, are emitted as raw `<a id=…>` and *are* linked to —
# untouched. Remove once `sanitized_anchor_label` covers these characters.
function DocumenterVitepress.render(
        io::IO,
        mime::MIME"text/plain",
        node::Documenter.MarkdownAST.Node,
        header::Documenter.AnchoredHeader,
        page,
        doc;
        kwargs...,
    )
    anchor = header.anchor
    label = DocumenterVitepress.sanitized_anchor_label(anchor)
    id = replace(replace(label, r"[\\{}%]" => ""), " " => "-")
    heading = first(node.children)
    println(io)
    print(io, "#"^(heading.element.level), " ")
    heading_iob = IOBuffer()
    DocumenterVitepress.render(heading_iob, mime, node, heading.children, page, doc; kwargs...)
    print(io, rstrip(String(take!(heading_iob))))
    print(io, " {#$(id)}")
    if haskey(kwargs, :inventory)
        item = DocumenterVitepress.InventoryItem(
            name = anchor.id,
            domain = "std",
            role = "label",
            dispname = DocumenterVitepress._get_inventory_dispname(
                anchor.id, Documenter.MDFlatten.mdflatten(anchor.node)
            ),
            priority = -1,
            uri = DocumenterVitepress._get_inventory_uri(doc, page, id),
        )
        push!(kwargs[:inventory], item)
    end
    println(io)
    return nothing
end

makedocs(;
    modules = [PoroMechanics],
    authors = "Anthony Soive and Jean-François Barthélémy",
    sitename = "PoroMechanics.jl",
    # The favicon and the logo are picked up automatically from `docs/src/assets`,
    # and the sidebar is derived from `pages`, so neither needs declaring here.
    # `prettyurls`, `collapselevel` and `size_threshold_warn` are HTMLWriter
    # settings with no counterpart: VitePress always writes pretty URLs, the
    # sidebar collapse is decided in `config.mts`, and there is no page-size limit.
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "https://github.com/MicroPoroChemoMechanics/PoroMechanics.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://MicroPoroChemoMechanics.github.io/PoroMechanics.jl",
        description = "Reactive transport and poromechanics of porous media, in Julia",
    ),
    pages = pages,
    plugins = [bib],
    warnonly = [:docs_block, :missing_docs],
)

# DocumenterVitepress writes a real directory per version rather than the symlinks
# Documenter used, so it needs its own `deploydocs`, pointed at the built site.
DocumenterVitepress.deploydocs(;
    repo = "github.com/MicroPoroChemoMechanics/PoroMechanics.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = false,
)
