# Building these docs

The site is emitted as Markdown by Documenter and rendered by VitePress
(DocumenterVitepress), so the build no longer produces a browsable
`docs/build/index.html`.

From the repository root:

```
julia +1.12 --project=docs docs/make.jl
```

That runs every `@example` block on the site, so it is not quick. The result
lands in `docs/build/1`, and its links carry no `.html` suffix (`cleanUrls`), so
a plain static server answers 404 to all of them. Serve it with VitePress
instead, from `docs/`:

```
npm run docs:preview
```

`npm` comes from the `NodeJS_20_jll` artifact DocumenterVitepress pulls in, so
nothing has to be installed system-wide; if it is not on `PATH`:

```
PATH="$(dirname $(ls ~/.julia/artifacts/*/bin/npm | head -1)):$PATH" npm run docs:preview
```

`npm run docs:dev` does the same with hot reload while editing.

## What is tracked here, and what is substituted

DocumenterVitepress ships default templates for the VitePress config, the two
Markdown plugins and the four theme files, and substitutes in each one it does
not find under `docs/src/.vitepress/`. Only the files that actually differ are
committed, because a committed file is a frozen copy that stops tracking
upstream:

| File | Why it is here |
| :--- | :--- |
| `src/.vitepress/config.mts` | collapses the sidebar groups, folds the navbar |
| `src/.vitepress/mathjax-plugin.ts` | loads the `mhchem` TeX extension |
| `src/.vitepress/theme/index.ts` | pops the bibliography entry when a citation is hovered |
| `src/.vitepress/theme/overrides.css` | the house style — the port of the old `assets/custom.css` |
| `package.json` | adds the mhchem font extension to the default dependencies |

Formulas are typeset at build time, in Node, into static SVG: a reader's browser
fetches no MathJax bundle. New TeX extensions therefore go in
`mathjax-plugin.ts`, never in a Documenter `mathengine`.

## Mathematical notation

Use `$...$` for inline math and fenced `math` blocks for displayed equations in all
Markdown pages and in the prose comments of Literate scripts. Both are recognized by
Documenter and VS Code's built-in Markdown math preview. Double-backtick inline math
is specific to Julia Markdown and appears as code in VS Code. Keep single backticks
for code names, such as `phi`, and use math notation for the physical quantity.
Start explanatory paragraphs with prose before an inline formula: Julia Markdown can
interpret a paragraph-leading dollar formula as a displayed equation.

For generated example, demo, and validation pages, change the corresponding script in
`examples/`, `demos/`, or `benchmarks/` and regenerate with `docs/make.jl`.

## Theory pages and figures

The introductory course is maintained directly in `docs/src/theory/`; it is not generated
by Literate. Keep its symbols, units, and implementation notes consistent with the API.
The original explanatory figures are SVG files in `docs/src/assets/theory/`. To regenerate
them with Python, NumPy, and Matplotlib:

```sh
python3 docs/src/assets/theory/draw_figures.py
```

Pass `--preview-dir /tmp/poromechanics-theory` to also produce PNGs for visual review.
The plot parameters are illustrative; the consolidation curves use the analytical
Terzaghi series, not solver output. Figure generation is separate from the documentation
build and adds no Python dependency to the Julia package.
