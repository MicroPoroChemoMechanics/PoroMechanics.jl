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
