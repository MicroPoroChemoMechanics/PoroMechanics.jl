# Bil comparison harness — run Bil, read what it wrote, compare it to ours.
#
# Bil (https://github.com/dangla/bil) covers the same physics as this package, from an
# independent implementation in C/C++. Agreeing with it is a real check, and disagreeing
# with it is worth understanding; neither is possible while the reference numbers are
# copied into a `Dict` by hand, which is how `benchmarks/bbm_bil.jl` started.
#
# Nothing here belongs in `src/`: the library must not know that Bil exists. The harness is
# test-side tooling, and it is written so that everything downstream of `read_bil` works
# from a file on disk — so the frozen references in `references/` can stand in for Bil on a
# machine that does not have it.
#
# Three rules the rest of the harness enforces:
#
#  1. **Never run Bil inside `bil-master`.** The `.t*`/`.p*` files committed under `base/`
#     are Bil's own reference outputs. `bil <deck>` overwrites them in place. Every run
#     therefore happens on a copy, and `run_bil` refuses a destination inside the source
#     tree.
#  2. **Read Bil's tabulated curves rather than re-evaluating its expressions.** A deck
#     writes `Curves = wrc2 pc = Range{...} sl = Expressions(1){...}`, and Bil dumps the
#     resulting table next to the deck. Feeding that same table to this package separates a
#     disagreement about the model from a disagreement about how a curve was discretized.
#  3. **Say which Bil produced a number.** `base/BBM` was written by 3.0.0 and
#     `base/Chloricem` by 2.9, while the installed binary is 2.11. `drift_report` measures
#     that gap before anyone blames this package for it.

using Printf: @printf, @sprintf
using ExtendableGrids: simplexgrid

# ── Locating Bil ──────────────────────────────────────────────────────────────

"""
    bil_root() -> String or nothing

Root of the Bil source tree, from `BIL_ROOT` or the default checkout location. Returns
`nothing` when it is not there, so callers can skip rather than fail.
"""
function bil_root()
    root = get(ENV, "BIL_ROOT", joinpath(homedir(), "Documents", "Modelisation", "bil-master"))
    return isdir(joinpath(root, "base")) ? root : nothing
end

"""
    bil_executable() -> String or nothing

Path to the `bil` binary, from `BIL_EXE` or the `PATH`.
"""
function bil_executable()
    exe = get(ENV, "BIL_EXE", "")
    isempty(exe) || return isfile(exe) ? exe : nothing
    found = Sys.which("bil")
    return found === nothing ? nothing : found
end

"""
    bil_available() -> Bool

Whether a run is possible at all: both the source tree (for the decks) and the binary.
"""
bil_available() = bil_root() !== nothing && bil_executable() !== nothing

"""
    bil_version() -> String

Best guess at the version of the *installed* binary, or `"unknown"`. Bil has no
`--version`, so this reads the name of the versioned library installed beside it, and only
then falls back to the source tree's `VERSION.txt`.

The order matters: `VERSION.txt` describes the checkout, which can be updated without
reinstalling. The authoritative answer is the `# Version` header Bil writes into its own
output — use that (`read_bil(...).version`) wherever a run has already happened.
"""
function bil_version()
    for dir in ("/usr/local/lib", "/usr/local/bin", "/opt/homebrew/lib")
        isdir(dir) || continue
        for f in readdir(dir)
            m = match(r"^(?:lib)?bil-([0-9][0-9.]*)-", f)
            m === nothing || return m.captures[1]
        end
    end
    root = bil_root()
    if root !== nothing
        vfile = joinpath(root, "VERSION.txt")
        if isfile(vfile)
            v = strip(read(vfile, String))
            isempty(v) || return v
        end
    end
    return "unknown"
end

# ── Running a case ────────────────────────────────────────────────────────────

"""
    case_dir(relative) -> String

Absolute path of a case directory under Bil's `base/`, e.g. `case_dir("Richards-2d")`.
"""
function case_dir(relative)
    root = bil_root()
    root === nothing && error("Bil source tree not found — set BIL_ROOT")
    dir = joinpath(root, "base", relative)
    isdir(dir) || error("no such Bil case directory: $dir")
    return dir
end

"""
    scratch_root() -> String

Where copies of Bil cases are run. Outside the repository and outside `bil-master`.
"""
function scratch_root()
    return get(ENV, "BIL_SCRATCH", joinpath(tempdir(), "poromechanics-bil"))
end

"""
    run_bil(relative, deck; scratch = nothing, overrides = ()) -> String

Copy the Bil case directory `relative` (under `base/`) to a scratch location, run
`bil <deck>` there, and return the path of the copy. The deck's mesh and curve files are
referenced relatively, which is why the whole directory travels and why the working
directory of the run is the copy.

The copy is what makes this safe: `bil` writes `<deck>.t0`, `<deck>.t1`, … next to the deck
and would otherwise overwrite the outputs committed under `base/`.

`overrides` rewrites `key = value` settings in the deck before running, as pairs:

```julia
run_bil("Richards-2d", "Richards-2d"; overrides = ("Dtmax" => 1.0))
```

This is what makes a convergence study possible — refining Bil's own time step and watching
where its answer goes is the only way to tell a disagreement about the equations from a
disagreement about how they are stepped. Only the first occurrence of each key is rewritten,
and a key that is absent is an error rather than a silent no-op, because a study that
quietly failed to change anything looks exactly like a converged one.
"""
function run_bil(relative, deck; scratch = nothing, overrides = ())
    exe = bil_executable()
    exe === nothing && error("the `bil` executable was not found — set BIL_EXE")

    src = case_dir(relative)
    dest = scratch === nothing ? joinpath(scratch_root(), replace(relative, "/" => "_")) : scratch

    ## Refusing to write inside the source tree is the whole point of this function.
    root = bil_root()
    if root !== nothing && startswith(abspath(dest), abspath(root))
        error("refusing to run Bil inside its own source tree: $dest")
    end

    rm(dest; force = true, recursive = true)
    mkpath(dirname(dest))
    cp(src, dest)
    ## `cp` preserves read-only bits from the source; the run needs to write here.
    chmod(dest, 0o755; recursive = true)

    deck_path = joinpath(dest, deck)
    isfile(deck_path) || error("deck \"$deck\" not found in $src")

    ## The copy carries Bil's shipped outputs with it, and a run only rewrites as many `.tN`
    ## as its own `Dates` block has. Leaving the rest in place means a later read can pick up
    ## a file written years ago by another version — `Plast/Plast1` still ships a `.t5` from
    ## Bil 2.4 that a 2.11 run does not touch. Purge them, so everything read afterwards was
    ## produced by this run.
    for f in readdir(dest)
        occursin(Regex("^" * escape_string(deck) * raw"\.(t|p)\d+$"), f) &&
            rm(joinpath(dest, f))
    end

    for (key, value) in overrides
        text = read(deck_path, String)
        pattern = Regex("(" * escape_string(String(key)) * "\\s*=\\s*)([0-9.eE+-]+)")
        occursin(pattern, text) || error(
            "override \"$key\" matches nothing in $relative/$deck — " *
                "a study that silently changed nothing looks exactly like a converged one"
        )
        write(deck_path, replace(text, pattern => SubstitutionString("\\g<1>$value"); count = 1))
    end

    log = joinpath(dest, deck * ".bil.log")
    open(log, "w") do io
        cmd = pipeline(Cmd(`$exe $deck`; dir = dest); stdout = io, stderr = io)
        try
            run(cmd)
        catch err
            error("bil failed on $relative/$deck ($err)\n$(log_tail(log))")
        end
    end

    ## Bil reports its errors and then exits **zero**. Without this check a run that died at
    ## its first plastic step looks exactly like a run that finished, and the comparison
    ## silently uses whatever partial output is on disk. Both wordings appear and both are
    ## fatal in practice: `base/Plast/Camclay` aborts at t = 8.1 of 10 with a "fatal error"
    ## in `PlasticityCamClay_RM`, and `base/Plast/Plast1` never starts, with a "runtime
    ## error" on an obsolete keyword.
    for line in eachline(log)
        low = lowercase(line)
        if occursin("fatal error", low) || occursin("runtime error", low)
            error(
                "bil reported an error on $relative/$deck but exited 0 — " *
                    "its output is absent or incomplete.\n$(log_tail(log))"
            )
        end
    end
    return dest
end

"""
    log_tail(path, n = 20) -> String

The last `n` lines of a Bil log, labeled, for an error message.
"""
function log_tail(path, n = 20)
    lines = collect(eachline(path))
    tail = join(lines[max(1, end - n + 1):end], "\n")
    return "last lines of $path:\n$tail"
end

# ── Reading Bil output ────────────────────────────────────────────────────────

"""
    BilOutput

One `.tN` (spatial profile at a date) or `.pN` (time history at a point) file.

* `kind` — `:profile` or `:history`.
* `stamp` — the date of a profile, or the coordinates of a history point.
* `views` — view name to the range of columns it occupies, in file order.
* `data` — the raw numeric block, one row per node (profile) or per step (history).

The leading block is a view like any other: `Coordinates` for a profile, `Time` for a
history, so `out["Time"]` and `out["Coordinates"]` work.
"""
struct BilOutput
    path::String
    version::String
    model::String
    kind::Symbol
    stamp::Vector{Float64}
    names::Vector{String}
    views::Dict{String, UnitRange{Int}}
    data::Matrix{Float64}
end

"""
    read_bil(path) -> BilOutput

Parse one of Bil's output files. The format is self-describing: a header naming the views
and the column each one starts at, then a numeric block.

Column widths are taken from the *starting columns* on the label line rather than from
`# Numbers of components per view`, because the leading `Coordinates`/`Time` block is
absent from that list. The list is used as a cross-check.

The label line cannot be split on whitespace — Bil emits view names such as
`n_Friedel's salt(21)`, `total mass flow(52)` and `Ca/Si ratio(59)`.
"""
function read_bil(path)
    isfile(path) || error("no such Bil output: $path")

    version = "unknown"
    model = "unknown"
    kind = :profile
    stamp = Float64[]
    labels = Tuple{String, Int}[]
    ncomponents = Int[]
    first_data = 0

    lines = readlines(path)
    for (i, line) in pairs(lines)
        if !startswith(line, "#")
            isempty(strip(line)) && continue
            first_data = i
            break
        end
        body = strip(lstrip(line, ['#', ' ']))
        if (m = match(r"^Version\s+([^\s,]+)", body)) !== nothing
            version = m.captures[1]
        elseif (m = match(r"^Model\s*=\s*(\S+)", body)) !== nothing
            model = m.captures[1]
        elseif (m = match(r"^Time\s*=\s*(\S+)", body)) !== nothing
            kind = :profile
            stamp = [parse(Float64, m.captures[1])]
        elseif (m = match(r"^Point\s*=\s*(.+)$", body)) !== nothing
            kind = :history
            stamp = [parse(Float64, t) for t in split(strip(m.captures[1]))]
        elseif (m = match(r"^Numbers of components per view\s*=\s*(.+)$", body)) !== nothing
            ncomponents = [parse(Int, t) for t in split(strip(m.captures[1]))]
        elseif occursin(r"\(\d+\)", body)
            ## The label line: `Coordinates(1) Liquid_pore_pressure(4) Displacements(5) …`
            for m in eachmatch(r"([^()]+?)\((\d+)\)", body)
                push!(labels, (strip(m.captures[1]), parse(Int, m.captures[2])))
            end
        end
    end

    isempty(labels) && error("no view labels found in $path — is this a Bil output file?")
    first_data == 0 && error("no data block in $path")

    data = parse_data_block(view(lines, first_data:length(lines)), path)
    ncols = size(data, 2)

    ## Width of a view = distance to the next one; the last runs to the end of the row.
    names = String[first(l) for l in labels]
    starts = Int[last(l) for l in labels]
    views = Dict{String, UnitRange{Int}}()
    for (k, name) in pairs(names)
        stop = k < length(starts) ? starts[k + 1] - 1 : ncols
        starts[k] <= stop <= ncols || error(
            "inconsistent header in $path: view \"$name\" spans $(starts[k]):$stop of $ncols columns"
        )
        views[name] = starts[k]:stop
    end

    ## Cross-check against the declared component counts. They cover every view but the
    ## leading Coordinates/Time block, so they line up with the tail of `names`.
    if !isempty(ncomponents) && length(ncomponents) == length(names) - 1
        for (k, n) in pairs(ncomponents)
            got = length(views[names[k + 1]])
            got == n || error(
                "header disagrees with itself in $path: view \"$(names[k + 1])\" " *
                    "declares $n components but spans $got columns"
            )
        end
    end

    return BilOutput(path, version, model, kind, stamp, names, views, data)
end

"""
    parse_data_block(lines, path) -> Matrix{Float64}

The numeric tail of a Bil output. Non-finite entries (`nan`, `-nan`, `inf`) are kept as
such rather than rejected: a diverged run is a result too, and hiding it behind a parse
error would only make it harder to see.
"""
function parse_data_block(lines, path)
    rows = Vector{Vector{Float64}}()
    for line in lines
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        row = Float64[]
        for tok in split(s)
            v = tryparse(Float64, tok)
            push!(row, v === nothing ? NaN : v)
        end
        isempty(row) || push!(rows, row)
    end
    isempty(rows) && error("empty data block in $path")

    widths = unique(length.(rows))
    if length(widths) > 1
        ## Not corruption: a deck with several `Material` blocks of *different models* writes
        ## each element with its own view set into one file, and the header describes only
        ## the first. `base/Plast/Plast0` declares `Model = Plast` and `Model = Elast`, and
        ## its rows are 29 and 18 columns wide accordingly. There is no honest way to read
        ## such a file against a single header, so refuse it by name rather than let a
        ## comparison silently align the wrong columns.
        listed = join(sort(widths), " and ")
        error(
            "$path mixes rows of $listed columns.\n" *
                "That is what Bil writes when a deck declares several Material blocks with " *
                "different models: each element carries its own views, and the header " *
                "describes only the first. Compare such a case per material, from a deck " *
                "with one model in it."
        )
    end
    width = first(widths)
    return reduce(vcat, (permutedims(r) for r in rows))
end

"""
    Base.getindex(out::BilOutput, name) -> AbstractMatrix

The columns of one view, as a matrix (one column per component). Use `column` for the
common single-component case.
"""
function Base.getindex(out::BilOutput, name::AbstractString)
    if !haskey(out.views, name)
        available = join(out.names, ", ")
        error("no view \"$name\" in $(out.path)\navailable: $available")
    end
    return @view out.data[:, out.views[name]]
end

"""
    column(out, name, k = 1) -> Vector{Float64}

Component `k` of a view, as a plain vector.
"""
column(out::BilOutput, name::AbstractString, k::Int = 1) = vec(out[name][:, k])

"""
    coordinates(out) -> Matrix{Float64}

Node coordinates of a profile. Bil always writes three columns, whatever the dimension.
"""
coordinates(out::BilOutput) = out["Coordinates"]

"""
    times(out) -> Vector{Float64}

Sample times of a history file.
"""
times(out::BilOutput) = column(out, "Time")

"""
    BIL_OUTPUT_DIGITS

Significant digits Bil writes into its output files, and therefore the floor on every
tolerance in this harness.

Bil prints with `%e` and six decimals — `-8.486096e+02`. Measured on `Richards-2d`, the
coordinates it reports differ from the mesh's own by up to 5e-8, which is exactly that
truncation and nothing else. **No comparison against a Bil output file can be tighter than
about 1e-7 relative**, no matter how well the two codes agree. A case tolerance below that
is not strictness, it is a test that fails on printing.
"""
const BIL_OUTPUT_DIGITS = 7

"""
    nodal_on(out, cellnodes, name, k = 1) -> Vector{Float64}

A nodal field of a profile, mapped onto the nodes of the mesh Bil used, by element order.

Bil writes its rows element by element: row `3(e-1)+j` of a triangular mesh is local node
`j` of element `e`, hence global node `cellnodes[j, e]`. Using that ordering pairs the two
codes by *index*, with no coordinate matching and no tolerance — which matters, because
Bil's printed coordinates are only good to `BIL_OUTPUT_DIGITS` and a coordinate match would
have to invent a tolerance to work at all.

The values are written into a vector of length `maximum(cellnodes)`; a node shared by
several elements is written several times with the same number, which is what makes this
valid for a continuous unknown and invalid for a flux. Use [`nodal`](@ref) when the mesh is
not available — a Bil deck can define its 1D mesh inline, with no `.msh` to read.
"""
function nodal_on(out::BilOutput, cellnodes, name::AbstractString, k::Int = 1)
    out.kind === :profile || error("`nodal_on` applies to a .tN profile, got $(out.kind)")
    values = column(out, name, k)
    npe, ncells = size(cellnodes)
    expected = npe * ncells
    length(values) == expected || error(
        "$(basename(out.path)) has $(length(values)) rows but the mesh has $ncells cells " *
            "of $npe nodes ($expected rows expected) — is this the mesh the deck used?"
    )
    nodal_values = Vector{Float64}(undef, maximum(cellnodes))
    for e in 1:ncells, j in 1:npe
        nodal_values[cellnodes[j, e]] = values[npe * (e - 1) + j]
    end
    return nodal_values
end

"""
    nodal(out, name, k = 1; strict = true, digits = 12) -> (points, values)

A nodal field of a profile, with Bil's element-wise duplication removed.

Bil writes **one row per element node**, not one per node: `Richards-2d.t3` has 7932 rows
for a mesh of 1433 nodes and 2644 triangles. A continuous unknown repeats identically on
every element that touches a node, so it collapses exactly — the liquid pressure of that
file is single-valued to the last bit.

An element field does not collapse, and must not be silently averaged. A flux is
discontinuous across a material interface by construction, and the same file carries a
spread of 3e-2 on `flow` between elements meeting at an interface node. `strict = true`
therefore refuses rather than hides it; pass `strict = false` to take the mean anyway, when
that is genuinely what is wanted.

Returns the unique node coordinates (one row each) and the corresponding values, ordered by
first appearance so the pairing with `coordinates(out)` is reproducible.
"""
function nodal(out::BilOutput, name::AbstractString, k::Int = 1; strict = true, digits = 12)
    out.kind === :profile || error("`nodal` applies to a .tN profile, got $(out.kind)")
    xyz = coordinates(out)
    values = column(out, name, k)

    order = Vector{NTuple{3, Float64}}()
    seen = Dict{NTuple{3, Float64}, Vector{Float64}}()
    for i in axes(xyz, 1)
        key = (round(xyz[i, 1]; digits), round(xyz[i, 2]; digits), round(xyz[i, 3]; digits))
        bucket = get(seen, key, nothing)
        if bucket === nothing
            seen[key] = [values[i]]
            push!(order, key)
        else
            push!(bucket, values[i])
        end
    end

    if strict
        worst, where_ = 0.0, first(order)
        for key in order
            v = seen[key]
            spread = maximum(v) - minimum(v)
            spread > worst && ((worst, where_) = (spread, key))
        end
        scale = max(maximum(abs, values), eps())
        if worst > 1.0e-10 * scale
            error(
                "view \"$name\" is not single-valued per node (spread $worst at $where_) — " *
                    "it is element data, such as a flux discontinuous across a material " *
                    "interface. Compare it element-wise, or pass strict = false to average."
            )
        end
    end

    points = Matrix{Float64}(undef, length(order), 3)
    vals = Vector{Float64}(undef, length(order))
    for (i, key) in pairs(order)
        points[i, 1], points[i, 2], points[i, 3] = key
        v = seen[key]
        vals[i] = sum(v) / length(v)
    end
    return points, vals
end

Base.show(io::IO, out::BilOutput) = print(
    io,
    "BilOutput(", basename(out.path), ", model=", out.model, ", ", out.kind,
    ", ", size(out.data, 1), "×", size(out.data, 2), ", ", length(out.names), " views)",
)

# ── Reading Bil's tabulated curves ────────────────────────────────────────────

"""
    read_bil_curve(path) -> (labels, table)

A curve file written by Bil — `wrc2`, `krc2`, `lc`, or a hand-supplied one such as
`billes`. These are the tables the solver actually interpolated, so reading them is how a
comparison avoids blaming the model for a difference in curve discretization.

Bil-generated files carry `# Labels: pc(1) sl(2) kl(3)`; hand-written ones have no header
at all, and then the labels come back as `x1`, `x2`, …
"""
function read_bil_curve(path)
    isfile(path) || error("no such curve file: $path")
    lines = readlines(path)
    labels = String[]
    data_start = 1
    for (i, line) in pairs(lines)
        if startswith(line, "#")
            body = strip(lstrip(line, ['#', ' ']))
            if startswith(body, "Labels:")
                for m in eachmatch(r"([^()\s]+)\((\d+)\)", body)
                    push!(labels, m.captures[1])
                end
            end
            data_start = i + 1
        else
            isempty(strip(line)) || break
        end
    end
    table = parse_data_block(view(lines, data_start:length(lines)), path)
    isempty(labels) && (labels = ["x$(k)" for k in 1:size(table, 2)])
    return labels, table
end

# ── Reading Bil's meshes ──────────────────────────────────────────────────────

"""
    read_gmsh_simplexgrid(path) -> (; grid, cell_tag, bface_tag, coord)

Build an `ExtendableGrids.simplexgrid` from the Gmsh 2.2 ASCII mesh a Bil deck points at.

Using Bil's own mesh, rather than remeshing the same geometry, is what makes the comparison
exact: node `i` here is node `i` there, so fields line up without interpolation and no error
of ours is hidden behind a difference of discretization.

**Regions come from the elementary tag, not the physical one.** That is Bil's convention:
`columncomposite.geo` puts all four sides in `Physical Line(1)` while numbering them
`Line(11) … Line(14)`, and the deck imposes its boundary condition on `Region = 11`. Reading
the physical tag would collapse the four sides into one and silently impose the pressure on
the whole boundary.

`cell_tag` and `bface_tag` map a Gmsh elementary tag to the compact region index used in the
grid, because `VoronoiFVM` wants regions numbered from 1 and Bil's tags are 100, 101, 11…
Address a region by Bil's number through the map:

```julia
mesh = read_gmsh_simplexgrid(path)
k = zeros(length(mesh.cell_tag))
k[mesh.cell_tag[100]] = 8.9e-12    # outer zone
k[mesh.cell_tag[101]] = 8.9e-13    # inclusion
```

The dimension is read from the mesh rather than assumed: the highest simplex present becomes
the cell, and the next one down becomes the boundary face. Tetrahedra (Gmsh type 4) give a
3D grid bounded by triangles, triangles (type 2) a 2D grid bounded by lines (type 1), and
lines alone a 1D grid bounded by points (type 15). Nothing else is supported, because
nothing else appears in a Bil deck — Bil also uses points as output locations, which is
harmless: a `Points` block adds boundary vertices that a 1D grid wants anyway and that a 2D
or 3D grid ignores.
"""
function read_gmsh_simplexgrid(path)
    isfile(path) || error("no such mesh: $path")
    lines = readlines(path)

    version_ok = false
    coord3 = Matrix{Float64}(undef, 3, 0)

    ## Gmsh element type → nodes per element, for the simplices Bil uses.
    simplex_nodes = Dict(15 => 1, 1 => 2, 2 => 3, 4 => 4)
    by_type = Dict{Int, Vector{Vector{Int}}}()
    tags_by_type = Dict{Int, Vector{Int}}()

    i = 1
    while i <= length(lines)
        section = strip(lines[i])
        if section == "\$MeshFormat"
            fmt = split(strip(lines[i + 1]))
            startswith(first(fmt), "2.") || error(
                "$path is Gmsh format $(first(fmt)); only the 2.x ASCII format is read here"
            )
            length(fmt) >= 2 && fmt[2] == "0" || error("$path is binary Gmsh; ASCII expected")
            version_ok = true
            i += 2
        elseif section == "\$Nodes"
            n = parse(Int, strip(lines[i + 1]))
            coord3 = Matrix{Float64}(undef, 3, n)
            for k in 1:n
                t = split(strip(lines[i + 1 + k]))
                id = parse(Int, t[1])
                id == k || error("$path: node ids are not 1..n in order (got $id at row $k)")
                coord3[1, k] = parse(Float64, t[2])
                coord3[2, k] = parse(Float64, t[3])
                coord3[3, k] = parse(Float64, t[4])
            end
            i += n + 2
        elseif section == "\$Elements"
            n = parse(Int, strip(lines[i + 1]))
            for k in 1:n
                t = split(strip(lines[i + 1 + k]))
                etype = parse(Int, t[2])
                ntags = parse(Int, t[3])
                ## Gmsh 2.2 tags are `physical elementary …`; Bil reads the elementary one.
                elementary = ntags >= 2 ? parse(Int, t[5]) : parse(Int, t[4])
                haskey(simplex_nodes, etype) || continue
                nodes = [parse(Int, s) for s in t[(4 + ntags):end]]
                length(nodes) == simplex_nodes[etype] || error(
                    "$path: element type $etype should have $(simplex_nodes[etype]) nodes, got $(length(nodes))"
                )
                push!(get!(by_type, etype, Vector{Vector{Int}}()), nodes)
                push!(get!(tags_by_type, etype, Int[]), elementary)
            end
            i += n + 2
        else
            i += 1
        end
    end

    version_ok || error("$path has no \$MeshFormat section")

    ## The highest simplex present is the cell; the one below it is the boundary face.
    cell_type = maximum(t for t in keys(by_type) if t in (1, 2, 4); init = 0)
    cell_type == 0 && error("$path contains no lines, triangles or tetrahedra")
    face_type = Dict(4 => 2, 2 => 1, 1 => 15)[cell_type]
    dim = simplex_nodes[cell_type] - 1

    "Pack a list of node lists into a `nodes × count` matrix."
    pack(list, npe) = begin
        m = Matrix{Int32}(undef, npe, length(list))
        for (k, nodes) in pairs(list), j in 1:npe
            m[j, k] = nodes[j]
        end
        m
    end

    ## Compact the tags, keeping the map back to Bil's numbering.
    compact(tags) = Dict(t => k for (k, t) in pairs(sort(unique(tags))))

    cells = by_type[cell_type]
    cell_tags = tags_by_type[cell_type]
    cell_tag = compact(cell_tags)
    cellnodes = pack(cells, simplex_nodes[cell_type])
    cellregions = Int32[cell_tag[t] for t in cell_tags]

    faces = get(by_type, face_type, Vector{Vector{Int}}())
    face_tags = get(tags_by_type, face_type, Int[])
    bface_tag = compact(face_tags)
    bfacenodes = pack(faces, simplex_nodes[face_type])
    bfaceregions = Int32[bface_tag[t] for t in face_tags]

    coord = coord3[1:dim, :]
    grid = simplexgrid(coord, cellnodes, cellregions, bfacenodes, bfaceregions)
    return (; grid, cell_tag, bface_tag, coord, dim)
end

# ── Comparing ─────────────────────────────────────────────────────────────────

"""
    interpolate_at(x, y, xq) -> Vector

Piecewise-linear interpolation of `y(x)` at `xq`, clamped outside the range. `x` need not
be sorted — Bil writes nodes in mesh order, which is not always monotone.
"""
function interpolate_at(x, y, xq)
    p = sortperm(x)
    xs, ys = x[p], y[p]
    out = similar(ys, length(xq))
    for (k, q) in pairs(xq)
        if q <= first(xs)
            out[k] = first(ys)
        elseif q >= last(xs)
            out[k] = last(ys)
        else
            j = searchsortedlast(xs, q)
            t = (q - xs[j]) / (xs[j + 1] - xs[j])
            out[k] = ys[j] + t * (ys[j + 1] - ys[j])
        end
    end
    return out
end

"""
    FieldComparison

One field compared between two codes: the relative L2 deviation, the largest single
deviation and where it sits.
"""
struct FieldComparison
    field::String
    n::Int
    l2::Float64
    worst::Float64
    at::Int
    reference::Float64
    candidate::Float64
end

"""
    compare_fields(reference, candidate) -> Vector{FieldComparison}

Both arguments map a field name to a vector of equal length — typically Bil's values and
ours, already sampled at the same points. The deviation is relative to the norm of the
reference, with an absolute fallback when the reference is zero.
"""
function compare_fields(reference::AbstractDict, candidate::AbstractDict)
    out = FieldComparison[]
    for field in sort(collect(keys(reference)))
        haskey(candidate, field) || continue
        r = collect(Float64, reference[field])
        c = collect(Float64, candidate[field])
        length(r) == length(c) || error(
            "field \"$field\": reference has $(length(r)) values, candidate $(length(c))"
        )
        d = abs.(c .- r)
        scale = max(sqrt(sum(abs2, r)), eps())
        i = isempty(d) ? 0 : argmax(d)
        push!(
            out,
            FieldComparison(
                field, length(r), sqrt(sum(abs2, d)) / scale,
                isempty(d) ? 0.0 : d[i], i,
                i == 0 ? 0.0 : r[i], i == 0 ? 0.0 : c[i],
            ),
        )
    end
    return out
end

"""
    report(comparisons; title, io = stdout)

Print a comparison as a table. Deliberately plain text: these numbers end up in a
`benchmarks/` page, where the reader needs to see them next to the explanation.
"""
function report(comparisons; title = "", io = stdout)
    isempty(title) || println(io, title)
    @printf(io, "  %-28s %6s  %12s  %12s\n", "field", "n", "rel. L2", "worst abs.")
    for c in comparisons
        @printf(io, "  %-28s %6d  %12.4e  %12.4e\n", c.field, c.n, c.l2, c.worst)
    end
    return nothing
end

# ── Drift between Bil versions ────────────────────────────────────────────────

"""
    drift_report(relative, deck; date = nothing) -> Vector{FieldComparison}

Compare the output Bil 2.11 produces *now* against the output committed under `base/`.

Produce this before concluding anything about a disagreement with this package: the files
in `base/` were written by earlier versions (2.9 for `Chloricem`, 3.0.0 for `BBM`), and a
difference that already exists between Bil and its own shipped reference is not this
package's to explain.

`date` selects the `.tN` file; the default is the last one present.
"""
function drift_report(relative, deck; date = nothing)
    shipped_dir = case_dir(relative)
    n = date === nothing ? last_date(shipped_dir, deck) : date
    n === nothing && error("no .tN output shipped for $relative/$deck")

    fresh_dir = run_bil(relative, deck)
    shipped = read_bil(joinpath(shipped_dir, "$deck.t$n"))
    fresh = read_bil(joinpath(fresh_dir, "$deck.t$n"))

    common = intersect(Set(shipped.names), Set(fresh.names))
    ref = Dict{String, Vector{Float64}}()
    cand = Dict{String, Vector{Float64}}()
    for name in common
        a, b = shipped[name], fresh[name]
        size(a) == size(b) || continue
        ref[name] = vec(a)
        cand[name] = vec(b)
    end

    ## Both versions are read from the files themselves rather than guessed: `fresh.version`
    ## is by definition what the installed binary stamps on its own output.
    println("Bil drift on $relative/$deck at date $n")
    println("  shipped under base/ : version ", shipped.version)
    println("  installed binary    : version ", fresh.version)
    if !isempty(setdiff(Set(shipped.names), common))
        println("  views only in the shipped file: ", join(sort(collect(setdiff(Set(shipped.names), common))), ", "))
    end
    comparisons = compare_fields(ref, cand)
    report(comparisons)
    return comparisons
end

"""
    last_date(dir, deck) -> Int or nothing

Highest `N` for which `<deck>.tN` exists in `dir`.
"""
function last_date(dir, deck)
    best = nothing
    for f in readdir(dir)
        m = match(Regex("^" * escape_string(deck) * raw"\.t(\d+)$"), f)
        m === nothing && continue
        n = parse(Int, m.captures[1])
        best = best === nothing ? n : max(best, n)
    end
    return best
end
