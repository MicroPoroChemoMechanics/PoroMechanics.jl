# Bil comparison case registry.
#
# One entry per Bil reference case that this package has a counterpart for. Each entry knows
# three things:
#
#  * how to get Bil's answer — which deck, which output file, which view;
#  * how to get ours — a function returning the same quantities at the same points;
#  * how close they are entitled to be.
#
# The Bil half is run once and frozen into `references/<name>.txt` by `generate.jl`, so the
# test suite compares against a file and never needs Bil installed. `test/bil.jl` is the
# consumer.
#
# ## The tolerances are not 1e-10, and that is the point
#
# `test/regression` compares this package against *itself*, where anything above round-off
# is a real change. Here two independent codes are compared, and three separate floors sit
# under any agreement:
#
#  1. **Printing.** Bil writes `%e` with six decimals, so nothing in its files is known to
#     better than about 1e-7 relative (`BIL_OUTPUT_DIGITS`).
#  2. **The tangent.** Bil builds its Jacobian by finite differences with a perturbation set
#     by the deck's `Objective Variations`. Its converged answer therefore depends on a
#     tuning parameter that has no counterpart here.
#  3. **The time stepping.** The two codes take different steps, and for a first-order
#     scheme the difference shows up directly in the answer. That is the whole subject of
#     `benchmarks/bbm_bil.jl`.
#
# A tolerance here is a *measured* value with a stated reason, not an aspiration. Tightening
# one without understanding why it can be tightened defeats the purpose.

include("harness.jl")

using Dates: Dates
using PoroMechanics
using VoronoiFVM: VoronoiFVM, unknowns, solve
using ExtendableGrids: CellNodes, num_nodes

const BIL_REFERENCE_DIR = joinpath(@__DIR__, "references")

"""
    BilCase(name, relative, deck, bil, ours, rtol, why)

* `name` — test label and reference file stem.
* `relative`, `deck` — the case directory under Bil's `base/`, and the deck inside it.
* `bil()` — returns `(; values, version)`: the reference vector, and the Bil version that
  produced it. This runs Bil, so `generate.jl` needs the binary and not only the source
  tree. It used to read the outputs shipped under `base/`, and that turned out to be the
  wrong reference: those were produced with the deck's own coarse time step and carry a
  visible discretisation error (see `richards_2d` below). Freezing them meant testing that
  we stay a fixed distance from a number known to be wrong.
* `ours()` — returns the same quantities, computed here.
* `rtol` — relative L2 tolerance, with `why` recording where the number came from.
"""
struct BilCase
    name::String
    relative::String
    deck::String
    bil::Function
    ours::Function
    rtol::Float64
    why::String
end

# ── Richards 2D — drainage of a composite column ──────────────────────────────
#
# `base/Richards-2d`, the case `examples/richards_2d/run.jl` already reproduces: a 0.02 ×
# 0.20 m column of glass beads with a less permeable inclusion in the middle
# (8.9e-12 against 8.9e-13 m²), drained from the bottom.
#
# Three things make this the right case to prove the harness on, because each removes one
# possible source of disagreement:
#
#  * **the same mesh** — Bil's own `columncomposite.msh` is read through
#    `read_gmsh_simplexgrid`, so node `i` is node `i` and nothing is interpolated;
#  * **the same retention curve** — `billes` is the table Bil interpolated, read back
#    through `read_bil_curve` instead of refitting a Van Genuchten to it;
#  * **the same initial state** — hydrostatic, and it matches Bil's `t0` file to 1.4e-7,
#    which is exactly the printing floor.
#
# What is left is the schemes, and they do differ — the difference is measured and explained
# in `benchmarks/bil_richards.jl`. In short: Bil evaluates the flux with the permeability of
# the *previous* step (`double k_l = val_n.Permeability_liquid;` in `Richards.cpp`), an
# explicit lag that costs it first-order accuracy in time, while the mobility here is taken
# at the current iterate. The reference is therefore taken from a Bil run with its own step
# refined to 1 s, not from the file shipped with the deck.

"Dates written by the deck's `Dates` block, in file order: `Richards-2d.tN` is `DATES[N+1]`."
const RICHARDS_2D_DATES = collect(0.0:200.0:3000.0)

"Output indices compared. A subset, so the reference file stays reviewable in a diff."
const RICHARDS_2D_PROBES = (0, 1, 2, 3, 5, 10, 15)

"Every 16th node of the mesh, so a 1433-node profile reduces to ~90 values per date."
const RICHARDS_2D_STRIDE = 16

"`Dtmax` forced on Bil when producing the reference. See `benchmarks/bil_richards.jl`."
const RICHARDS_2D_BIL_DT = 1.0

"""
    richards_2d_mesh() -> NamedTuple

Bil's own mesh for the case, with the region maps. Read once and cached: the file is parsed
by both halves of the comparison and by the generator.
"""
function richards_2d_mesh()
    dir = case_dir("Richards-2d")
    return read_gmsh_simplexgrid(joinpath(dir, "columncomposite.msh"))
end

"""
    richards_2d_bil() -> (; values, version)

Bil's liquid pressure at the probe dates, subsampled by `RICHARDS_2D_STRIDE`, together with
the version stamped on the files it was read from.
"""
function richards_2d_bil()
    ## Bil's own time step refined from the deck's `Dtmax = 1000` down to 1 s. At the deck's
    ## setting Bil's answer sits 2.8e-2 away from the limit both codes converge to; at 1 s it
    ## is 7.2e-4 away. Refining further buys little and costs a lot — 102 s of Bil for this
    ## one case already.
    dir = run_bil("Richards-2d", "Richards-2d"; overrides = ("Dtmax" => RICHARDS_2D_BIL_DT,))
    cellnodes = richards_2d_mesh().grid[CellNodes]
    values = Float64[]
    version = "unknown"
    for n in RICHARDS_2D_PROBES
        out = read_bil(joinpath(dir, "Richards-2d.t$n"))
        version = out.version
        append!(values, nodal_on(out, cellnodes, "pressure")[1:RICHARDS_2D_STRIDE:end])
    end
    return (; values, version)
end

"""
    richards_2d_ours(; verbose = false) -> Vector{Float64}

The same profile computed here, on Bil's mesh and with Bil's retention table.

The solver control is not arbitrary. `Δu_opt = 100` Pa plays the role of Bil's
`Objective Variations p_l = 1.e3` — the scale at which the step controller considers the
solution to have moved — and `Δt_max = 10` s resolves the 360 s drainage ramp imposed at the
bottom. Left at the `VoronoiFVM` defaults (`Δu_opt = 0.1`) the controller stalls
immediately, because the unknown is measured in kilopascals.
"""
function richards_2d_ours(; verbose = false)
    dir = case_dir("Richards-2d")
    mesh = richards_2d_mesh()

    ## The curve Bil actually interpolated, not a refit of it.
    _, table = read_bil_curve(joinpath(dir, "billes"))
    pc_tab, sl_tab, krl_tab = table[:, 1], table[:, 2], table[:, 3]

    retention = Tabulated(pc_tab, sl_tab)
    rel_perm = TabulatedKrl(pc_tab, krl_tab)

    k_int = zeros(length(mesh.cell_tag))
    k_int[mesh.cell_tag[100]] = 8.9e-12      # outer zone
    k_int[mesh.cell_tag[101]] = 8.9e-13      # inclusion
    bottom = mesh.bface_tag[11]              # Line(11) = {1,2}, the base of the column

    model = RichardsModel(;
        phi = 0.38, rho_l = 1.0e3, k_int = k_int, mu_l = 1.0e-3,
        p_g = 0.0, gravity = -9.81, gravity_axis = 2,
        retention = retention, rel_perm = rel_perm,
    )

    ## `Field 1` of the deck is the hydrostatic profile p_l = -9810 (y - 0.2); the boundary
    ## condition is that field at the base, scaled by `Function 1`, which ramps from 1 at
    ## t = 0 to 0 at t = 360 s and stays there.
    hydrostatic(y) = -9810.0 * (y - 0.2)
    ramp(t) = t <= 360.0 ? 1.0 - t / 360.0 : 0.0
    base_pressure = hydrostatic(0.0)

    sys = VoronoiFVM.System(
        mesh.grid; species = [1],
        storage = (f, u, node, data) -> PoroMechanics.storage!(f, u, node, model, data),
        flux = (f, u, edge, data) -> PoroMechanics.flux!(f, u, edge, model, data),
        bcondition = function (f, u, bnode, data)
            VoronoiFVM.boundary_dirichlet!(
                f, u, bnode;
                species = 1, region = bottom, value = base_pressure * ramp(bnode.time),
            )
        end,
    )

    inival = unknowns(sys)
    for n in 1:num_nodes(mesh.grid)
        inival[1, n] = hydrostatic(mesh.coord[2, n])
    end

    tsol = solve(
        sys; inival, times = (0.0, last(RICHARDS_2D_DATES)),
        control = VoronoiFVM.SolverControl(;
            Δt = 1.0, Δt_min = 1.0e-8, Δt_max = 10.0, Δu_opt = 100.0,
            damp_initial = 0.5, damp_growth = 1.5, reltol = 1.0e-10,
            verbose = verbose ? "en" : "",
        ),
    )

    values = Float64[]
    for n in RICHARDS_2D_PROBES
        append!(values, tsol(RICHARDS_2D_DATES[n + 1])[1, 1:RICHARDS_2D_STRIDE:end])
    end
    return values
end

# ── The registry ──────────────────────────────────────────────────────────────

const BIL_CASES = [
    BilCase(
        "richards_2d",
        "Richards-2d", "Richards-2d",
        richards_2d_bil,
        richards_2d_ours,
        1.0e-2,
        """
        Measured 5.6e-3 at t = 600 s, the worst date, against Bil refined to Dtmax = 1 s.
        Almost all of that is *our* time step: the test solves with Δt_max = 10 s, which sits
        4.9e-3 from our own converged answer, and Bil's residual error at 1 s is 7.2e-4. The
        physics agreement is therefore better than 1e-3, between a finite element code and a
        finite volume one.

        Against the file shipped with the deck the same solve is 3.4e-2 away, and that gap is
        Bil's time discretisation, not a disagreement: 2.8e-2 of it separates the shipped
        output from Bil's own refined answer. `benchmarks/bil_richards.jl` has the two-sided
        convergence study.
        """,
    ),
]

# ── Reference file I/O ────────────────────────────────────────────────────────
# Same plain-text-at-full-precision convention as `test/regression/references`, for the same
# reason: a reference that cannot be read in a diff cannot be reviewed.

bil_reference_path(name) = joinpath(BIL_REFERENCE_DIR, name * ".txt")

function write_bil_reference(case, values, version)
    mkpath(BIL_REFERENCE_DIR)
    open(bil_reference_path(case.name), "w") do io
        println(io, "# Bil reference — ", case.name)
        println(io, "# Case: base/", case.relative, ", deck ", case.deck)
        println(io, "# Written by Bil ", version, " (the version stamped on the output read),")
        println(io, "# frozen ", Dates.format(Dates.now(), "yyyy-mm-dd"), " against installed binary ", bil_version())
        println(io, "# Regenerate with: julia --project test/bil/generate.jl ", case.name)
        println(io, "# Tolerance: ", case.rtol)
        println(io, "# ", length(values), " values, printed with %.17g")
        for v in values
            @printf(io, "%.17g\n", v)
        end
    end
    return bil_reference_path(case.name)
end

function read_bil_reference(name)
    path = bil_reference_path(name)
    isfile(path) || error(
        "missing Bil reference for \"$name\" at $path — " *
            "run `julia --project test/bil/generate.jl` on a machine that has Bil"
    )
    return [
        parse(Float64, l) for l in eachline(path)
            if !startswith(l, "#") && !isempty(strip(l))
    ]
end
