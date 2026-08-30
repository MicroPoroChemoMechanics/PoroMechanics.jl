"""
Water retention curves — the relation between capillary pressure and liquid saturation.

Every curve here is a struct parameterized by the *type* of its coefficients, not by
`Float64`. That is deliberate: it lets a `ForwardDiff.Dual` enter a coefficient, so a
result can be differentiated with respect to the material parameters themselves and not
only with respect to the unknowns. A curve declared `a::Float64` would make that
impossible.
"""

"""
    AbstractRetention

Supertype of the water retention curves. A concrete curve implements

- [`saturation`](@ref)`(curve, pc)` — liquid saturation ``S_l`` at capillary pressure `pc`
- [`dsaturation_dpc`](@ref)`(curve, pc)` — its derivative ``\\partial S_l / \\partial p_c``
"""
abstract type AbstractRetention end

"""
    saturation(curve, pc)

Liquid saturation ``S_l`` at capillary pressure `pc` [Pa]. Returns 1 for `pc ≤ 0`, the
saturated branch.
"""
function saturation end

"""
    dsaturation_dpc(curve, pc)

Derivative ``\\partial S_l / \\partial p_c`` [Pa⁻¹] of the retention curve. Zero on the
saturated branch.
"""
function dsaturation_dpc end

# ── Van Genuchten ─────────────────────────────────────────────────────────────

"""
    VanGenuchten(a, n, m)
    VanGenuchten(a, m)

Van Genuchten retention curve [vangenuchten1980](@cite):

```math
S_l(p_c) = \\left(1 + (p_c / a)^n\\right)^{-m}, \\qquad p_c > 0
```

`a` is the air-entry pressure [Pa]; `n` and `m` are dimensionless exponents.

The two-argument form applies the classical constraint ``n = 1/(1-m)``. The
three-argument form leaves the exponents free — which is not pedantry: published
parameter sets often quote a rounded `n` (1.06383 rather than 1.0638297…), and silently
recomputing it from `m` shifts the curve.
"""
struct VanGenuchten{T} <: AbstractRetention
    a::T   # air-entry pressure [Pa]
    n::T   # exponent [-]
    m::T   # exponent [-]
end

VanGenuchten(a, n, m) = VanGenuchten(promote(a, n, m)...)
VanGenuchten(a, m) = VanGenuchten(a, one(m) / (one(m) - m), m)

Base.eltype(::VanGenuchten{T}) where {T} = T

function saturation(r::VanGenuchten, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return one(T)
    return (one(T) + (pc / r.a)^r.n)^(-r.m)
end

function dsaturation_dpc(r::VanGenuchten, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return zero(T)
    u = (pc / r.a)^r.n
    return -r.m * (one(T) + u)^(-(r.m + one(T))) * r.n * u / pc
end

# ── Exponential regularization ────────────────────────────────────────────────

"""
    ExponentialCutoff(raw, p_c3)

Wrap a retention curve so that below `p_c3` it is replaced by an exponential branch
joined at `p_c3`:

```math
S_l(p_c) = 1 - \\left(1 - S_l^{\\text{raw}}(p_{c3})\\right)
           \\exp\\!\\left(\\frac{p_c - p_{c3}}{p_{c3}}\\right), \\qquad 0 < p_c < p_{c3}
```

Van Genuchten curves have an unbounded derivative as ``p_c \\to 0``, which starves the
Newton solver near saturation. The exponential branch keeps the slope finite while
matching the raw curve at the junction.
"""
struct ExponentialCutoff{R <: AbstractRetention, T} <: AbstractRetention
    raw::R
    p_c3::T
end

Base.eltype(r::ExponentialCutoff) = promote_type(eltype(r.raw), typeof(r.p_c3))

function saturation(r::ExponentialCutoff, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return one(T)
    pc >= r.p_c3 && return saturation(r.raw, pc)
    sl3 = saturation(r.raw, r.p_c3)
    return one(T) - (one(T) - sl3) * exp((pc - r.p_c3) / r.p_c3)
end

function dsaturation_dpc(r::ExponentialCutoff, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return zero(T)
    pc >= r.p_c3 && return dsaturation_dpc(r.raw, pc)
    sl3 = saturation(r.raw, r.p_c3)
    return -(one(T) - sl3) * exp((pc - r.p_c3) / r.p_c3) / r.p_c3
end

# ── Gardner ───────────────────────────────────────────────────────────────────

"""
    Gardner(α)

Exponential retention curve [gardner1958](@cite):

```math
S_l(p_c) = \\exp(-\\alpha\\, p_c), \\qquad p_c > 0
```

`α` [Pa⁻¹] is the inverse of a characteristic capillary pressure.

The exponential model is the one nonlinearity for which the steady Richards equation
integrates in closed form, which is what makes it a verification case rather than a
fitting curve. Real soils are usually better described by [`VanGenuchten`](@ref).
"""
struct Gardner{T} <: AbstractRetention
    α::T
end

Base.eltype(::Gardner{T}) where {T} = T

function saturation(r::Gardner, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return one(T)
    return exp(-r.α * pc)
end

function dsaturation_dpc(r::Gardner, pc)
    T = promote_type(typeof(pc), eltype(r))
    pc <= 0 && return zero(T)
    return -r.α * exp(-r.α * pc)
end

# ── Tabulated ─────────────────────────────────────────────────────────────────

"""
    Tabulated(pc, sl)

A retention curve given as a table rather than a formula: `sl[k]` is the saturation at
capillary pressure `pc[k]`, with linear interpolation between the points and clamping to the
end values outside the range.

Measured curves arrive this way, and so do the curves another code has already discretised —
Bil writes the table it interpolated next to its deck, and reading that table back is what
lets a comparison separate a disagreement about the physics from a disagreement about how a
curve was sampled.

`pc` must be sorted and strictly increasing; the constructor checks it, because a table that
is merely *nearly* sorted produces a plausible curve with a wrong branch in it.

Both vectors are type parameters, so a table of `Dual`s differentiates like any other
coefficient — the values of a measured curve are material parameters too.

```julia
pc, sl, krl = eachcol(readdlm("billes"))
model = RichardsModel(; retention = Tabulated(pc, sl), rel_perm = TabulatedKrl(pc, krl))
```
"""
struct Tabulated{V <: AbstractVector, W <: AbstractVector} <: AbstractRetention
    pc::V
    sl::W

    function Tabulated(pc::V, sl::W) where {V <: AbstractVector, W <: AbstractVector}
        length(pc) == length(sl) || throw(
            ArgumentError("Tabulated: $(length(pc)) pressures but $(length(sl)) saturations")
        )
        length(pc) >= 2 || throw(ArgumentError("Tabulated: need at least two points"))
        issorted(pc) && allunique(pc) || throw(
            ArgumentError("Tabulated: `pc` must be sorted and strictly increasing")
        )
        return new{V, W}(pc, sl)
    end
end

"""
    interpolate_table(x, y, q)

Linear interpolation of the table `(x, y)` at `q`, clamped to the end values outside the
range.

The `+ zero(q)` on the clamped branches is not decoration: without it a `Dual` argument
falls back to a `Float64` return and the derivative is silently lost — the same trap as
returning the literal `1.0` from a saturated branch.
"""
function interpolate_table(x, y, q)
    q <= first(x) && return first(y) + zero(q)
    q >= last(x) && return last(y) + zero(q)
    j = searchsortedlast(x, q)
    t = (q - x[j]) / (x[j + 1] - x[j])
    return y[j] + t * (y[j + 1] - y[j])
end

saturation(r::Tabulated, pc) = interpolate_table(r.pc, r.sl, pc)

function dsaturation_dpc(r::Tabulated, pc)
    ## The slope of the segment containing `pc`; zero outside the table, where the curve is
    ## clamped and therefore flat.
    (pc <= first(r.pc) || pc >= last(r.pc)) && return zero(pc) / oneunit(eltype(r.pc))
    j = searchsortedlast(r.pc, pc)
    return (r.sl[j + 1] - r.sl[j]) / (r.pc[j + 1] - r.pc[j])
end
