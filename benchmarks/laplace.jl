# Numerical inversion of Laplace transforms, shared by the benchmarks whose reference
# solution is closed-form in Laplace space rather than in time: Cryer's sphere, de Leeuw's
# cylinder, and the transient Gardner column.

# ── Stehfest inversion ────────────────────────────────────────────────────────

"""
    stehfest_weights(T, N)

Stehfest weights `V_k`, `N` even. Build them in `BigFloat`: they alternate in sign and reach
``3.6\\times10^{9}`` at `N = 16`, and in `Float64` the cancellation leaves nothing.
"""
function stehfest_weights(::Type{T}, N) where {T}
    V = zeros(T, N)
    h = N ÷ 2
    for k in 1:N
        s = zero(T)
        for j in cld(k, 2):min(k, h)
            s += T(j)^h * factorial(T(2j)) / (
                factorial(T(h - j)) * factorial(T(j)) * factorial(T(j - 1)) *
                    factorial(T(k - j)) * factorial(T(2j - k))
            )
        end
        V[k] = (-1)^(k + h) * s
    end
    return V
end

"""
    stehfest(F, t; N = 16, T = BigFloat)

Invert a Laplace transform numerically: `f(t) ≈ (ln2/t) Σ_k V_k F(k ln2 / t)`.

`F` is called with a `T`-typed argument and may return whatever precision it can manage —
elementary transforms evaluate happily in `BigFloat`, whereas one built on Bessel functions
has to drop to `Float64`, which the weights still carry.
"""
function stehfest(F, t; N = 16, T = BigFloat)
    V = stehfest_weights(T, N)
    ln2 = log(T(2))
    return Float64(ln2 / t * sum(V[k] * F(k * ln2 / T(t)) for k in 1:N))
end
