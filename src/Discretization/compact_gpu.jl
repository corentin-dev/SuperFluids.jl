# =============================================================================
# CUDA backend for the 6th-order periodic compact scheme
#
# On GPU the compact relation is solved with a dedicated Thomas kernel instead
# of the spectral (Fourier-multiplier) path: one thread per grid line performs
# the stencil, the Thomas sweep, the cyclic correction and the back
# substitution. The banded Thomas multipliers (`CompactAxis`, CPU side) are
# uploaded to the device once per axis at plan construction, so the kernel
# only does the per-call work.
#
# Kernels solve along the leading (fastest) dimension of a 2D `(n, L)` or 3D
# `(n, L2, L3)` raw array; the y/z directions are handled with PencilArray
# transposes that bring the derivative axis to the front, mirroring the CPU
# dispatch in `derivatives.jl`.
#
# The kernels are dtype-agnostic (real or complex fields).
#
# This file is included from `Plans.jl` inside `try using CUDA ... catch
# CUDA = nothing end`: when CUDA cannot be loaded, none of it is defined and
# the compact plan does not offer the GPU kernel backend.
# =============================================================================

@inline function _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    @inbounds begin
        for i in 2:n
            r[i, j] -= r[i - 1, j] * s[i]
        end
        r[n, j] *= w[n]
        for i in (n - 1):-1:1
            r[i, j] = (r[i, j] - f[i] * r[i + 1, j]) * w[i]
        end
        sx = (r[1, j] - α * r[n, j]) / denom
        for i in 1:n
            r[i, j] -= sx * t[i]
        end
    end
    return
end

@inline function _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    @inbounds begin
        for i in 2:n
            r[i, j, k] -= r[i - 1, j, k] * s[i]
        end
        r[n, j, k] *= w[n]
        for i in (n - 1):-1:1
            r[i, j, k] = (r[i, j, k] - f[i] * r[i + 1, j, k]) * w[i]
        end
        sx = (r[1, j, k] - α * r[n, j, k]) / denom
        for i in 1:n
            r[i, j, k] -= sx * t[i]
        end
    end
    return
end

# --- 2D kernels (array layout (n, L), line j is processed by thread j) ------

function _compact_cu_1k!(u, r, n, a, b, s, w, f, t, α, denom, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = a * (u[2, j] - u[n, j]) + b * (u[3, j] - u[n - 1, j])
        r[2, j] = a * (u[3, j] - u[1, j]) + b * (u[4, j] - u[n, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - u[i - 1, j]) + b * (u[i + 2, j] - u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - u[n - 2, j]) + b * (u[1, j] - u[n - 3, j])
        r[n, j] = a * (u[1, j] - u[n - 1, j]) + b * (u[2, j] - u[n - 2, j])
    end
    _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    return
end

function _compact_cu_2k!(u, r, n, a, b, s, w, f, t, α, denom, L)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j > L && return
    @inbounds begin
        r[1, j] = a * (u[2, j] - 2u[1, j] + u[n, j]) + b * (u[3, j] - 2u[1, j] + u[n - 1, j])
        r[2, j] = a * (u[3, j] - 2u[2, j] + u[1, j]) + b * (u[4, j] - 2u[2, j] + u[n, j])
        for i in 3:(n - 2)
            r[i, j] = a * (u[i + 1, j] - 2u[i, j] + u[i - 1, j]) + b * (u[i + 2, j] - 2u[i, j] + u[i - 2, j])
        end
        r[n - 1, j] = a * (u[n, j] - 2u[n - 1, j] + u[n - 2, j]) + b * (u[1, j] - 2u[n - 1, j] + u[n - 3, j])
        r[n, j] = a * (u[1, j] - 2u[n, j] + u[n - 1, j]) + b * (u[2, j] - 2u[n, j] + u[n - 2, j])
    end
    _compact_cu_solve!(r, n, j, s, w, f, t, α, denom)
    return
end

# --- 3D kernels (array layout (n, L2, L3), one thread per (j, k) line) -------

function _compact_cu_1k3!(u, r, n, a, b, s, w, f, t, α, denom, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = a * (u[2, j, k] - u[n, j, k]) + b * (u[3, j, k] - u[n - 1, j, k])
        r[2, j, k] = a * (u[3, j, k] - u[1, j, k]) + b * (u[4, j, k] - u[n, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - u[i - 1, j, k]) + b * (u[i + 2, j, k] - u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - u[n - 2, j, k]) + b * (u[1, j, k] - u[n - 3, j, k])
        r[n, j, k] = a * (u[1, j, k] - u[n - 1, j, k]) + b * (u[2, j, k] - u[n - 2, j, k])
    end
    _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    return
end

function _compact_cu_2k3!(u, r, n, a, b, s, w, f, t, α, denom, L2, L3)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = L2 * L3
    idx > L && return
    j = (idx - 1) % L2 + 1
    k = (idx - 1) ÷ L2 + 1
    @inbounds begin
        r[1, j, k] = a * (u[2, j, k] - 2u[1, j, k] + u[n, j, k]) + b * (u[3, j, k] - 2u[1, j, k] + u[n - 1, j, k])
        r[2, j, k] = a * (u[3, j, k] - 2u[2, j, k] + u[1, j, k]) + b * (u[4, j, k] - 2u[2, j, k] + u[n, j, k])
        for i in 3:(n - 2)
            r[i, j, k] = a * (u[i + 1, j, k] - 2u[i, j, k] + u[i - 1, j, k]) + b * (u[i + 2, j, k] - 2u[i, j, k] + u[i - 2, j, k])
        end
        r[n - 1, j, k] = a * (u[n, j, k] - 2u[n - 1, j, k] + u[n - 2, j, k]) + b * (u[1, j, k] - 2u[n - 1, j, k] + u[n - 3, j, k])
        r[n, j, k] = a * (u[1, j, k] - 2u[n, j, k] + u[n - 1, j, k]) + b * (u[2, j, k] - 2u[n, j, k] + u[n - 2, j, k])
    end
    _compact_cu_solve3!(r, n, j, k, s, w, f, t, α, denom)
    return
end

# --- line drivers -------------------------------------------------------------
# One thread per line; the sequential Thomas sweep inside a thread makes the
# kernel memory/latency bound (one pass reads the line, one writes it), which
# is the best achievable without a parallel (e.g. cyclic-reduction) solve.

function _compact_cu_1line!(r, u, n, c)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L)
    return
end

function _compact_cu_2line!(r, u, n, c)
    L = size(u, 2)
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L)
    return
end

function _compact_cu_1line3!(r, u, n, c)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_1k3!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L2, L3)
    return
end

function _compact_cu_2line3!(r, u, n, c)
    L2, L3 = size(u, 2), size(u, 3)
    L = L2 * L3
    threads = 128
    @cuda threads=threads blocks=(L + threads - 1) ÷ threads _compact_cu_2k3!(u, r, n, c.a, c.b, c.s, c.w, c.f, c.t, c.alpha, c.denom, L2, L3)
    return
end

# Device-side per-axis compact multipliers: the same precomputed Thomas data
# as the CPU `CompactAxis`, with the banded vectors uploaded to the device so
# they can be read directly from the kernels.
struct CompactAxisGPU
    "number of grid points along the axis."
    n::Int
    "compact coefficient: 1/3 (1st order) or 2/11 (2nd order)."
    alpha::Float64
    "stencil coefficient a."
    a::Float64
    "stencil coefficient b."
    b::Float64
    "Thomas forward multipliers (device)."
    s::CuArray{Float64, 1}
    "reciprocals of the Thomas pivots (device)."
    w::CuArray{Float64, 1}
    "super-diagonal of the (unchanged) tridiagonal (device)."
    f::CuArray{Float64, 1}
    "precomputed cyclic correction vector (device)."
    t::CuArray{Float64, 1}
    "cyclic correction denominator."
    denom::Float64
end

"""$(TYPEDSIGNATURES)

Build the device-side compact multipliers for one axis by reusing the CPU
[`compact_setup`](@ref) (guaranteeing identical coefficients) and uploading
the banded vectors to the GPU.
"""
function compact_setup_gpu(n::Int, Δ::Real, order::Int)
    c = compact_setup(n, Δ, order)
    return CompactAxisGPU(n, c.alpha, c.a, c.b,
                          CuArray(c.s), CuArray(c.w), CuArray(c.f), CuArray(c.t), c.denom)
end
