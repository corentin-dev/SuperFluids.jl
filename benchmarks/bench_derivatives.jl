# Benchmark: spectral (FFT) vs finite difference vs compact, for the derivatives
# dx, ddx, dy, ddy of one complex component (2D, single rank).
#
# Two questions answered:
#   1. Numerical: accuracy (max error vs analytic) as a function of resolution N,
#      and the implied convergence order.
#   2. Cost: wall time of computeDerivatives!, arithmetic throughput estimate,
#      and the "cost to reach a target accuracy" (N required for err < threshold).
#
# Schemes:
#   FFT : spectral (exact for resolved modes)
#   FD6 : plain central difference, order 6 (the package default)
#   FD8 : plain central difference, order 8 (via low-level kernels)
#   CP6 : compact finite difference, order 6 (the new periodic compact scheme)
#
# Run:  julia --project=. -t1 -O3 benchmarks/bench_derivatives.jl

using SuperFluids
using Printf: @printf, @sprintf
using PencilArrays: parent, transpose!

const T = Float64

# Smooth multi-mode periodic test function + analytic derivatives.
const MODES = [(2, 3, T(1.0)), (4, 1, T(0.5)), (1, 5, T(0.25)), (6, 2, T(0.125))]
function fval(x, y)
    s = zero(T)
    for (m, n, amp) in MODES
        s += amp * cos(2π * (m * x / 24 + n * y / 24) + 0.3)
    end
    return s
end
function fdx(x, y)
    s = zero(T)
    for (m, n, amp) in MODES
        s += -amp * 2π * (m / 24) * sin(2π * (m * x / 24 + n * y / 24) + 0.3)
    end
    return s
end
function fdy(x, y)
    s = zero(T)
    for (m, n, amp) in MODES
        s += -amp * 2π * (n / 24) * sin(2π * (m * x / 24 + n * y / 24) + 0.3)
    end
    return s
end
function fddx(x, y)
    s = zero(T)
    for (m, n, amp) in MODES
        s += -amp * (2π * m / 24)^2 * cos(2π * (m * x / 24 + n * y / 24) + 0.3)
    end
    return s
end
function fddy(x, y)
    s = zero(T)
    for (m, n, amp) in MODES
        s += -amp * (2π * n / 24)^2 * cos(2π * (m * x / 24 + n * y / 24) + 0.3)
    end
    return s
end

function fill_field!(field)
    x = vec(field.x); y = vec(field.y)
    A = parent(field.ϕ)
    nx, ny = size(A)
    for j in 1:ny, i in 1:nx
        @inbounds A[i, j] = fval(x[i], y[j])
    end
    return field
end

function exact_fields(field)
    x = vec(field.x); y = vec(field.y)
    nx, ny = size(parent(field.ϕ))
    d1x = [fdx(x[i], y[j]) for i=1:nx, j=1:ny]
    d2x = [fddx(x[i], y[j]) for i=1:nx, j=1:ny]
    d1y = [fdy(x[i], y[j]) for i=1:nx, j=1:ny]
    d2y = [fddy(x[i], y[j]) for i=1:nx, j=1:ny]
    return d1x, d2x, d1y, d2y
end

function max_err(gf, d1x, d2x, d1y, d2y)
    e = maximum(abs.(parent(gf.dx) .- d1x))
    e = max(e, maximum(abs.(parent(gf.dy) .- d1y)))
    e = max(e, maximum(abs.(parent(gf.ddx) .- d2x)))
    e = max(e, maximum(abs.(parent(gf.ddy) .- d2y)))
    return e
end

# trimmed min of several runs
function bench_min(f, iters=21)
    for _ in 1:7; f(); end
    ts = Vector{Float64}(undef, iters)
    for k in 1:iters
        t0 = time_ns()
        f()
        ts[k] = (time_ns() - t0) / 1e9
    end
    sort!(ts)
    return ts[5]   # 5th smallest of 21
end

# FLOP models (complex, one field of n*m points, four derivative fields).
# Rough counts for the stencil / solve arithmetic (for relative comparison only;
# at these sizes the kernels are memory/overhead-bound, not FLOP-bound).
const FLOP = Dict(
    "FFT" => (n, m) -> 4 * 5 * (n * m) * (log2(n) + log2(m)),
    "FD2" => (n, m) -> 8 * n * m,
    "FD4" => (n, m) -> 18 * n * m,
    "FD6" => (n, m) -> 24 * n * m,
    "FD8" => (n, m) -> 32 * n * m,
    "CP6" => (n, m) -> 29 * n * m,
)

# Low-level FD call for an arbitrary order, with pre-allocated y scratch.
function fd_lowlevel!(gf, plan, field, scratch1, scratch2, order)
    SuperFluids.computedxddx!(parent(field.ϕ), parent(gf.dx), parent(gf.ddx), field.g.Δx; order=order)
    transpose!(plan.ϕytmp, field.ϕ)
    SuperFluids.computedyddy!(parent(plan.ϕytmp), parent(scratch1), parent(scratch2), field.g.Δy; order=order)
    transpose!(gf.dy, scratch1); transpose!(gf.ddy, scratch2)
    return nothing
end

function main()
    Ns = [64, 128, 256, 512]
    println("Spectral vs finite-difference vs compact derivative schemes")
    println("Input: 4 cosine modes on [0,24]^2, one complex component.")
    println("Outputs measured: dx, ddx, dy, ddy.  Thread count: 1.")
    println()

    # accumulate: results[scheme][N] = (err, tmin)
    schemes = ["FFT", "FD6", "FD8", "CP6"]
    R = Dict(s => Dict{Int, Tuple{Float64, Float64}}() for s in schemes)

    for N in Ns
        field = Field(Grid((N, N), ((0, 24), (0, 24))), ComplexField())
        fill_field!(field)
        d1x, d2x, d1y, d2y = exact_fields(field)
        gf = GradientField(field; rotation=false)
        plan_fft = Plan(field)
        plan_fd = Plan(field; t=SuperFluids.FiniteDifferencePlan())
        plan_cp = Plan(field; t=SuperFluids.CompactPlan())
        s1 = similar(plan_fd.ϕytmp); s2 = similar(plan_fd.ϕytmp)

        # FFT
        cFFT = () -> SuperFluids.computeDerivatives!(gf, plan_fft, field.ϕ)
        cFFT(); e = max_err(gf, d1x, d2x, d1y, d2y)
        R["FFT"][N] = (e, bench_min(cFFT))

        # FD6 (default, high-level API)
        cFD6 = () -> SuperFluids.computeDerivatives!(gf, plan_fd, field.ϕ)
        cFD6(); e = max_err(gf, d1x, d2x, d1y, d2y)
        R["FD6"][N] = (e, bench_min(cFD6))

        # FD8 (low-level, order=8)
        cFD8 = () -> fd_lowlevel!(gf, plan_fd, field, s1, s2, 8)
        cFD8(); e = max_err(gf, d1x, d2x, d1y, d2y)
        R["FD8"][N] = (e, bench_min(cFD8))

        # CP6 (high-level API)
        cCP6 = () -> SuperFluids.computeDerivatives!(gf, plan_cp, field.ϕ)
        cCP6(); e = max_err(gf, d1x, d2x, d1y, d2y)
        R["CP6"][N] = (e, bench_min(cCP6))
    end

    # ---- Table 1: accuracy vs resolution ----
    println("Table 1 — max error vs analytic (lower is better)")
    @printf("%-6s", "N")
    for s in schemes; @printf("  %-12s", s); end
    println()
    for N in Ns
        @printf("%-6d", N)
        for s in schemes; @printf("  %-12.2e", R[s][N][1]); end
        println()
    end

    # ---- Table 2: implied convergence order ----
    println()
    println("Table 2 — implied convergence order (ΔN=2x)")
    @printf("%-6s", "N")
    for s in schemes; @printf("  %-12s", s); end
    println()
    for i in 2:length(Ns)
        @printf("%-6d", Ns[i])
        for s in schemes
            e0 = R[s][Ns[i-1]][1]; e1 = R[s][Ns[i]][1]
            o = (e0 > 0 && e1 > 0) ? log(e0/e1)/log(2) : NaN
            @printf("  %-12.2f", o)
        end
        println()
    end

    # ---- Table 3: cost (time, GFLOP/s estimate) ----
    println()
    println("Table 3 — wall time (s) and arithmetic throughput estimate (GFLOP/s)")
    println("  (kernels are memory/overhead-bound at these sizes; GFLOP/s is indicative)")
    @printf("%-6s", "N")
    for s in schemes
        @printf("  %-16s", s)
    end
    println()
    for N in Ns
        @printf("%-6d", N)
        for s in schemes
            e, t = R[s][N]
            fl = FLOP[s](N, N)
            @printf("  %-16s", string(s, " t=", @sprintf("%.3e", t), "  GF/s=", @sprintf("%.1f", fl/t/1e9)))
        end
        println()
    end

    # ---- Table 4: cost to reach a target accuracy ----
    println()
    println("Table 4 — grid points per axis (N) needed to reach a max-error target")
    for (thr, label) in [(1e-2, "1e-2"), (1e-4, "1e-4"), (1e-6, "1e-6"),
                         (1e-8, "1e-8"), (1e-10, "1e-10"), (1e-12, "1e-12")]
        @printf("%-8s", label)
        for s in schemes
            hit = nothing
            for N in Ns
                if R[s][N][1] < thr
                    hit = N; break
                end
            end
            @printf("  %-12s", hit === nothing ? ">512" : string(hit))
        end
        println()
    end

    # ---- Summary: accuracy per unit time ----
    println()
    println("Summary — accuracy (decades of error below 1.0) per run, N=256")
    N = 256
    for s in schemes
        e, t = R[s][N]
        dec = -log10(e)
        @printf("  %-4s: err=%.1e  time=%.3e s  -> %.2f decades in %.3e s  (%.1f decades/s)\n",
                s, e, t, dec, t, dec/t)
    end
end

# run f once, then measure the error of the resulting gf
function max_err_after!(gf, d1x, d2x, d1y, d2y, f)
    f()
    return max_err(gf, d1x, d2x, d1y, d2y)
end

main()
