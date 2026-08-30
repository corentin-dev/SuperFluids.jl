using Test
using SuperFluids
using PencilArrays: localgrid

# Relative error helper: max|a-b| / max|b|
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

# Tolerances.
# FFT is spectral -> exact at machine precision for resolved modes.
# FD is 6th order -> error ~ O((k*Δx)^6); keep a generous bound that still
# catches factor-of-2 and gross layout errors.
const TOL_FFT = 1e-10
const TOL_FD  = 1e-3

# --- helpers to lay out a single Fourier mode on a grid ---
function set_mode_2d!(field, m, n_)
    X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
    Lx, Ly = field.g.Lx, field.g.Ly
    kx, ky = 2π*m/Lx, 2π*n_/Ly
    field.ϕ .= exp.(1im * (kx * X .+ ky * Y))
    return kx, ky
end

function set_mode_3d!(field, m, n_, p)
    X = reshape(vec(field.x), :, 1, 1); Y = reshape(vec(field.y), 1, :, 1)
    Z = reshape(vec(field.z), 1, 1, :)
    Lx, Ly, Lz = field.g.Lx, field.g.Ly, field.g.Lz
    kx, ky, kz = 2π*m/Lx, 2π*n_/Ly, 2π*p/Lz
    field.ϕ .= exp.(1im * (kx * X .+ ky * Y .+ kz * Z))
    return kx, ky, kz
end

# ==========================================================================
# 2D FFT  (rotation=true)  -- the GP/BEC 2D path
# ==========================================================================
@testset "2D FFT derivatives (rotation=true)" begin
    field = Field(Grid((64, 64), ((-12, 12), (-12, 12))), ComplexField())
    X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=true)
    plan = Plan(field)
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FFT
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FFT
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FFT
    @test relerr(gf.rx,  Y .* gf.dx)   < TOL_FFT
    @test relerr(gf.ry, -X .* gf.dy)   < TOL_FFT
end

# ==========================================================================
# 3D FFT  (rotation=true)  -- the GP/BEC 3D path
# ==========================================================================
@testset "3D FFT derivatives (rotation=true)" begin
    field = Field(Grid((48, 48, 48), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    X = reshape(vec(field.x), :, 1, 1); Y = reshape(vec(field.y), 1, :, 1)
    Z = reshape(vec(field.z), 1, 1, :)
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=true)
    plan = Plan(field)
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FFT
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FFT
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FFT
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FFT
    @test relerr(gf.rx,  Y .* gf.dx)   < TOL_FFT
    @test relerr(gf.ry, -X .* gf.dy)   < TOL_FFT
end

# ==========================================================================
# 2D FFT  (rotation=false)  -- GradientField2D
# ==========================================================================
@testset "2D FFT derivatives (rotation=false)" begin
    field = Field(Grid((64, 64), ((-12, 12), (-12, 12))), ComplexField())
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=false)
    plan = Plan(field)
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FFT
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FFT
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FFT
end

# ==========================================================================
# 3D FFT  (rotation=false)  -- GradientField3D
# ==========================================================================
@testset "3D FFT derivatives (rotation=false)" begin
    field = Field(Grid((48, 48, 48), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=false)
    plan = Plan(field)
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FFT
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FFT
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FFT
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FFT
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FFT
end

# ==========================================================================
# 2D FD  (rotation=true)  -- the BEC_2D_DF path
# ==========================================================================
@testset "2D FD derivatives (rotation=true)" begin
    field = Field(Grid((128, 128), ((-12, 12), (-12, 12))), ComplexField())
    X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=true)
    plan = Plan(field; t=SuperFluids.FiniteDifferencePlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.rx,  Y .* gf.dx)   < TOL_FD
    @test relerr(gf.ry, -X .* gf.dy)   < TOL_FD
end

# ==========================================================================
# 2D FD  (rotation=false)  -- GradientField2D
# ==========================================================================
@testset "2D FD derivatives (rotation=false)" begin
    field = Field(Grid((128, 128), ((-12, 12), (-12, 12))), ComplexField())
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=false)
    plan = Plan(field; t=SuperFluids.FiniteDifferencePlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
end

# ==========================================================================
# 3D FD  (rotation=true)
# ==========================================================================
@testset "3D FD derivatives (rotation=true)" begin
    field = Field(Grid((64, 64, 64), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=true)
    plan = Plan(field; t=SuperFluids.FiniteDifferencePlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FD
end

# ==========================================================================
# 3D FD  (rotation=false)  -- GradientField3D
# ==========================================================================
@testset "3D FD derivatives (rotation=false)" begin
    field = Field(Grid((64, 64, 64), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=false)
    plan = Plan(field; t=SuperFluids.FiniteDifferencePlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FD
end

# ==========================================================================
# Compact  (6th-order periodic compact scheme, port of GPS cdl==0)
# ==========================================================================
# Same Fourier-mode accuracy checks as the FD/FFT tests; the compact operator
# is 6th order, so the same generous TOL_FD bound applies (and is comfortably
# met).
@testset "2D compact derivatives (rotation=true)" begin
    field = Field(Grid((128, 128), ((-12, 12), (-12, 12))), ComplexField())
    X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=true)
    plan = Plan(field; t=SuperFluids.CompactPlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.rx,  Y .* gf.dx)   < TOL_FD
    @test relerr(gf.ry, -X .* gf.dy)   < TOL_FD
end

@testset "2D compact derivatives (rotation=false)" begin
    field = Field(Grid((128, 128), ((-12, 12), (-12, 12))), ComplexField())
    kx, ky = set_mode_2d!(field, 2, 3)
    gf = GradientField(field; rotation=false)
    plan = Plan(field; t=SuperFluids.CompactPlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
end

@testset "3D compact derivatives (rotation=true)" begin
    field = Field(Grid((64, 64, 64), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    X = reshape(vec(field.x), :, 1, 1); Y = reshape(vec(field.y), 1, :, 1)
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=true)
    plan = Plan(field; t=SuperFluids.CompactPlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FD
    @test relerr(gf.rx,  Y .* gf.dx)   < TOL_FD
    @test relerr(gf.ry, -X .* gf.dy)   < TOL_FD
end

@testset "3D compact derivatives (rotation=false)" begin
    field = Field(Grid((64, 64, 64), ((-12, 12), (-12, 12), (-12, 12))), ComplexField())
    kx, ky, kz = set_mode_3d!(field, 2, 3, 4)
    gf = GradientField(field; rotation=false)
    plan = Plan(field; t=SuperFluids.CompactPlan())
    SuperFluids.computeDerivatives!(gf, plan, field.ϕ)
    ϕ = field.ϕ
    @test relerr(gf.dx,  1im * kx * ϕ) < TOL_FD
    @test relerr(gf.dy,  1im * ky * ϕ) < TOL_FD
    @test relerr(gf.dz,  1im * kz * ϕ) < TOL_FD
    @test relerr(gf.ddx, -kx^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddy, -ky^2 * ϕ)   < TOL_FD
    @test relerr(gf.ddz, -kz^2 * ϕ)   < TOL_FD
end

# ==========================================================================
# GP explicit Runge-Kutta (NumModelGPRK) — port of the reference GP_RK4
# ==========================================================================
# Linear case (β = V = Ω = 0): ψ̂(t) = exp(i·coeffΔ·k²·t) ψ̂(0), an exact phase
# rotation. A plane wave k=(1,1) has a known exact solution, giving a clean
# convergence-order check for RK1/RK2/RK4 and a norm check.
@testset "GP explicit RK: plane-wave order and norm preservation (2D)" begin
    grid = Grid((32, 32), ((-2π, 2π), (-2π, 2π)))
    coeffΔ = -0.5
    T = 0.4
    k2 = 2.0
    nrm(a) = sqrt(sum(abs.(a) .^ 2))
    function run(stepper, nsteps)
        field = Field(grid, ComplexField())
        X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
        @. field.ϕ = exp(1im * (X + Y))
        pot = PotentialZero(field)
        param = GrossPitaevskiiParameters(coeffΔ=coeffΔ, β=0.0, Ω=0.0, pot=pot)
        n = NumModelGPRK(field, param, T / nsteps, nsteps, 1; stepper=stepper)
        for _ in 1:nsteps
            SuperFluids.timeStep!(n)
        end
        exf = Field(grid, ComplexField())
        Xe = reshape(vec(exf.x), :, 1); Ye = reshape(vec(exf.y), 1, :)
        @. exf.ϕ = exp(1im * (Xe + Ye))
        ψx = parent(exf.ϕ) .* exp(1im * coeffΔ * k2 * T)
        ψm = parent(field.ϕ)
        return nrm(ψm - ψx) / nrm(ψx), nrm(ψm) / nrm(ψx)
    end
    for (stepper, lo, hi) in (("RK1", 0.85, 1.15), ("RK2", 1.85, 2.15), ("RK4", 3.85, 4.15))
        e_c, n_c = run(stepper, 40)
        e_f, n_f = run(stepper, 80)
        order = log2(e_c / e_f)
        @test lo < order < hi
        @test abs(n_f - 1.0) < 5e-3          # phase rotation preserves the norm
        @test e_f > 0.0                      # the fine run is not exact (no trivial pass)
    end
end

# Nonlinear case (β > 0): the mass ∫|ψ|² is conserved by the GP equation; the
# explicit RK integrator must preserve it (up to the truncation error) on a
# Gaussian initial state.
@testset "GP explicit RK: nonlinear mass conservation (2D)" begin
    grid = Grid((32, 32), ((-2π, 2π), (-2π, 2π)))
    function mass_rel(stepper, dt, T, β)
        field = Field(grid, ComplexField())
        @. field.ϕ = exp(-((field.x)^2 + (field.y)^2) / 6.0)
        m0 = sum(abs.(parent(field.ϕ)) .^ 2)
        pot = PotentialZero(field)
        param = GrossPitaevskiiParameters(coeffΔ=-0.5, β=β, Ω=0.0, pot=pot)
        nsteps = round(Int, T / dt)
        n = NumModelGPRK(field, param, dt, nsteps, 1; stepper=stepper)
        for _ in 1:nsteps
            SuperFluids.timeStep!(n)
        end
        m1 = sum(abs.(parent(field.ϕ)) .^ 2)
        return abs(m1 - m0) / m0
    end
    @test mass_rel("RK2", 0.004, 0.2, 2.0) < 1e-6
    @test mass_rel("RK4", 0.004, 0.2, 2.0) < 1e-7
end

# ==========================================================================
# 3D vector field: FFT round-trip (NS velocity field, ndims=3)
# ==========================================================================
@testset "3D vector FFT round-trip" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-3, 3)))
    field = Field(grid, ComplexField(); ndims=3)
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz
    kx, ky, kz = 2π/Lx, 2π/Ly, 2π/Lz
    X = reshape(vec(field.x), :, 1, 1); Y = reshape(vec(field.y), 1, :, 1)
    Z = reshape(vec(field.z), 1, 1, :)
    field.ux .= sin.(kx*X) .* cos.(ky*Y)
    field.uy .= cos.(ky*Y) .* sin.(kz*Z)
    field.uz .= sin.(kz*Z)
    plan = Plan(field)
    # mul_all! outputs in the last direction's layout (pen_z for 3D); ldiv_all!
    # takes that back to pen_x. So FFT forward into uz_hat (pen_z), inverse
    # into a fresh pen_x field.
    SuperFluids.mul_all!(plan.uz_hat, plan, field.u)
    back = SuperFluids.similar_data(field.u)
    SuperFluids.ldiv_all!(back, plan, plan.uz_hat)
    @test relerr(back[1], field.ux) < TOL_FFT
    @test relerr(back[2], field.uy) < TOL_FFT
    @test relerr(back[3], field.uz) < TOL_FFT
end

# ==========================================================================
# 3D curl (ω = ∇ × u) via FFT -- used by the Navier-Stokes solver
# ==========================================================================
@testset "3D curl (FFT)" begin
    # Non-cubic grid with different lengths, so an axis misalignment
    # (x/y/z swapped) would show up as an O(1) error.
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-3, 3)))
    field = Field(grid, ComplexField(); ndims=3)
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz
    kx, ky, kz = 2π/Lx, 2π/Ly, 2π/Lz
    X = reshape(vec(field.x), :, 1, 1); Y = reshape(vec(field.y), 1, :, 1)
    Z = reshape(vec(field.z), 1, 1, :)
    # u = (cos(ky y), sin(kz z), sin(kx x))
    field.ux .= cos.(ky*Y)
    field.uy .= sin.(kz*Z)
    field.uz .= sin.(kx*X)
    # ω = curl u = (∂y uz - ∂z uy, ∂z ux - ∂x uz, ∂x uy - ∂y ux)
    ωx_ref = -kz * cos.(kz*Z)
    ωy_ref = -kx * cos.(kx*X)
    ωz_ref =  ky * sin.(ky*Y)
    gf = GradientField(field; rotation=false, laplacian=false, vorticity=true)
    plan = Plan(field)
    SuperFluids.computeDerivatives!(gf, plan, field.u)
    ω = gf.ω
    @test length(gf.ωdata) == 3
    @test relerr(ω[1], ωx_ref) < TOL_FFT
    @test relerr(ω[2], ωy_ref) < TOL_FFT
    @test relerr(ω[3], ωz_ref) < TOL_FFT
end

# ==========================================================================
# norm / normalize!
# ==========================================================================
@testset "norm and normalize!" begin
    field = Field(Grid((64, 64), ((-12, 12), (-12, 12))), ComplexField())
    Lx, Ly = field.g.Lx, field.g.Ly
    field.ϕ .= 1.0
    @test norm(field) ≈ sqrt(Lx * Ly)
    normalize!(field)
    @test norm(field) ≈ 1.0
end

# ==========================================================================
# Navier-Stokes 3D
# ==========================================================================
# spectral divergence of a velocity given as a Vector{PencilArray} in the
# last-pencil layout:  div = max |imag(i k · û)|. Dimension-agnostic (2D/3D).
function div_max(n, u_hat)
    gridξ = SuperFluids.spectral_grid(n.plan)
    d = similar(u_hat[1])
    @. d = gridξ[1] * u_hat[1]
    for i in 2:ndims(gridξ)
        @. d += gridξ[i] * u_hat[i]
    end
    return maximum(abs.(imag.(parent(d))))
end

# ==========================================================================
# Helmholtz projector: removes the divergence, idempotent
# ==========================================================================
@testset "NS projector" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-6, 6)))
    field = Field(grid, ComplexField(); ndims=3)
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), 0.01, 1, 1)
    for c in 1:3
        parent(field.u[c]) .= ComplexF64.(randn(size(parent(field.u[c])))) +
            1im * ComplexF64.(randn(size(parent(field.u[c]))))
    end
    SuperFluids.mul_all!(n.u_hat, n.plan, field.u)
    d0 = div_max(n, n.u_hat)
    @test d0 > 10  # a random field is not divergence-free
    SuperFluids.project!(n, n.u_hat)
    @test div_max(n, n.u_hat) < 1e-10 * d0  # divergence removed
    # idempotency: projecting again changes nothing (to roundoff)
    u_before = copy(n.u_hat[1])
    SuperFluids.project!(n, n.u_hat)
    @test maximum(abs.(parent(n.u_hat[1]) .- parent(u_before))) <
          1e-12 * maximum(abs.(parent(u_before)))
end

# ==========================================================================
# taylor_green!: periodic and exactly divergence-free
# ==========================================================================
@testset "NS taylor_green initialisation" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-6, 6)))
    field = Field(grid, ComplexField(); ndims=3)
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), 0.01, 1, 1)
    taylor_green!(field, grid.x, grid.y, grid.z)
    SuperFluids.mul_all!(n.u_hat, n.plan, field.u)
    @test div_max(n, n.u_hat) < 1e-10
end

# ==========================================================================
# taylor_green! (3D): energy matches the closed-form analytic value
# E = A² Lx Ly Lz / 16 · (1 + (kx/ky)²)   (uz ≡ 0)
# Validates the energy normalisation (½, Δx·Δy·Δz) and the kx/ky scaling.
# ==========================================================================
@testset "NS taylor_green energy (3D)" begin
    for (nx, ny, nz, rng) in ((32, 24, 40, ((-4, 4), (-5, 5), (-6, 6))),
                              (64, 48, 56, ((-3, 3), (-4, 4), (-5, 5))))
        grid = Grid((nx, ny, nz), rng)
        A = 1.3
        field = Field(grid, ComplexField(); ndims=3)
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
        taylor_green!(field, grid.x, grid.y, grid.z)
        for c in 1:3
            parent(field.u[c]) .*= A
        end
        E_num = SuperFluids.energy(n)[2]
        E_ana = 0.5 * A^2 * grid.Lx * grid.Ly * grid.Lz / 8 * (1 + (grid.Ly / grid.Lx)^2)
        @test abs(E_num - E_ana) / E_ana < 1e-12
    end
end

# ==========================================================================
# Beltrami mode: u×ω = 0 holds identically, so u(t) = u(0) exp(-ν|k|²t)
# is the EXACT solution of the semi-implicit scheme (the implicit viscosity
# multiplier is exact). Validates the whole RHS pipeline at once.
# ==========================================================================
@testset "NS Beltrami exact decay" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-6, 6)))
    Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz
    kx, ky, kz = 2π / Lx, 2π / Ly, 2π / Lz
    knorm = sqrt(kx^2 + ky^2 + kz^2)
    # unit vector perpendicular to k
    k̂ = [kx, ky, kz] / knorm
    b = [k̂[1], 0.3, k̂[3]]
    b .-= (sum(b .* k̂)) * k̂
    b ./= sqrt(sum(b .* b))
    # û = (b - i k̂×b)/2  satisfies  k × û = i|k| û  =>  ω = -|k| u,
    # hence u × ω = -|k| (u × u) = 0.
    k̂×b = [k̂[2] * b[3] - k̂[3] * b[2],
           k̂[3] * b[1] - k̂[1] * b[3],
           k̂[1] * b[2] - k̂[2] * b[1]]
    û = (b .- 1im * k̂×b) / 2

    for (ν, expect_exact_decay) in ((0.02, true), (0.0, false))
        field = Field(grid, ComplexField(); ndims=3)
        Δt, Nsteps = 0.01, 10
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), Δt, Nsteps, 1)
        X = reshape(grid.x, :, 1, 1)
        Y = reshape(grid.y, 1, :, 1)
        Z = reshape(grid.z, 1, 1, :)
        ph = exp.(1im * (kx * X .+ ky * Y .+ kz * Z))
        field.ux .= û[1] * ph
        field.uy .= û[2] * ph
        field.uz .= û[3] * ph
        u0 = [copy(parent(field.ux)), copy(parent(field.uy)), copy(parent(field.uz))]
        for _ in 1:Nsteps
            SuperFluids.timeStep!(n)
        end
        decay = exp(-ν * knorm^2 * Nsteps * Δt)
        for c in 1:3
            err = maximum(abs.(parent(field.u[c]) .- decay * u0[c])) / maximum(abs.(u0[c]))
            @test err < 1e-8
        end
    end
end

# ==========================================================================
# RK4 order: with ν=0 the semi-implicit step is a plain RK4 applied to the
# nonlinear term, so the global error must converge as Δt⁴. The amplitude is
# chosen large enough that the O(Δt⁵) error is above the roundoff floor.
# ==========================================================================
@testset "NS RK4 convergence order" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-6, 6)))
    A = 2.0
    t_final = 0.1
    ν = 0.0

    function run(dt)
        field = Field(grid, ComplexField(); ndims=3)
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), dt,
                           Int(t_final / dt), 1)
        taylor_green!(field, grid.x, grid.y, grid.z)
        for c in 1:3
            parent(field.u[c]) .*= A
        end
        for _ in 1:Int(t_final / dt)
            SuperFluids.timeStep!(n)
        end
        return [parent(field.ux), parent(field.uy), parent(field.uz)]
    end

    ref = run(t_final / 1000)   # reference run (error ~128× smaller)
    e1 = run(t_final / 4)
    e2 = run(t_final / 8)
    e3 = run(t_final / 16)
    function nrm_err(e)
        sqrt(sum(abs.(e[1] .- ref[1]).^2) + sum(abs.(e[2] .- ref[2]).^2) +
             sum(abs.(e[3] .- ref[3]).^2))
    end
    o1 = log(nrm_err(e1) / nrm_err(e2)) / log(2)
    o2 = log(nrm_err(e2) / nrm_err(e3)) / log(2)
    @test 3.8 < o1 < 4.2
    @test 3.8 < o2 < 4.2
end

# ==========================================================================
# Taylor-Green time integration: energy strictly decays (dE/dt ≤ 0),
# the divergence stays at roundoff level, and there is no blow-up.
# ==========================================================================
@testset "NS Taylor-Green time integration" begin
    grid = Grid((32, 24, 40), ((-4, 4), (-5, 5), (-6, 6)))
    field = Field(grid, ComplexField(); ndims=3)
    Δt, Nsteps = 0.02, 40
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), Δt, Nsteps, 1)
    taylor_green!(field, grid.x, grid.y, grid.z)
    E0 = SuperFluids.energy(n)[2]
    Emax = E0
    for _ in 1:Nsteps
        SuperFluids.timeStep!(n)
        E = SuperFluids.energy(n)[2]
        Emax = max(Emax, E)
        # divergence of the *current* u_hat (kept by timeStep!) stays ~0
        @test div_max(n, n.u_hat) < 1e-8 * maximum(abs.(parent(n.u_hat[1])))
    end
    E = SuperFluids.energy(n)[2]
    @test E < E0          # viscosity damps the flow
    @test E > 0.5 * E0    # ... but not by an unphysical amount in 0.8s
    @test Emax < 1.01 * E0  # no blow-up
end

# ==========================================================================
# Navier-Stokes 2D
# ==========================================================================
# 2D vector field: FFT round-trip (NS velocity field, ndims=2)
# ==========================================================================
@testset "2D vector FFT round-trip" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    field = Field(grid, ComplexField(); ndims=2)
    Lx, Ly = grid.Lx, grid.Ly
    kx, ky = 2π / Lx, 2π / Ly
    X = reshape(vec(field.x), :, 1); Y = reshape(vec(field.y), 1, :)
    field.ux .= sin.(kx * X) .* cos.(ky * Y)
    field.uy .= cos.(ky * Y) .* sin.(kx * X)
    plan = Plan(field)
    # mul_all! outputs in the last direction's layout (pen_y for 2D); ldiv_all!
    # takes that back to pen_x.
    SuperFluids.mul_all!(plan.uy_hat, plan, field.u)
    back = SuperFluids.similar_data(field.u)
    SuperFluids.ldiv_all!(back, plan, plan.uy_hat)
    @test relerr(back[1], field.ux) < TOL_FFT
    @test relerr(back[2], field.uy) < TOL_FFT
end

# ==========================================================================
# Helmholtz projector (2D): removes the divergence, idempotent
# ==========================================================================
@testset "NS projector 2D" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    field = Field(grid, ComplexField(); ndims=2)
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), 0.01, 1, 1)
    for c in 1:2
        parent(field.u[c]) .= ComplexF64.(randn(size(parent(field.u[c])))) +
            1im * ComplexF64.(randn(size(parent(field.u[c]))))
    end
    SuperFluids.mul_all!(n.u_hat, n.plan, field.u)
    d0 = div_max(n, n.u_hat)
    @test d0 > 1  # a random field is not divergence-free
    SuperFluids.project!(n, n.u_hat)
    @test div_max(n, n.u_hat) < 1e-10 * d0  # divergence removed
    # idempotency: projecting again changes nothing (to roundoff)
    u_before = copy(n.u_hat[1])
    SuperFluids.project!(n, n.u_hat)
    @test maximum(abs.(parent(n.u_hat[1]) .- parent(u_before))) <
          1e-12 * maximum(abs.(parent(u_before)))
end

# ==========================================================================
# taylor_green! (2D): periodic and exactly divergence-free
# ==========================================================================
@testset "NS taylor_green initialisation 2D" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    field = Field(grid, ComplexField(); ndims=2)
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), 0.01, 1, 1)
    taylor_green!(field, grid.x, grid.y)
    SuperFluids.mul_all!(n.u_hat, n.plan, field.u)
    @test div_max(n, n.u_hat) < 1e-10
end

# ==========================================================================
# taylor_green! (2D): energy matches the closed-form analytic value
# E = A² Lx Ly / 8 · (1 + (kx/ky)²)
# Validates the energy normalisation (½, Δx·Δy) and the kx/ky scaling.
# ==========================================================================
@testset "NS taylor_green energy (2D)" begin
    for (nx, ny, rng) in ((32, 24, ((-4, 4), (-5, 5))),
                          (64, 48, ((-3, 3), (-4, 4))))
        grid = Grid((nx, ny), rng)
        A = 1.3
        field = Field(grid, ComplexField(); ndims=2)
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.0), 0.01, 1, 1)
        taylor_green!(field, grid.x, grid.y)
        for c in 1:2
            parent(field.u[c]) .*= A
        end
        E_num = SuperFluids.energy(n)[2]
        E_ana = 0.5 * A^2 * grid.Lx * grid.Ly / 4 * (1 + (grid.Ly / grid.Lx)^2)
        @test abs(E_num - E_ana) / E_ana < 1e-12
    end
end

# ==========================================================================
# 2D shear-flow exact decay: u = (U0 sin(ky y), 0) is divergence-free and its
# projected nonlinear term P(u × ω) is identically zero, so u(t) = u(0)·e^{-νky²t}
# is the EXACT solution of the semi-implicit scheme (the implicit viscosity
# multiplier is exact). Validates the whole 2D RHS pipeline at once.
# ==========================================================================
@testset "NS 2D shear-flow exact decay" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    Ly = grid.Ly
    ky = 2π / Ly
    U0 = 2.0
    for ν in (0.3, 0.0)
        field = Field(grid, ComplexField(); ndims=2)
        Δt, Nsteps = 0.01, 20
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), Δt, Nsteps, 1)
        Y = reshape(vec(grid.y), 1, :)
        field.ux .= U0 * sin.(ky * Y)
        field.uy .= 0
        u0 = copy(parent(field.ux))
        for _ in 1:Nsteps
            SuperFluids.timeStep!(n)
        end
        decay = exp(-ν * ky^2 * Nsteps * Δt)
        err = maximum(abs.(parent(field.ux) .- decay * u0)) / maximum(abs.(u0))
        @test err < 1e-8
        # the y-component must stay ~ 0
        @test maximum(abs.(parent(field.uy))) < 1e-8
    end
end

# ==========================================================================
# RK4 order (2D): with ν=0 the semi-implicit step is a plain RK4 applied to
# the nonlinear term, so the global error must converge as Δt⁴.
#
# A single 2D vortex mode has ζ ∝ ψ, so its Jacobian J(ψ,ζ) vanishes and the
# flow is frozen (nothing to converge). We use a superposition of two
# divergence-free modes, for which ζ ∉ span{ψ} and the nonlinear dynamics is
# genuinely active (|u(T)-u(0)| ≫ roundoff), so the O(Δt⁵) error sits above
# the roundoff floor and the order is measurable.
# ==========================================================================
@testset "NS RK4 convergence order 2D" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    Lx, Ly = grid.Lx, grid.Ly
    kx1, ky = 2π / Lx, 2π / Ly
    kx2 = 4π / Lx
    X = reshape(vec(grid.x), :, 1); Y = reshape(vec(grid.y), 1, :)
    # two divergence-free modes superposed (each ∂x ux_i + ∂y uy_i = 0)
    ux0 = sin.(kx1 * X) .* cos.(ky * Y) + sin.(kx2 * X) .* cos.(ky * Y)
    uy0 = -(kx1 / ky) .* cos.(kx1 * X) .* sin.(ky * Y) -
          (kx2 / ky) .* cos.(kx2 * X) .* sin.(ky * Y)
    A = 1.0
    t_final = 0.1
    ν = 0.0

    function run(dt)
        field = Field(grid, ComplexField(); ndims=2)
        n = NumModelRK4Imp(field, NavierStokesParameters(; ν=ν), dt,
                           Int(t_final / dt), 1)
        field.ux .= A * ux0
        field.uy .= A * uy0
        for _ in 1:Int(t_final / dt)
            SuperFluids.timeStep!(n)
        end
        return [parent(field.ux), parent(field.uy)]
    end

    ref = run(t_final / 1000)   # reference run (error ~128× smaller)
    e1 = run(t_final / 4)
    e2 = run(t_final / 8)
    e3 = run(t_final / 16)
    function nrm_err(e)
        sqrt(sum(abs.(e[1] .- ref[1]).^2) + sum(abs.(e[2] .- ref[2]).^2))
    end
    # the flow must actually evolve (nonlinear, ν=0): not a frozen vortex
    @test maximum(abs.(ref[1] .- A * ux0)) > 1e-3
    o1 = log(nrm_err(e1) / nrm_err(e2)) / log(2)
    o2 = log(nrm_err(e2) / nrm_err(e3)) / log(2)
    @test 3.8 < o1 < 4.2
    @test 3.8 < o2 < 4.2
end

# ==========================================================================
# Taylor-Green time integration (2D): energy strictly decays (dE/dt ≤ 0),
# the divergence stays at roundoff level, and there is no blow-up.
# ==========================================================================
@testset "NS Taylor-Green time integration 2D" begin
    grid = Grid((32, 24), ((-4, 4), (-5, 5)))
    field = Field(grid, ComplexField(); ndims=2)
    Δt, Nsteps = 0.02, 40
    n = NumModelRK4Imp(field, NavierStokesParameters(; ν=0.01), Δt, Nsteps, 1)
    taylor_green!(field, grid.x, grid.y)
    E0 = SuperFluids.energy(n)[2]
    Emax = E0
    for _ in 1:Nsteps
        SuperFluids.timeStep!(n)
        E = SuperFluids.energy(n)[2]
        Emax = max(Emax, E)
        # divergence of the *current* u_hat (kept by timeStep!) stays ~0
        @test div_max(n, n.u_hat) < 1e-8 * maximum(abs.(parent(n.u_hat[1])))
    end
    E = SuperFluids.energy(n)[2]
    @test E < E0          # viscosity damps the flow
    @test E > 0.5 * E0    # ... but not by an unphysical amount in 0.8s
    @test Emax < 1.01 * E0  # no blow-up
end

# ==========================================================================
# Coupled GP-NS two-fluid model (NSGP)
# ==========================================================================

# A regularised 2D quantum vortex psi = A(r) e^{i theta} with A(r) = r/sqrt(r^2+a^2)
# has superfluid velocity  u_s = 2|alpha| Im(psi_bar grad psi)/|psi|^2 = 2|alpha| r/(r^2+a^2) e_theta
# (finite core, -> 2|alpha|/r away from the core). Validates compute_u_adv_Fns!
# (regularised velocity, one-way counterflow, V_xm) end to end.
@testset "NSGP regularised vortex superfluid velocity (2D)" begin
    grid = Grid((64, 64), ((-8, 8), (-8, 8)))
    fgp = Field(grid, ComplexField(); ndims=2)
    fns = Field(grid, ComplexField(); ndims=2)
    a, α = 0.8, -0.02
    X = vec(grid.x); Y = vec(grid.y)
    X2 = reshape(X, :, 1); Y2 = reshape(Y, 1, :)
    r2 = X2.^2 .+ Y2.^2
    r  = sqrt.(r2)
    # regularised quantum vortex: psi = A(r) e^{i theta}, A = r/sqrt(r^2+a^2)
    fgp.ϕ .= (r ./ sqrt.(r2 .+ a^2)) .* exp.(1im * atan.(Y2, X2))
    p = NSGPParameters(; α=α, ν=0.0, β=1.0, ρn=0.5, ρs=0.5,
                       Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.05, one_way=true)
    n = NumModelNSGP(fgp, fns, p, 0.01, 1, 1; stepper="RK2Imp")
    SuperFluids.compute_u_adv_Fns!(n, n.phihat, n.u_hat)
    usx = real.(collect(n.us_phys[1])); usy = real.(collect(n.us_phys[2]))
    r_s = sqrt.(r2 .+ 1e-8)
    mag = sqrt.(usx .^ 2 .+ usy .^ 2)
    etx, ety = -Y2 ./ r_s, X2 ./ r_s          # e_theta
    # 1) the regularised core is finite and bounded (no 1/r singularity)
    @test all(isfinite.(usx)) && all(isfinite.(usy))
    @test maximum(mag[r .< 1.2a]) < 2 * abs(α) / a
    # 2) away from the core the velocity is azimuthal (rotational): u_s ~ |u| e_theta.
    # (checked in an annulus away from the core and the periodic corners)
    outer = (r .> 4a) .& (r .< 7a)
    cosang = sum(mag[outer] .* (usx[outer] .* etx[outer] .+ usy[outer] .* ety[outer])) /
             sum(mag[outer] .^ 2)
    @test cosang > 0.95
    # 3) the velocity is non-trivial there (it is a vortex, not a frozen field)
    @test maximum(mag[outer]) > 0.05 * (2 * abs(α) / a)
end

# A real GP field has u_s = 0 (no phase gradient). With one-way coupling the
# normal fluid then evolves as a (viscous) NS flow: it stays divergence-free,
# finite, and the GP mass is kept stable by the mass correction.
@testset "NSGP real GP field: u_s = 0 and stable one-way NS evolution (2D)" begin
    grid = Grid((48, 36), ((-4, 4), (-5, 5)))
    fgp = Field(grid, ComplexField(); ndims=2)
    fns = Field(grid, ComplexField(); ndims=2)
    @. fgp.ϕ = 1.0 + 0.5 * exp(-(fgp.x^2 + fgp.y^2))        # real -> u_s = 0
    kx = 2π / grid.Lx; ky = 2π / grid.Ly
    @. fns.ux = sin(ky * fns.y) * cos(kx * fns.x)
    @. fns.uy = -cos(ky * fns.y) * sin(kx * fns.x)
    p = NSGPParameters(; α=-0.01, ν=0.01, β=1.0, ρn=0.5, ρs=0.5,
                       Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.1, one_way=true)
    n = NumModelNSGP(fgp, fns, p, 0.005, 10, 1; stepper="RK2Imp")
    SuperFluids.compute_u_adv_Fns!(n, n.phihat, n.u_hat)
    @test maximum(abs.(real.(parent(n.us_phys[1])))) < 1e-10
    @test maximum(abs.(real.(parent(n.us_phys[2])))) < 1e-10
    N0 = sum(abs2.(parent(fgp.ϕ)))
    for _ in 1:10
        SuperFluids.timeStep!(n)
        @test all(isfinite.(parent(n.phihat)))
        @test all(isfinite.(parent(n.u_hat[1])))
    end
    g = SuperFluids.spectral_grid(n.plan)
    d = @. g.x * n.u_hat[1] + g.y * n.u_hat[2]
    @test maximum(abs.(parent(d))) < 1e-9 * maximum(abs.(parent(n.u_hat[1])))
    N = sum(abs2.(parent(fgp.ϕ)))
    @test abs(N - N0) < 1e-5 * N0
end

# Full two-way coupling in 2D: an initial superfluid vortex (with vorticity)
# and a background normal flow. The mutual-friction force is non-zero, both
# fields stay finite, and the normal fluid remains divergence-free.
@testset "NSGP two-way 2D coupled evolution stays finite and divergence-free" begin
    grid = Grid((48, 36), ((-4, 4), (-5, 5)))
    fgp = Field(grid, ComplexField(); ndims=2)
    fns = Field(grid, ComplexField(); ndims=2)
    a = 1.0
    X = reshape(vec(fgp.x), :, 1); Y = reshape(vec(fgp.y), 1, :)
    r2 = X.^2 .+ Y.^2
    r = sqrt.(r2)
    A = r ./ sqrt.(r2 .+ a^2)
    fgp.ϕ .= A .* exp.(1im * atan.(Y, X))
    kx = 2π / grid.Lx; ky = 2π / grid.Ly
    @. fns.ux = 0.3 * sin(ky * fns.y) * cos(kx * fns.x)
    @. fns.uy = -0.3 * cos(ky * fns.y) * sin(kx * fns.x)
    p = NSGPParameters(; α=-0.01, ν=0.01, β=1.0, ρn=0.5, ρs=0.5,
                       Btab=0.4, Bptab=0.1, ξ=1.0, ε2=0.1)
    n = NumModelNSGP(fgp, fns, p, 0.003, 8, 1; stepper="RK2Imp")
    # the mutual friction force is non-zero where the counterflow and
    # superfluid vorticity are both non-zero
    SuperFluids.compute_u_adv_Fns!(n, n.phihat, n.u_hat)
    @test maximum(abs.(real.(parent(n.fns_phys[1])))) > 1e-6
    @test maximum(abs.(real.(parent(n.fns_phys[2])))) > 1e-6
    for _ in 1:8
        SuperFluids.timeStep!(n)
        @test all(isfinite.(parent(n.phihat)))
        @test all(isfinite.(parent(n.u_hat[1])))
    end
    g = SuperFluids.spectral_grid(n.plan)
    d = @. g.x * n.u_hat[1] + g.y * n.u_hat[2]
    @test maximum(abs.(parent(d))) < 1e-9 * maximum(abs.(parent(n.u_hat[1])))
end

# ==========================================================================
# HVBK linear two-fluid model
# ==========================================================================

# The linear mutual friction F = -1/2 rb |w_s| (u_n - u_s) is an internal force:
# the TOTAL momentum rho_n u_n + rho_s u_s must be conserved, both fields must
# stay divergence-free, and the friction (plus viscosity) must dissipate energy.
@testset "HVBK total momentum conservation, div-free, energy decay (2D)" begin
    grid = Grid((48, 36), ((-4, 4), (-5, 5)))
    fn = Field(grid, ComplexField(); ndims=2)
    fs = Field(grid, ComplexField(); ndims=2)
    kx = 2π / grid.Lx; ky = 2π / grid.Ly
    @. fn.ux = sin(ky * fn.y) * cos(kx * fn.x)
    @. fn.uy = -cos(ky * fn.y) * sin(kx * fn.x)
    @. fs.ux = 0.7 * cos(ky * fs.y) * sin(kx * fs.x)
    @. fs.uy = 0.7 * sin(ky * fs.y) * cos(kx * fs.x)
    p = HBVKParameters(; ν=0.01, νs=0.001, rb=1.5, ρn=1.0, ρs=1.0)
    n = NumModelHBVK(fn, fs, p, 0.01, 40, 1; stepper="RK2")
    mom() = [p.ρn * real(sum(parent(fn.u[c]))) + p.ρs * real(sum(parent(fs.u[c]))) for c in 1:2]
    P0 = mom()
    # natural momentum scale (the conserved k=0 mode is ~0, so normalise by this)
    Pscale = (grid.n[1] * grid.Δx * grid.Δy) * (maximum(abs.(parent(fn.ux))) + maximum(abs.(parent(fs.ux))))
    E0 = SuperFluids.energy(n)[4]
    Pmax = 0.0
    for _ in 1:40
        SuperFluids.timeStep!(n)
        Pmax = max(Pmax, maximum(abs.(mom() .- P0)))
        g = SuperFluids.spectral_grid(n.plan)
        @test maximum(abs.(parent(@. g.x * n.un_hat[1] + g.y * n.un_hat[2]))) <
              1e-9 * maximum(abs.(parent(n.un_hat[1])))
    end
    E1 = SuperFluids.energy(n)[4]
    @test Pmax < 1e-13 * Pscale                 # total momentum conserved
    @test E1 < E0                                # friction dissipates
    @test E1 > 0.3 * E0                          # ... but not unphysically
end

# With no mutual friction (rb = 0) and a superfluid at rest, nothing drives the
# superfluid: it must stay (numerically) at rest, while the normal fluid
# evolves as a standalone (viscous) NS flow.
@testset "HVBK zero-friction: rest superfluid stays at rest (2D)" begin
    grid = Grid((48, 36), ((-4, 4), (-5, 5)))
    fn = Field(grid, ComplexField(); ndims=2)
    fs = Field(grid, ComplexField(); ndims=2)
    kx = 2π / grid.Lx; ky = 2π / grid.Ly
    @. fn.ux = sin(ky * fn.y) * cos(kx * fn.x)
    @. fn.uy = -cos(ky * fn.y) * sin(kx * fn.x)
    @. fs.ux = 0.0
    @. fs.uy = 0.0
    p = HBVKParameters(; ν=0.01, νs=0.01, rb=0.0, ρn=1.0, ρs=1.0)
    n = NumModelHBVK(fn, fs, p, 0.01, 20, 1; stepper="RK2")
    for _ in 1:20
        SuperFluids.timeStep!(n)
    end
    @test maximum(abs.(parent(fs.ux))) < 1e-12
    @test maximum(abs.(parent(fs.uy))) < 1e-12
    # the normal fluid still evolved (nonzero)
    @test maximum(abs.(parent(fn.ux))) > 0.1
end

# 3D: both steppers stay finite and conserve the total momentum.
@testset "HVBK 3D evolution finite, momentum-conserving" begin
    grid = Grid((20, 16, 20), ((-3, 3), (-4, 4), (-3, 3)))
    fn = Field(grid, ComplexField(); ndims=3)
    fs = Field(grid, ComplexField(); ndims=3)
    kx, ky, kz = 2π / grid.Lx, 2π / grid.Ly, 2π / grid.Lz
    @. fn.ux = sin(ky * fn.y) * cos(kx * fn.x)
    @. fn.uy = -cos(ky * fn.y) * sin(kx * fn.x)
    @. fn.uz = 0.0
    @. fs.ux = 0.7 * cos(ky * fs.y) * sin(kx * fs.x)
    @. fs.uy = 0.7 * sin(ky * fs.y) * cos(kx * fs.x)
    @. fs.uz = 0.3 * sin(kz * fs.z)
    p = HBVKParameters(; ν=0.01, νs=0.001, rb=1.5, ρn=1.0, ρs=1.0)
    for stepper in ("RK1", "RK2")
        n = NumModelHBVK(fn, fs, p, 0.008, 6, 1; stepper=stepper)
        mom() = [p.ρn * real(sum(parent(fn.u[c]))) + p.ρs * real(sum(parent(fs.u[c]))) for c in 1:3]
        P0 = mom()
        Pscale = (grid.n[1] * grid.Δx * grid.Δy * grid.Δz) * (maximum(abs.(parent(fn.ux))) + maximum(abs.(parent(fs.ux))))
        Pmax = 0.0
        for _ in 1:6
            SuperFluids.timeStep!(n)
            Pmax = max(Pmax, maximum(abs.(mom() .- P0)))
            @test all(isfinite.(parent(n.un_hat[1])))
            @test all(isfinite.(parent(n.us_hat[1])))
        end
        @test Pmax < 1e-12 * Pscale
    end
end
