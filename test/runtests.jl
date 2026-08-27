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
# spectral divergence of a velocity given as a Vector{PencilArray} in pen_z
# layout:  div = max |i k · û|.
function div_max(n, u_hat)
    gridξ = localgrid(n.plan.pen_z, (n.plan.ξx, n.plan.ξy, n.plan.ξz))
    d = imag.(gridξ.x .* u_hat[1] + gridξ.y .* u_hat[2] + gridξ.z .* u_hat[3])
    return maximum(abs.(d))
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
