using Test
using SuperFluids

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
