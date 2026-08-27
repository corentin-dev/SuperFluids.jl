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
