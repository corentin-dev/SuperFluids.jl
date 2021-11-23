using SuperFluids

# simulation parameters
nx = 128
ny = 128
nz = 128

xrange = (-12, 12)
yrange = (-12, 12)
zrange = (-12, 12)

device = SuperFluids.MPI2D()
#if device.rank==0
#    println(device)
#end

# creating a grid
grid = Grid((nx,ny,nz), (xrange,yrange,zrange), device=device)
#println(grid)


pen_x = SuperFluids.Pencil(device.topo, (nx,ny,nz), (2,3))
ϕpx = SuperFluids.PencilArray{Complex{Float64}}(undef, pen_x)
ϕg = SuperFluids.global_view(ϕpx)
r = axes(ϕg)
println("$(device.rank) size:$(size(ϕpx)) sizeg:$(size(ϕg)) r:$(r)")


# create plan for FFT
# plan = PencilFFTPlan(ϕ₀, Transforms.FFT())
# plan_x = PencilFFTPlan(ϕ₀, (Transforms.FFT(),Transforms.NoTransform(),Transforms.NoTransform()))
# plan_y = PencilFFTPlan(ϕ₀, (Transforms.NoTransform(),Transforms.FFT(),Transforms.NoTransform()))
# plan_z = PencilFFTPlan(ϕ₀, (Transforms.NoTransform(),Transforms.NoTransform(),Transforms.FFT()))
# temporary arrays
# ϕhat = allocate_output(plan)
# ϕhat_x = allocate_output(plan_x)
# ϕhat_y = allocate_output(plan_y)
# ϕhat_z = allocate_output(plan_z)
# @assert get_permutation(ϕhat_x) === Permutation(3, 2, 1)
# @assert get_permutation(ϕhat_y) === Permutation(3, 2, 1)
# @assert get_permutation(ϕhat_z) === Permutation(3, 2, 1)
# global views
# ϕ_glob = global_view(ϕ₀)
# ϕhat_glob = global_view(ϕhat)
# # ranges
# rx,ry,rz = axes(ϕ_glob)
# rx_hat,ry_hat,rz_hat = axes(ϕhat_glob)
# # local size
# nxl,nyl,nzl = size(ϕ₀)
# nxl_hat,nyl_hat,nzl_hat = size(ϕhat)
# # for broadcasting
# X = reshape(x[rx],nxl,1,1)
# Y = reshape(y[ry],1,nyl,1)
# Z = reshape(z[rz],1,1,nzl)
# X_hat = reshape(x[rx_hat],nxl_hat,1,1)
# Y_hat = reshape(y[ry_hat],1,nyl_hat,1)
# Z_hat = reshape(z[rz_hat],1,1,nzl_hat)
# Ξx = reshape(ξx[rx_hat],nxl_hat,1,1)
# Ξy = reshape(ξy[ry_hat],1,nyl_hat,1)
# Ξz = reshape(ξz[rz_hat],1,1,nzl_hat)




# # allocating a field
# field = Field(grid, ComplexField())
# println(field)
nothing