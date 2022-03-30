# FFT

## 2D
function computeDerivatives!(gf :: GradientField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # FFT x
   # frequencies
   x, y, ξx, ξy = grid_x(p)
   mul_x!(p.ϕx_hat, p, ϕt)
   # compute dx
   p.ϕxtmp_hat .= im .* ξx .* p.ϕx_hat
   ldiv_x!(gf.dx, p, p.ϕxtmp_hat)
   # compute ddx
   p.ϕxtmp_hat .= - ξx.^2 .* p.ϕx_hat
   ldiv_x!(gf.ddx, p, p.ϕxtmp_hat)
   # FFT y
   x, y, ξx, ξy = grid_y(p)
   mul_y!(p.ϕy_hat, plan, ϕt)
   # compute dy
   p.ϕytmp_hat .= im .* ξy .* p.ϕy_hat
   ldiv_y!(gf.dy, p, p.ϕytmp_hat)
   # compute ddy
   p.ϕytmp_hat .= - ξy.^2 .* p.ϕy_hat
   ldiv_y!(gf.ddy, p, p.ϕytmp_hat)
   return nothing
end

function computeDerivatives!(gf :: GradientRotField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # FFT x
   # frequencies
   x, y, ξx, ξy = grid_x(p)
   mul_x!(p.ϕx_hat, p, ϕt)
   # compute dx
   p.a_tmp2x .= im .* ξx .* p.ϕx_hat
   ldiv_x!(gf.dx, p, p.a_tmp2x)
   # compute rx
   p.a_tmp2x .= im .* y .* ξx .* p.ϕx_hat
   ldiv_x!(gf.rx, p, p.a_tmp2x)
   # compute ddx
   p.a_tmp2x .= - ξx.^2 .* p.ϕx_hat
   ldiv_x!(gf.ddx, p, p.a_tmp2x)
   # FFT y
   x, y, ξx, ξy = grid_y(p)
   mul_y!(p.ϕy_hat, p, ϕt)
   # compute dy
   p.a_tmp2y .= im .* ξy .* p.ϕy_hat
   ldiv_y!(gf.dy, p, p.a_tmp2y)
   # compute ry
   p.a_tmp2y .= -im .* x .* ξy .* p.ϕy_hat
   ldiv_y!(gf.ry, p, p.a_tmp2y)
   # compute ddy
   p.a_tmp2y .= - ξy.^2 .* p.ϕy_hat
   ldiv_y!(gf.ddy, p, p.a_tmp2y)
   return nothing
end

## 3D

function computeDerivatives!(gf :: GradientField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # FFT x
   # frequencies
   x, y, z, ξx, ξy, ξz = grid_x(p)
   mul_x!(p.ϕx_hat, p, ϕt)
   # compute dx
   p.a_tmp2x .= im .* ξx .* p.ϕx_hat
   ldiv_x!(gf.dx, p, p.a_tmp2x)
   # compute ddx
   p.a_tmp2x .= - ξx.^2 .* p.ϕx_hat
   ldiv_x!(gf.ddx, p, p.a_tmp2x)
   # FFT y
   x, y, z, ξx, ξy, ξz = grid_y(p)
   mul_y!(p.ϕy_hat, p, ϕt)
   # compute dy
   p.a_tmp2y .= im .* ξy .* p.ϕy_hat
   ldiv_y!(gf.dy, p, p.a_tmp2y)
   # compute ddy
   p.a_tmp2y .= - ξy.^2 .* p.ϕy_hat
   ldiv_y!(gf.ddy, p, p.a_tmp2y)
   # FFT z
   x, y, z, ξx, ξy, ξz = grid_z(p)
   mul_z!(p.ϕz_hat, p, ϕt)
   # compute dz
   p.ϕztmp_hat .= im .* ξz .* p.ϕz_hat
   ldiv_z!(gf.dz, p, p.ϕztmp_hat)
   # compute ddz
   p.a_tmp2z .= - ξy.^2 .* p.ϕz_hat
   ldiv_y!(gf.ddz, p, p.a_tmp2z)
   return nothing
end

function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # FFT x
   # frequencies
   x, y, z, ξx, ξy, ξz = grid_x(p)
   mul_x!(p.ϕx_hat, p, ϕt)
   # compute dx
   p.a_tmp2x .= im .* ξx .* p.ϕx_hat
   ldiv_x!(gf.dx, p, p.a_tmp2x)
   # compute rx
   p.a_tmp2x .= im .* y .* ξx .* p.ϕx_hat
   ldiv_x!(gf.rx, p, p.a_tmp2x)
   # compute ddx
   p.a_tmp2x .= - ξx.^2 .* p.ϕx_hat
   ldiv_x!(gf.ddx, p, p.a_tmp2x)
   # FFT y
   x, y, z, ξx, ξy, ξz = grid_y(p)
   mul_y!(p.ϕy_hat, p, ϕt)
   # compute dy
   p.a_tmp2y .= im .* ξy .* p.ϕy_hat
   ldiv_y!(gf.dy, p, p.a_tmp2y)
   # compute ry
   p.a_tmp2y .= -im .* x .* ξy .* p.ϕy_hat
   ldiv_y!(gf.ry, p, p.a_tmp2y)
   # compute ddy
   p.a_tmp2y .= - ξy.^2 .* p.ϕy_hat
   ldiv_y!(gf.ddy, p, p.a_tmp2y)
   # FFT z
   x, y, z, ξx, ξy, ξz = grid_z(p)
   mul_z!(p.ϕz_hat, p, ϕt)
   # compute dz
   p.a_tmp2z .= im .* ξz .* p.ϕz_hat
   ldiv_z!(gf.dz, p, p.a_tmp2z)
   # compute ddz
   p.a_tmp2z .= - ξz.^2 .* p.ϕz_hat
   ldiv_z!(gf.ddz, p, p.a_tmp2z)
   return nothing
end

function computeDerivatives!(gf :: GradientCurlField3D, p :: AbstractFFTPlan, ut :: AbstractArray)
   # get ̂u
   mul_all!(p.uz_hat, p, ut)
   # new field for ̂ω
   ω_hat = similar_data(p.uz_hat)
   x, y, z, ξx, ξy, ξz = grid_z(p)
   # ̂ω  = ∇ × ̂u
   @. ω_hat[1] = im * (ξy * uz_hat[3] - ξz * uz_hat[2])
   @. ω_hat[2] = im * (ξz * uz_hat[1] - ξx * uz_hat[3])
   @. ω_hat[3] = im * (ξx * uz_hat[2] - ξy * uz_hat[1])
   # get ω with iFFT
   ldiv_all!(gf.ω, p, ω_hat)
   return nothing
end

# Finite Difference

## 2D
function computeDerivatives!(gf :: GradientField2D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # x direction
   computedxddx!(ϕt, gf.dx, gf.ddx, p.Δx, order=6)
   # y direction
   transpose!(p.ϕytmp,ϕt)
   dϕytmp = similar(p.ϕytmp)
   ddϕytmp = similar(p.ϕytmp)
   computedyddy!(p.ϕytmp, dϕytmp, ddϕytmp, p.Δy, order=6)
   transpose!(gf.dy,dϕytmp)
   transpose!(gf.ddy,ddϕytmp)
   return nothing
end

function computeDerivatives!(gf :: GradientRotField2D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
   x, y = grid.x, grid.y
   # x direction
   computedxddx!(ϕt, gf.dx, gf.ddx, p.Δx, order=6)
   gf.rx .= y .* gf.dx
   # y direction
   transpose!(p.ϕytmp,ϕt)
   dϕytmp = similar(p.ϕytmp)
   ddϕytmp = similar(p.ϕytmp)
   computedyddy!(p.ϕytmp, dϕytmp, ddϕytmp, p.Δy, order=6)
   transpose!(gf.dy,dϕytmp)
   transpose!(gf.ddy,ddϕytmp)
   dϕytmp = nothing
   ddϕytmp = nothing
   # compute (in pen_x)
   gf.ry = -x .* gf.dy
   return nothing
end

## 3D

function computeDerivatives!(gf :: GradientField3D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # x direction
   computedxddx!(ϕt, gf.dx, gf.ddx, p.Δx, order=6)
   # y direction
   transpose!(p.ϕytmp,ϕt)
   dϕytmp = similar(p.ϕytmp)
   ddϕytmp = similar(p.ϕytmp)
   computedyddy!(p.ϕytmp, dϕytmp, ddϕytmp, p.Δy, order=6)
   transpose!(gf.dy,dϕytmp)
   transpose!(gf.ddy,ddϕytmp)
   # z direction
   transpose!(p.ϕztmp,p.ϕytmp)
   dϕztmp = similar(p.ϕztmp)
   ddϕztmp = similar(p.ϕztmp)
   computedyddy!(p.ϕytmp, dϕytmp, ddϕytmp, p.Δy, order=6)
   transpose!(dϕytmp,dϕztmp)
   transpose!(ddϕytmp,ddϕztmp)
   transpose!(gf.dz,dϕytmp)
   transpose!(gf.ddz,ddϕytmp)
   # deallocate
   dϕztmp = nothing
   ddϕztmp = nothing
   dϕytmp = nothing
   ddϕytmp = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # x direction
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y = grid.x, grid.y
   computedxddx!(ϕt, gf.dx, gf.ddx, p.Δx, order=6)
   gf.rx .= y .* gf.dx
   # y direction
   grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y = grid.x, grid.y
   transpose!(p.ϕytmp,ϕt)
   dϕytmp = similar(p.ϕytmp)
   ddϕytmp = similar(p.ϕytmp)
   computedyddy!(p.ϕytmp, dϕytmp, ddϕytmp, p.Δy, order=6)
   transpose!(gf.dy,dϕytmp)
   transpose!(gf.ddy,ddϕytmp)
   gf.ry = -x .* gf.dy
   # z direction
   transpose!(p.ϕztmp,p.ϕytmp)
   dϕztmp = similar(p.ϕztmp)
   ddϕztmp = similar(p.ϕztmp)
   computedzddz!(p.ϕztmp, dϕztmp, ddϕztmp, p.Δz, order=6)
   transpose!(dϕytmp,dϕztmp)
   transpose!(ddϕytmp,ddϕztmp)
   transpose!(gf.dz,dϕytmp)
   transpose!(gf.ddz,ddϕytmp)
   # deallocate
   dϕztmp = nothing
   ddϕztmp = nothing
   dϕytmp = nothing
   ddϕytmp = nothing
   return nothing
end

# Finite difference helper functions
function computedxddx!(ϕt :: AbstractArray{A,2}, dx :: AbstractArray{A,2}, ddx :: AbstractArray{A,2}, Δx :: Real; order :: Integer = 2) where A
   N = size(ϕt)
   nx = N[1]
   if order == 2
      for i = 1:nx
         dx[i,:] = (ϕt[ i%nx + 1,:] - ϕt[ mod(i%nx-2,nx) + 1,:]) / (2*Δx)
         ddx[i,:] = (-2 * ϕt[i,:] + ϕt[ i%nx + 1,:] + ϕt[ mod(i%nx-2,nx) + 1,:]) / Δx^2
      end
   elseif order == 4
      for i = 1:nx
         dx[i,:] = (-ϕt[ (i+1)%nx + 1,:] + 8*ϕt[ i%nx + 1,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:] + ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*Δx)
         ddx[i,:] = (-30 * ϕt[i,:] - ϕt[ (i+1)%nx + 1,:] + 16*ϕt[ i%nx + 1,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:] - ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*Δx^2)
      end
   elseif order == 6
      for i = 1:nx
         dx[i,:] = (ϕt[ (i+2)%nx + 1,:] - 9*ϕt[ (i+1)%nx + 1,:] + 45*ϕt[ i%nx + 1,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:] - ϕt[ mod(i%nx-4,nx) + 1,:]) / (60*Δx)
         ddx[i,:] = (-490 * ϕt[i,:] + 2*ϕt[ (i+2)%nx + 1,:] - 27*ϕt[ (i+1)%nx + 1,:] + 270*ϕt[ i%nx + 1,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:]) / (180*Δx^2)
      end
   elseif order == 8
      for i = 1:nx
         dx[i,:] = (-3*ϕt[ (i+3)%nx + 1,:] + 32*ϕt[ (i+2)%nx + 1,:] - 168*ϕt[ (i+1)%nx + 1,:] + 672*ϕt[ i%nx + 1,:]
            - 672*ϕt[ mod(i%nx-2,nx) + 1,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:]) / (840*Δx)
         ddx[i,:] = (-14350 * ϕt[i,:] - 9*ϕt[ (i+3)%nx + 1,:] + 128*ϕt[ (i+2)%nx + 1,:] - 1008*ϕt[ (i+1)%nx + 1,:] + 8064*ϕt[ i%nx + 1,:]
            + 8064*ϕt[ mod(i%nx-2,nx) + 1,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:]) / (5040*Δx^2)
      end
   end
   return nothing
end

function computedxddx!(ϕt :: AbstractArray{A,3}, dx :: AbstractArray{A,3}, ddx :: AbstractArray{A,3}, Δx :: Real; order :: Integer = 2) where A
   N = size(ϕt)
   nx = N[1]
   if order == 2
      for i = 1:nx
         dx[i,:,:] = (ϕt[ i%nx + 1,:,:] - ϕt[ mod(i%nx-2,nx) + 1,:,:]) / (2*Δx)
         ddx[i,:,:] = (-2 * ϕt[i,:,:] + ϕt[ i%nx + 1,:,:] + ϕt[ mod(i%nx-2,nx) + 1,:,:]) / Δx^2
      end
   elseif order == 4
      for i = 1:nx
         dx[i,:,:] = (-ϕt[ (i+1)%nx + 1,:,:] + 8*ϕt[ i%nx + 1,:,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:,:] + ϕt[ mod(i%nx-3,nx) + 1,:,:]) / (12*Δx)
         ddx[i,:,:] = (-30 * ϕt[i,:,:] - ϕt[ (i+1)%nx + 1,:,:] + 16*ϕt[ i%nx + 1,:,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:,:] - ϕt[ mod(i%nx-3,nx) + 1,:,:]) / (12*Δx^2)
      end
   elseif order == 6
      for i = 1:nx
         dx[i,:,:] = (ϕt[ (i+2)%nx + 1,:,:] - 9*ϕt[ (i+1)%nx + 1,:,:] + 45*ϕt[ i%nx + 1,:,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:,:] - ϕt[ mod(i%nx-4,nx) + 1,:,:]) / (60*Δx)
         ddx[i,:,:] = (-490 * ϕt[i,:,:] + 2*ϕt[ (i+2)%nx + 1,:,:] - 27*ϕt[ (i+1)%nx + 1,:,:] + 270*ϕt[ i%nx + 1,:,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:,:]) / (180*Δx^2)
      end
   elseif order == 8
      for i = 1:nx
         dx[i,:,:] = (-3*ϕt[ (i+3)%nx + 1,:,:] + 32*ϕt[ (i+2)%nx + 1,:,:] - 168*ϕt[ (i+1)%nx + 1,:,:] + 672*ϕt[ i%nx + 1,:,:]
            - 672*ϕt[ mod(i%nx-2,nx) + 1,:,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:,:]) / (840*Δx)
         ddx[i,:,:] = (-14350 * ϕt[i,:,:] - 9*ϕt[ (i+3)%nx + 1,:,:] + 128*ϕt[ (i+2)%nx + 1,:,:] - 1008*ϕt[ (i+1)%nx + 1,:,:] + 8064*ϕt[ i%nx + 1,:,:]
            + 8064*ϕt[ mod(i%nx-2,nx) + 1,:,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:,:]) / (5040*Δx^2)
      end
   end
   return nothing
end

function computedyddy!(ϕt :: AbstractArray{A,2}, dy :: AbstractArray{A,2}, ddy :: AbstractArray{A,2}, Δy :: Real; order :: Integer = 2) where A
   N = size(ϕt)
   ny = N[2]
   if order == 2
      for j = 1:ny
         dy[:,j] = (ϕt[ :, j%ny + 1] - ϕt[ :, mod(j%ny-2,ny) + 1]) / (2*Δy)
         ddy[:,j] = (-2 * ϕt[:,j] + ϕt[ :, j%ny + 1] + ϕt[ :, mod(j%ny-2,ny) + 1]) / Δy^2
      end
   elseif order == 4
      for j = 1:ny
         dy[:,j] = (-ϕt[:, (j+1)%ny + 1] + 8*ϕt[:, j%ny + 1] - 8*ϕt[:, mod(j%ny-2,ny) + 1] + ϕt[:, mod(j%ny-3,ny) + 1]) / (12*Δy)
         ddy[:,j] = (-30 * ϕt[:,j] - ϕt[:, (j+1)%ny + 1] + 16*ϕt[:, j%ny + 1] + 16*ϕt[:, mod(j%ny-2,ny) + 1] - ϕt[:, mod(j%ny-3,ny) + 1]) / (12*Δy^2)
      end
   elseif order == 6
      for j = 1:ny
         dy[:,j] = (ϕt[:, (j+2)%ny + 1] - 9*ϕt[:, (j+1)%ny + 1] + 45*ϕt[:, j%ny + 1] - 45*ϕt[:, mod(j%ny-2,ny) + 1] + 9*ϕt[:, mod(j%ny-3,ny) + 1] - ϕt[:, mod(j%ny-4,ny) + 1]) / (60*Δy)
         ddy[:,j] = (-490 * ϕt[:,j] + 2*ϕt[:, (j+2)%ny + 1] - 27*ϕt[:, (j+1)%ny + 1] + 270*ϕt[:, j%ny + 1] + 270*ϕt[:, mod(j%ny-2,ny) + 1] - 27*ϕt[:, mod(j%ny-3,ny) + 1] + 2*ϕt[:, mod(j%ny-4,ny) + 1]) / (180*Δy^2)
      end
   elseif order == 8
      for j = 1:ny
         dy[:,j] = (-3*ϕt[:, (j+3)%ny + 1] + 32*ϕt[:, (j+2)%ny + 1] - 168*ϕt[:, (j+1)%ny + 1] + 672*ϕt[:, j%ny + 1]
            - 672*ϕt[:, mod(j%ny-2,ny) + 1] + 168*ϕt[:, mod(j%ny-3,ny) + 1] - 32*ϕt[:, mod(j%ny-4,ny) + 1] + 3*ϕt[:, mod(j%ny-5,ny) + 1]) / (840*Δy)
         ddy[:,j] = (-14350 * ϕt[:,j] - 9*ϕt[:, (j+3)%ny + 1] + 128*ϕt[:, (j+2)%ny + 1] - 1008*ϕt[:, (j+1)%ny + 1] + 8064*ϕt[:, j%ny + 1]
            + 8064*ϕt[:, mod(j%ny-2,ny) + 1] - 1008*ϕt[:, mod(j%ny-3,ny) + 1] + 128*ϕt[:, mod(j%ny-4,ny) + 1] - 9*ϕt[:, mod(j%ny-5,ny) + 1]) / (5040*Δy^2)
      end
   end
   return nothing
end

function computedyddy!(ϕt :: AbstractArray{A,3}, dy :: AbstractArray{A,3}, ddy :: AbstractArray{A,3}, Δy :: Real; order :: Integer = 2) where A
   N = size(ϕt)
   ny = N[2]
   if order == 2
      for j = 1:ny
         dy[:,j,:] = (ϕt[ :, j%ny + 1,:] - ϕt[ :, mod(j%ny-2,ny) + 1,:]) / (2*Δy)
         ddy[:,j,:] = (-2 * ϕt[:,j,:] + ϕt[ :, j%ny + 1,:] + ϕt[ :, mod(j%ny-2,ny) + 1,:]) / Δy^2
      end
   elseif order == 4
      for j = 1:ny
         dy[:,j,:] = (-ϕt[:, (j+1)%ny + 1,:] + 8*ϕt[:, j%ny + 1,:] - 8*ϕt[:, mod(j%ny-2,ny) + 1,:] + ϕt[:, mod(j%ny-3,ny) + 1,:]) / (12*Δy)
         ddy[:,j,:] = (-30 * ϕt[:,j,:] - ϕt[:, (j+1)%ny + 1,:] + 16*ϕt[:, j%ny + 1,:] + 16*ϕt[:, mod(j%ny-2,ny) + 1,:] - ϕt[:, mod(j%ny-3,ny) + 1,:]) / (12*Δy^2)
      end
   elseif order == 6
      for j = 1:ny
         dy[:,j,:] = (ϕt[:, (j+2)%ny + 1,:] - 9*ϕt[:, (j+1)%ny + 1,:] + 45*ϕt[:, j%ny + 1,:] - 45*ϕt[:, mod(j%ny-2,ny) + 1,:] + 9*ϕt[:, mod(j%ny-3,ny) + 1,:] - ϕt[:, mod(j%ny-4,ny) + 1,:]) / (60*Δy)
         ddy[:,j,:] = (-490 * ϕt[:,j,:] + 2*ϕt[:, (j+2)%ny + 1,:] - 27*ϕt[:, (j+1)%ny + 1,:] + 270*ϕt[:, j%ny + 1,:] + 270*ϕt[:, mod(j%ny-2,ny) + 1,:] - 27*ϕt[:, mod(j%ny-3,ny) + 1,:] + 2*ϕt[:, mod(j%ny-4,ny) + 1,:]) / (180*Δy^2)
      end
   elseif order == 8
      for j = 1:ny
         dy[:,j,:] = (-3*ϕt[:, (j+3)%ny + 1,:] + 32*ϕt[:, (j+2)%ny + 1,:] - 168*ϕt[:, (j+1)%ny + 1,:] + 672*ϕt[:, j%ny + 1,:]
            - 672*ϕt[:, mod(j%ny-2,ny) + 1,:] + 168*ϕt[:, mod(j%ny-3,ny) + 1,:] - 32*ϕt[:, mod(j%ny-4,ny) + 1,:] + 3*ϕt[:, mod(j%ny-5,ny) + 1,:]) / (840*Δy)
         ddy[:,j,:] = (-14350 * ϕt[:,j,:] - 9*ϕt[:, (j+3)%ny + 1,:] + 128*ϕt[:, (j+2)%ny + 1,:] - 1008*ϕt[:, (j+1)%ny + 1,:] + 8064*ϕt[:, j%ny + 1,:]
            + 8064*ϕt[:, mod(j%ny-2,ny) + 1,:] - 1008*ϕt[:, mod(j%ny-3,ny) + 1,:] + 128*ϕt[:, mod(j%ny-4,ny) + 1,:] - 9*ϕt[:, mod(j%ny-5,ny) + 1,:]) / (5040*Δy^2)
      end
   end
   return nothing
end

function computedzddz!(ϕt :: AbstractArray{A,3}, dz :: AbstractArray{A,3}, ddz :: AbstractArray{A,3}, Δz :: Real; order :: Integer = 2) where A
   N = size(ϕt)
   nz = N[3]
   if order == 2
      for k = 1:nz
         dz[:,k,:] = (ϕt[ :, k%nz + 1,:] - ϕt[ :, mod(k%nz-2,nz) + 1,:]) / (2*Δz)
         ddz[:,k,:] = (-2 * ϕt[:,k,:] + ϕt[ :, k%nz + 1,:] + ϕt[ :, mod(k%nz-2,nz) + 1,:]) / Δz^2
      end
   elseif order == 4
      for k = 1:nz
         dz[:,k,:] = (-ϕt[:, (k+1)%nz + 1,:] + 8*ϕt[:, k%nz + 1,:] - 8*ϕt[:, mod(k%nz-2,nz) + 1,:] + ϕt[:, mod(k%nz-3,nz) + 1,:]) / (12*Δz)
         ddz[:,k,:] = (-30 * ϕt[:,k,:] - ϕt[:, (k+1)%nz + 1,:] + 16*ϕt[:, k%nz + 1,:] + 16*ϕt[:, mod(k%nz-2,nz) + 1,:] - ϕt[:, mod(k%nz-3,nz) + 1,:]) / (12*Δz^2)
      end
   elseif order == 6
      for k = 1:nz
         dz[:,k,:] = (ϕt[:, (k+2)%nz + 1,:] - 9*ϕt[:, (k+1)%nz + 1,:] + 45*ϕt[:, k%nz + 1,:] - 45*ϕt[:, mod(k%nz-2,nz) + 1,:] + 9*ϕt[:, mod(k%nz-3,nz) + 1,:] - ϕt[:, mod(k%nz-4,nz) + 1,:]) / (60*Δz)
         ddz[:,k,:] = (-490 * ϕt[:,k,:] + 2*ϕt[:, (k+2)%nz + 1,:] - 27*ϕt[:, (k+1)%nz + 1,:] + 270*ϕt[:, k%nz + 1,:] + 270*ϕt[:, mod(k%nz-2,nz) + 1,:] - 27*ϕt[:, mod(k%nz-3,nz) + 1,:] + 2*ϕt[:, mod(k%nz-4,nz) + 1,:]) / (180*Δz^2)
      end
   elseif order == 8
      for k = 1:nz
         dz[:,k,:] = (-3*ϕt[:, (k+3)%nz + 1,:] + 32*ϕt[:, (k+2)%nz + 1,:] - 168*ϕt[:, (k+1)%nz + 1,:] + 672*ϕt[:, k%nz + 1,:]
            - 672*ϕt[:, mod(k%nz-2,nz) + 1,:] + 168*ϕt[:, mod(k%nz-3,nz) + 1,:] - 32*ϕt[:, mod(k%nz-4,nz) + 1,:] + 3*ϕt[:, mod(k%nz-5,nz) + 1,:]) / (840*Δz)
         ddz[:,k,:] = (-14350 * ϕt[:,k,:] - 9*ϕt[:, (k+3)%nz + 1,:] + 128*ϕt[:, (k+2)%nz + 1,:] - 1008*ϕt[:, (k+1)%nz + 1,:] + 8064*ϕt[:, k%nz + 1,:]
            + 8064*ϕt[:, mod(k%nz-2,nz) + 1,:] - 1008*ϕt[:, mod(k%nz-3,nz) + 1,:] + 128*ϕt[:, mod(k%nz-4,nz) + 1,:] - 9*ϕt[:, mod(k%nz-5,nz) + 1,:]) / (5040*Δz^2)
      end
   end
   return nothing
end