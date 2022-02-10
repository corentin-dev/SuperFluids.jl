# FFT

## 2D
function computeDerivatives!(gf :: GradientField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # temporary fields
   ϕxthat = p.ϕx_hat
   ϕythat = p.ϕy_hat
   ϕxtmphat = p.ϕxtmp_hat
   ϕytmphat = p.ϕytmp_hat
   # plans
   plan_x, plan_y = p.plan_x, p.plan_y
   # FFT x
   # frequencies
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(p.pen_x, (p.ξx, p.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   mul!(parent(ϕxthat), plan_x, parent(ϕt))
   # compute dx
   ϕxtmphat .= im .* ξx .* ϕxthat
   ldiv!(parent(gf.dx), plan_x, parent(ϕxtmphat))
   # compute ddx
   ϕxtmphat .= - ξx.^2 .* ϕxthat
   ldiv!(parent(gf.ddx), plan_x, parent(ϕxtmphat))
   # FFT y
   grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(p.pen_y, (p.ξx, p.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   tmp_y = similar(ϕythat)
   transpose!(tmp_y, ϕt)
   mul!(parent(ϕythat), plan_y, parent(tmp_y))
   # compute dy
   ϕytmphat .= im .* ξy .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.dy,tmp_y)
   # compute ddy
   ϕytmphat .= - ξy.^2 .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.ddy,tmp_y)
   tmp_y = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # temporary fields
   ϕxthat = p.ϕx_hat
   ϕythat = p.ϕy_hat
   ϕxtmphat = p.ϕxtmp_hat
   ϕytmphat = p.ϕytmp_hat
   # plans
   plan_x, plan_y = p.plan_x, p.plan_y
   # FFT x
   # frequencies
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(p.pen_x, (p.ξx, p.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   mul!(parent(ϕxthat), plan_x, parent(ϕt))
   # compute dx
   ϕxtmphat .= im .* ξx .* ϕxthat
   ldiv!(parent(gf.dx), plan_x, parent(ϕxtmphat))
   # compute rx
   ϕxtmphat .= im .* y .* ξx .* ϕxthat
   ldiv!(parent(gf.rx), plan_x, parent(ϕxtmphat))
   # compute ddx
   ϕxtmphat .= - ξx.^2 .* ϕxthat
   ldiv!(parent(gf.ddx), plan_x, parent(ϕxtmphat))
   # FFT y
   grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y))
   x, y = grid.x, grid.y
   gridξ = localgrid(p.pen_y, (p.ξx, p.ξy))
   ξx, ξy = gridξ.x, gridξ.y
   tmp_y = similar(ϕythat)
   transpose!(tmp_y, ϕt)
   mul!(parent(ϕythat), plan_y, parent(tmp_y))
   # compute dy
   ϕytmphat .= im .* ξy .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.dy,tmp_y)
   # compute ry
   ϕytmphat .= -im .* x .* ξy .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.ry, tmp_y)
   # compute ddy
   ϕytmphat .= - ξy.^2 .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.ddy,tmp_y)
   tmp_y = nothing
   return nothing
end

## 3D

function computeDerivatives!(gf :: GradientField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # temporary fields
   ϕxthat = p.ϕx_hat
   ϕythat = p.ϕy_hat
   ϕzthat = p.ϕz_hat
   ϕxtmphat = p.ϕxtmp_hat
   ϕytmphat = p.ϕytmp_hat
   ϕztmphat = p.ϕztmp_hat
   # plans
   plan_x, plan_y, plan_z = p.plan_x, p.plan_y, p.plan_z
   # FFT x
   # frequencies
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_x, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   tmp_in = similar(ϕxthat)
   mul!(parent(ϕxthat), plan_x, parent(ϕt))
   # compute dx
   ϕxtmphat .= im .* ξx .* ϕxthat
   ldiv!(parent(tmp_in), plan_x, parent(ϕxtmphat))
   transpose!(gf.dx,tmp_in)
   # compute ddx
   ϕxtmphat .= - ξx.^2 .* ϕxthat
   ldiv!(parent(tmp_in), plan_x, parent(ϕxtmphat))
   transpose!(gf.ddx,tmp_in)
   # FFT y
   grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_y, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   tmp_in = similar(ϕythat)
   transpose!(tmp_in, ϕt)
   mul!(parent(ϕythat), plan_y, parent(tmp_in))
   # compute dy
   ϕytmphat .= im .* ξy .* ϕythat
   ldiv!(parent(tmp_in), plan_y, parent(ϕytmphat))
   transpose!(gf.dy,tmp_in)
   # compute ddy
   ϕytmphat .= - ξy.^2 .* ϕythat
   ldiv!(parent(tmp_in), plan_y, parent(ϕytmphat))
   transpose!(gf.ddy,tmp_in)
   # FFT z
   grid = localgrid(p.pen_z, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_z, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   tmp_in = similar(ϕzthat)
   transpose!(tmp_in,ϕt)
   mul!(parent(ϕzthat), plan_z, parent(tmp_in))
   # compute dz
   ϕztmphat .= im .* ξz .* ϕzthat
   ldiv!(parent(tmp_in), plan_z, parent(ϕztmphat))
   transpose!(gf.dz,tmp_in)
   # compute ddz
   ϕztmphat .= - ξz.^2 .* ϕzthat
   ldiv!(parent(tmp_in), plan_z, parent(ϕztmphat))
   transpose!(gf.ddz, tmp_in)
   tmp_in = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # temporary fields
   ϕxthat = p.ϕx_hat
   ϕythat = p.ϕy_hat
   ϕzthat = p.ϕz_hat
   ϕxtmphat = p.ϕxtmp_hat
   ϕytmphat = p.ϕytmp_hat
   ϕztmphat = p.ϕztmp_hat
   # plans
   plan_x, plan_y, plan_z = p.plan_x, p.plan_y, p.plan_z
   # FFT x
   # frequencies
   grid = localgrid(p.pen_x, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_x, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   mul!(parent(ϕxthat), plan_x, parent(ϕt))
   # compute dx
   ϕxtmphat .= im .* ξx .* ϕxthat
   ldiv!(parent(gf.dx), plan_x, parent(ϕxtmphat))
   # compute rx
   ϕxtmphat .= im .* y .* ξx .* ϕxthat
   ldiv!(parent(gf.rx), plan_x, parent(ϕxtmphat))
   # compute ddx
   ϕxtmphat .= - ξx.^2 .* ϕxthat
   ldiv!(parent(gf.ddx), plan_x, parent(ϕxtmphat))
   # FFT y
   grid = localgrid(p.pen_y, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_y, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   tmp_y = similar(ϕythat)
   transpose!(tmp_y, ϕt)
   mul!(parent(ϕythat), plan_y, parent(tmp_y))
   # compute dy
   ϕytmphat .= im .* ξy .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.dy,tmp_y)
   # compute ry
   ϕytmphat .= -im .* x .* ξy .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.ry, tmp_y)
   # compute ddy
   ϕytmphat .= - ξy.^2 .* ϕythat
   ldiv!(parent(tmp_y), plan_y, parent(ϕytmphat))
   transpose!(gf.ddy,tmp_y)
   # FFT z
   grid = localgrid(p.pen_z, (p.f.g.x, p.f.g.y, p.f.g.z))
   x, y, z = grid.x, grid.y, grid.z
   gridξ = localgrid(p.pen_z, (p.ξx, p.ξy, p.ξz))
   ξx, ξy, ξz = gridξ.x, gridξ.y, gridξ.z
   tmp_z = similar(ϕzthat)
   transpose!(tmp_y,ϕt)
   transpose!(tmp_z,tmp_y)
   mul!(parent(ϕzthat), plan_z, parent(tmp_z))
   # compute dz
   ϕztmphat .= im .* ξz .* ϕzthat
   ldiv!(parent(tmp_z), plan_z, parent(ϕztmphat))
   transpose!(tmp_y,tmp_z)
   transpose!(gf.dz,tmp_y)
   # compute ddz
   ϕztmphat .= - ξz.^2 .* ϕzthat
   ldiv!(parent(tmp_z), plan_z, parent(ϕztmphat))
   transpose!(tmp_y,tmp_z)
   transpose!(gf.ddz, tmp_y)
   tmp_y = nothing
   tmp_z = nothing
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