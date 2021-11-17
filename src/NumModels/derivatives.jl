# FFT

## 2D

function computeDerivatives!(gf :: GradientField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # references
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField2D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # references
   x, y = gf.f.g.x, gf.f.g.y
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute rx
   tmp .= im .* y .* ξx .* ϕthat_x
   ldiv!(gf.rx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ry
   tmp .= -im .* x .* ξy .* ϕhat_y
   ldiv!(gf.ry, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   return nothing
end

## 3D

function computeDerivatives!(gf :: GradientField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # references
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   # FFT z
   mul!(ϕthat_z, plan_z, ϕt)
   # compute dz
   tmp .= im .* ξz .* ϕthat_z
   ldiv!(gf.dz, plan_z, tmp)
   # compute ddz
   tmp .= - ξz.^2 .* ϕthat_z
   ldiv!(gf.ddz, plan_z, tmp)
   tmp = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFFTPlan, ϕt :: AbstractArray)
   # references
   x, y = gf.f.g.x, gf.f.g.y
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute rx
   tmp .= im .* y .* ξx .* ϕthat_x
   ldiv!(gf.rx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ry
   tmp .= -im .* x .* ξy .* ϕhat_y
   ldiv!(gf.ry, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   # FFT z
   mul!(ϕthat_z, plan_z, ϕt)
   # compute dz
   tmp .= im .* ξz .* ϕthat_z
   ldiv!(gf.dz, plan_z, tmp)
   # compute ddz
   tmp .= - ξz.^2 .* ϕthat_z
   ldiv!(gf.ddz, plan_z, tmp)
   tmp = nothing
   return nothing
end

# Finite Difference

## 2D

function computeDerivatives!(gf :: GradientField2D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   N = size(ϕt)
   nx = N[1]
   for i = 1:nx
      #dx[i,:] = (ϕt[ i%nx + 1,:] - ϕt[ mod(i%nx-2,nx) + 1,:]) / (2*n.f.g.Δx)
      # dx[i,:] = (-ϕt[ (i+1)%nx + 1,:] + 8*ϕt[ i%nx + 1,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:] + ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx)
      dx[i,:] = (ϕt[ (i+2)%nx + 1,:] - 9*ϕt[ (i+1)%nx + 1,:] + 45*ϕt[ i%nx + 1,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:] - ϕt[ mod(i%nx-4,nx) + 1,:]) / (60*n.f.g.Δx)
      # dx[i,:] = (-3*ϕt[ (i+3)%nx + 1,:] + 32*ϕt[ (i+2)%nx + 1,:] - 168*ϕt[ (i+1)%nx + 1,:] + 672*ϕt[ i%nx + 1,:]
      # - 672*ϕt[ mod(i%nx-2,nx) + 1,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:]) / (840*n.f.g.Δx)
      #ddx[i,:] = (-2 * ϕt[i,:] + ϕt[ i%nx + 1,:] + ϕt[ mod(i%nx-2,nx) + 1,:]) / n.f.g.Δx^2
      # ddx[i,:] = (-30 * ϕt[i,:] - ϕt[ (i+1)%nx + 1,:] + 16*ϕt[ i%nx + 1,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:] - ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx^2)
      ddx[i,:] = (-490 * ϕt[i,:] + 2*ϕt[ (i+2)%nx + 1,:] - 27*ϕt[ (i+1)%nx + 1,:] + 270*ϕt[ i%nx + 1,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:]) / (180*n.f.g.Δx^2)
      # ddx[i,:] = (-14350 * ϕt[i,:] - 9*ϕt[ (i+3)%nx + 1,:] + 128*ϕt[ (i+2)%nx + 1,:] - 1008*ϕt[ (i+1)%nx + 1,:] + 8064*ϕt[ i%nx + 1,:]
      # + 8064*ϕt[ mod(i%nx-2,nx) + 1,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:]) / (5040*n.f.g.Δx^2)
   end

   # references
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField2D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # references
   x, y = gf.f.g.x, gf.f.g.y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute rx
   tmp .= im .* y .* ξx .* ϕthat_x
   ldiv!(gf.rx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ry
   tmp .= -im .* x .* ξy .* ϕhat_y
   ldiv!(gf.ry, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   return nothing
end

## 3D

function computeDerivatives!(gf :: GradientField3D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # references
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   # FFT z
   mul!(ϕthat_z, plan_z, ϕt)
   # compute dz
   tmp .= im .* ξz .* ϕthat_z
   ldiv!(gf.dz, plan_z, tmp)
   # compute ddz
   tmp .= - ξz.^2 .* ϕthat_z
   ldiv!(gf.ddz, plan_z, tmp)
   tmp = nothing
   return nothing
end

function computeDerivatives!(gf :: GradientRotField3D, p :: AbstractFDPlan, ϕt :: AbstractArray)
   # references
   x, y = gf.f.g.x, gf.f.g.y
   ϕthat_x, ϕthat_y = p.ϕ_hat, p.ϕ_hat
   ξx, ξy = n.plan.ξx, n.plan.ξy
   plan_x, plan_y = n.plan.plan_x, n.plan.plan_y
   # FFT x
   mul!(ϕthat_x, plan_x, ϕt)
   # compute dx
   tmp = im .* ξx .* ϕthat_x
   ldiv!(gf.dx, plan_x, tmp)
   # compute rx
   tmp .= im .* y .* ξx .* ϕthat_x
   ldiv!(gf.rx, plan_x, tmp)
   # compute ddx
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(gf.ddx, plan_x, tmp)
   # FFT y
   mul!(ϕthat_y, plan_y, ϕt)
   # compute dy
   tmp .= im .* ξy .* ϕthat_y
   ldiv!(gf.dy, plan_y, tmp)
   # compute ry
   tmp .= -im .* x .* ξy .* ϕhat_y
   ldiv!(gf.ry, plan_y, tmp)
   # compute ddy
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(gf.ddy, plan_y, tmp)
   tmp = nothing
   # FFT z
   mul!(ϕthat_z, plan_z, ϕt)
   # compute dz
   tmp .= im .* ξz .* ϕthat_z
   ldiv!(gf.dz, plan_z, tmp)
   # compute ddz
   tmp .= - ξz.^2 .* ϕthat_z
   ldiv!(gf.ddz, plan_z, tmp)
   tmp = nothing
   return nothing
end

# Legacy

function computedxddx!(n::AbstractNumModel{F,P, Plan}, ϕt, dx, ddx) where {F<:AbstractField, P<:AbstractParameters, Plan<:AbstractFFTPlan}
   ϕthat_x = n.ϕ_hat
   ξx = n.plan.ξx
   plan_x = n.plan.plan_x
   mul!(ϕthat_x, plan_x, ϕt)
   tmp = im .* ξx .* ϕthat_x
   ldiv!(dx, plan_x, tmp)
   tmp .= - ξx.^2 .* ϕthat_x
   ldiv!(ddx, plan_x, tmp)
   return nothing
end

function computedyddy!(n::AbstractNumModel{F,P, Plan}, ϕt, dy, ddy) where {F<:AbstractField, P<:GrossPitaevskiiParameters, Plan<:AbstractFFTPlan}
   ϕthat_y = n.ϕ_hat
   ξy = n.plan.ξy
   plan_y = n.plan.plan_y
   mul!(ϕthat_y, plan_y, ϕt)
   tmp = im .* ξy .* ϕthat_y
   ldiv!(dy, plan_y, tmp)
   tmp .= - ξy.^2 .* ϕthat_y
   ldiv!(ddy, plan_y, tmp)
   return nothing
end

function computedzddz!(n::AbstractNumModel{F,P, Plan}, ϕt, dz, ddz) where {F<:AbstractField, P<:AbstractParameters, Plan<:AbstractFFTPlan}
   ϕthat_z = n.ϕ_hat
   ξz = n.plan.ξz
   plan_z = n.plan.plan_z
   mul!(ϕthat_z, plan_z, ϕt)
   tmp = im .* ξz .* ϕthat_z
   ldiv!(dz, plan_z, tmp)
   tmp .= - ξz.^2 .* ϕthat_z
   ldiv!(ddz, plan_z, tmp)
   return nothing
end

## order 2
# 1/2 0 -1/2
# 1 -2 +1
## order 4
# -1/12 2/3 0 -2/3 1/12
# -1/12 4/3 -5/2 4/3 -1/12
## order 6
# 1 -9 45 0 -45 9 -1 / 60
# 2 -27 270 -490 270 -27 2 / 180
## order 8
# -3 32 -168 672 0 -672 168 -32 3 / 840
# -9 128 -1008 8064 -14350 8064 -1008 128 -9 / 5040

function computedxddx!(n::AbstractNumModel{F,P, Plan}, ϕt, dx, ddx; order=2) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   N = size(ϕt)
   nx = N[1]
   if order == 2
      for i = 1:nx
         dx[i,:] = (ϕt[ i%nx + 1,:] - ϕt[ mod(i%nx-2,nx) + 1,:]) / (2*n.f.g.Δx)
         ddx[i,:] = (-2 * ϕt[i,:] + ϕt[ i%nx + 1,:] + ϕt[ mod(i%nx-2,nx) + 1,:]) / n.f.g.Δx^2
      end
   elseif order == 4
      for i = 1:nx
         dx[i,:] = (-ϕt[ (i+1)%nx + 1,:] + 8*ϕt[ i%nx + 1,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:] + ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx)
         ddx[i,:] = (-30 * ϕt[i,:] - ϕt[ (i+1)%nx + 1,:] + 16*ϕt[ i%nx + 1,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:] - ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx^2)
      end
   elseif order == 6
      for i = 1:nx
         dx[i,:] = (ϕt[ (i+2)%nx + 1,:] - 9*ϕt[ (i+1)%nx + 1,:] + 45*ϕt[ i%nx + 1,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:] - ϕt[ mod(i%nx-4,nx) + 1,:]) / (60*n.f.g.Δx)
         ddx[i,:] = (-490 * ϕt[i,:] + 2*ϕt[ (i+2)%nx + 1,:] - 27*ϕt[ (i+1)%nx + 1,:] + 270*ϕt[ i%nx + 1,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:]) / (180*n.f.g.Δx^2)
      end
   elseif order == 8
      for i = 1:nx
         dx[i,:] = (-3*ϕt[ (i+3)%nx + 1,:] + 32*ϕt[ (i+2)%nx + 1,:] - 168*ϕt[ (i+1)%nx + 1,:] + 672*ϕt[ i%nx + 1,:]
            - 672*ϕt[ mod(i%nx-2,nx) + 1,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:]) / (840*n.f.g.Δx)
         ddx[i,:] = (-14350 * ϕt[i,:] - 9*ϕt[ (i+3)%nx + 1,:] + 128*ϕt[ (i+2)%nx + 1,:] - 1008*ϕt[ (i+1)%nx + 1,:] + 8064*ϕt[ i%nx + 1,:]
            + 8064*ϕt[ mod(i%nx-2,nx) + 1,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:]) / (5040*n.f.g.Δx^2)
      end
   end
   return nothing
end

function computedxddx!(n::AbstractNumModel{F,P, Plan}, ϕt, dx, ddx; order=2) where {F<:AbstractField3D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   N = size(ϕt)
   nx = N[1]
   if order == 2
      for i = 1:nx
         dx[i,:,:] = (ϕt[ i%nx + 1,:,:] - ϕt[ mod(i%nx-2,nx) + 1,:,:]) / (2*n.f.g.Δx)
         ddx[i,:,:] = (-2 * ϕt[i,:,:] + ϕt[ i%nx + 1,:,:] + ϕt[ mod(i%nx-2,nx) + 1,:,:]) / n.f.g.Δx^2
      end
   elseif order == 4
      for i = 1:nx
         dx[i,:,:] = (-ϕt[ (i+1)%nx + 1,:,:] + 8*ϕt[ i%nx + 1,:,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:,:] + ϕt[ mod(i%nx-3,nx) + 1,:,:]) / (12*n.f.g.Δx)
         ddx[i,:,:] = (-30 * ϕt[i,:,:] - ϕt[ (i+1)%nx + 1,:,:] + 16*ϕt[ i%nx + 1,:,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:,:] - ϕt[ mod(i%nx-3,nx) + 1,:,:]) / (12*n.f.g.Δx^2)
      end
   elseif order == 6
      for i = 1:nx
         dx[i,:,:] = (ϕt[ (i+2)%nx + 1,:,:] - 9*ϕt[ (i+1)%nx + 1,:,:] + 45*ϕt[ i%nx + 1,:,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:,:] - ϕt[ mod(i%nx-4,nx) + 1,:,:]) / (60*n.f.g.Δx)
         ddx[i,:,:] = (-490 * ϕt[i,:,:] + 2*ϕt[ (i+2)%nx + 1,:,:] - 27*ϕt[ (i+1)%nx + 1,:,:] + 270*ϕt[ i%nx + 1,:,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:,:]) / (180*n.f.g.Δx^2)
      end
   elseif order == 8
      for i = 1:nx
         dx[i,:,:] = (-3*ϕt[ (i+3)%nx + 1,:,:] + 32*ϕt[ (i+2)%nx + 1,:,:] - 168*ϕt[ (i+1)%nx + 1,:,:] + 672*ϕt[ i%nx + 1,:,:]
            - 672*ϕt[ mod(i%nx-2,nx) + 1,:,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:,:]) / (840*n.f.g.Δx)
         ddx[i,:,:] = (-14350 * ϕt[i,:,:] - 9*ϕt[ (i+3)%nx + 1,:,:] + 128*ϕt[ (i+2)%nx + 1,:,:] - 1008*ϕt[ (i+1)%nx + 1,:,:] + 8064*ϕt[ i%nx + 1,:,:]
            + 8064*ϕt[ mod(i%nx-2,nx) + 1,:,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:,:]) / (5040*n.f.g.Δx^2)
      end
   end
   return nothing
end

function computedyddy!(n::AbstractNumModel{F,P, Plan}, ϕt, dy, ddy) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   N = size(ϕt)
   ny = N[2]
   if order == 2
      for j = 1:ny
         dy[:,j] = (ϕt[ :, j%ny + 1] - ϕt[ :, mod(j%ny-2,ny) + 1]) / (2*n.f.g.Δy)
         ddy[:,j] = (-2 * ϕt[:,j] + ϕt[ :, j%ny + 1] + ϕt[ :, mod(j%ny-2,ny) + 1]) / n.f.g.Δy^2
      end
   elseif order == 4
      for j = 1:ny
         dy[:,j] = (-ϕt[:, (j+1)%ny + 1] + 8*ϕt[:, j%ny + 1] - 8*ϕt[:, mod(j%ny-2,ny) + 1] + ϕt[:, mod(j%ny-3,ny) + 1]) / (12*n.f.g.Δy)
         ddy[:,j] = (-30 * ϕt[:,j] - ϕt[:, (j+1)%ny + 1] + 16*ϕt[:, j%ny + 1] + 16*ϕt[:, mod(j%ny-2,ny) + 1] - ϕt[:, mod(j%ny-3,ny) + 1]) / (12*n.f.g.Δy^2)
      end
   elseif order == 6
      for j = 1:ny
         dy[:,j] = (ϕt[:, (j+2)%ny + 1] - 9*ϕt[:, (j+1)%ny + 1] + 45*ϕt[:, j%ny + 1] - 45*ϕt[:, mod(j%ny-2,ny) + 1] + 9*ϕt[:, mod(j%ny-3,ny) + 1] - ϕt[:, mod(j%ny-4,ny) + 1]) / (60*n.f.g.Δy)
         ddy[:,j] = (-490 * ϕt[:,j] + 2*ϕt[:, (j+2)%ny + 1] - 27*ϕt[:, (j+1)%ny + 1] + 270*ϕt[:, j%ny + 1] + 270*ϕt[:, mod(j%ny-2,ny) + 1] - 27*ϕt[:, mod(j%ny-3,ny) + 1] + 2*ϕt[:, mod(j%ny-4,ny) + 1]) / (180*n.f.g.Δy^2)
      end
   elseif order == 8
      for j = 1:ny
         dy[:,j] = (-3*ϕt[:, (j+3)%ny + 1] + 32*ϕt[:, (j+2)%ny + 1] - 168*ϕt[:, (j+1)%ny + 1] + 672*ϕt[:, j%ny + 1]
            - 672*ϕt[:, mod(j%ny-2,ny) + 1] + 168*ϕt[:, mod(j%ny-3,ny) + 1] - 32*ϕt[:, mod(j%ny-4,ny) + 1] + 3*ϕt[:, mod(j%ny-5,ny) + 1]) / (840*n.f.g.Δy)
         ddy[:,j] = (-14350 * ϕt[:,j] - 9*ϕt[:, (j+3)%ny + 1] + 128*ϕt[:, (j+2)%ny + 1] - 1008*ϕt[:, (j+1)%ny + 1] + 8064*ϕt[:, j%ny + 1]
            + 8064*ϕt[:, mod(j%ny-2,ny) + 1] - 1008*ϕt[:, mod(j%ny-3,ny) + 1] + 128*ϕt[:, mod(j%ny-4,ny) + 1] - 9*ϕt[:, mod(j%ny-5,ny) + 1]) / (5040*n.f.g.Δy^2)
      end
   end
   return nothing
end

function computedyddy!(n::AbstractNumModel{F,P, Plan}, ϕt, dy, ddy) where {F<:AbstractField3D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   N = size(ϕt)
   ny = N[2]
   if order == 2
      for j = 1:ny
         dy[:,j,:] = (ϕt[ :, j%ny + 1,:] - ϕt[ :, mod(j%ny-2,ny) + 1,:]) / (2*n.f.g.Δy)
         ddy[:,j,:] = (-2 * ϕt[:,j,:] + ϕt[ :, j%ny + 1,:] + ϕt[ :, mod(j%ny-2,ny) + 1,:]) / n.f.g.Δy^2
      end
   elseif order == 4
      for j = 1:ny
         dy[:,j,:] = (-ϕt[:, (j+1)%ny + 1,:] + 8*ϕt[:, j%ny + 1,:] - 8*ϕt[:, mod(j%ny-2,ny) + 1,:] + ϕt[:, mod(j%ny-3,ny) + 1,:]) / (12*n.f.g.Δy)
         ddy[:,j,:] = (-30 * ϕt[:,j,:] - ϕt[:, (j+1)%ny + 1,:] + 16*ϕt[:, j%ny + 1,:] + 16*ϕt[:, mod(j%ny-2,ny) + 1,:] - ϕt[:, mod(j%ny-3,ny) + 1,:]) / (12*n.f.g.Δy^2)
      end
   elseif order == 6
      for j = 1:ny
         dy[:,j,:] = (ϕt[:, (j+2)%ny + 1,:] - 9*ϕt[:, (j+1)%ny + 1,:] + 45*ϕt[:, j%ny + 1,:] - 45*ϕt[:, mod(j%ny-2,ny) + 1,:] + 9*ϕt[:, mod(j%ny-3,ny) + 1,:] - ϕt[:, mod(j%ny-4,ny) + 1,:]) / (60*n.f.g.Δy)
         ddy[:,j,:] = (-490 * ϕt[:,j,:] + 2*ϕt[:, (j+2)%ny + 1,:] - 27*ϕt[:, (j+1)%ny + 1,:] + 270*ϕt[:, j%ny + 1,:] + 270*ϕt[:, mod(j%ny-2,ny) + 1,:] - 27*ϕt[:, mod(j%ny-3,ny) + 1,:] + 2*ϕt[:, mod(j%ny-4,ny) + 1,:]) / (180*n.f.g.Δy^2)
      end
   elseif order == 8
      for j = 1:ny
         dy[:,j,:] = (-3*ϕt[:, (j+3)%ny + 1,:] + 32*ϕt[:, (j+2)%ny + 1,:] - 168*ϕt[:, (j+1)%ny + 1,:] + 672*ϕt[:, j%ny + 1,:]
            - 672*ϕt[:, mod(j%ny-2,ny) + 1,:] + 168*ϕt[:, mod(j%ny-3,ny) + 1,:] - 32*ϕt[:, mod(j%ny-4,ny) + 1,:] + 3*ϕt[:, mod(j%ny-5,ny) + 1,:]) / (840*n.f.g.Δy)
         ddy[:,j,:] = (-14350 * ϕt[:,j,:] - 9*ϕt[:, (j+3)%ny + 1,:] + 128*ϕt[:, (j+2)%ny + 1,:] - 1008*ϕt[:, (j+1)%ny + 1,:] + 8064*ϕt[:, j%ny + 1,:]
            + 8064*ϕt[:, mod(j%ny-2,ny) + 1,:] - 1008*ϕt[:, mod(j%ny-3,ny) + 1,:] + 128*ϕt[:, mod(j%ny-4,ny) + 1,:] - 9*ϕt[:, mod(j%ny-5,ny) + 1,:]) / (5040*n.f.g.Δy^2)
      end
   end
   return nothing
end

function computedzddz!(n::AbstractNumModel{F,P, Plan}, ϕt, dz, ddz) where {F<:AbstractField3D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   N = size(ϕt)
   nz = N[3]
   if order == 2
      for k = 1:nz
         dz[:,k,:] = (ϕt[ :, k%nz + 1,:] - ϕt[ :, mod(k%nz-2,nz) + 1,:]) / (2*n.f.g.Δz)
         ddz[:,k,:] = (-2 * ϕt[:,k,:] + ϕt[ :, k%nz + 1,:] + ϕt[ :, mod(k%nz-2,nz) + 1,:]) / n.f.g.Δz^2
      end
   elseif order == 4
      for k = 1:nz
         dz[:,k,:] = (-ϕt[:, (k+1)%nz + 1,:] + 8*ϕt[:, k%nz + 1,:] - 8*ϕt[:, mod(k%nz-2,nz) + 1,:] + ϕt[:, mod(k%nz-3,nz) + 1,:]) / (12*n.f.g.Δz)
         ddz[:,k,:] = (-30 * ϕt[:,k,:] - ϕt[:, (k+1)%nz + 1,:] + 16*ϕt[:, k%nz + 1,:] + 16*ϕt[:, mod(k%nz-2,nz) + 1,:] - ϕt[:, mod(k%nz-3,nz) + 1,:]) / (12*n.f.g.Δz^2)
      end
   elseif order == 6
      for k = 1:nz
         dz[:,k,:] = (ϕt[:, (k+2)%nz + 1,:] - 9*ϕt[:, (k+1)%nz + 1,:] + 45*ϕt[:, k%nz + 1,:] - 45*ϕt[:, mod(k%nz-2,nz) + 1,:] + 9*ϕt[:, mod(k%nz-3,nz) + 1,:] - ϕt[:, mod(k%nz-4,nz) + 1,:]) / (60*n.f.g.Δz)
         ddz[:,k,:] = (-490 * ϕt[:,k,:] + 2*ϕt[:, (k+2)%nz + 1,:] - 27*ϕt[:, (k+1)%nz + 1,:] + 270*ϕt[:, k%nz + 1,:] + 270*ϕt[:, mod(k%nz-2,nz) + 1,:] - 27*ϕt[:, mod(k%nz-3,nz) + 1,:] + 2*ϕt[:, mod(k%nz-4,nz) + 1,:]) / (180*n.f.g.Δz^2)
      end
   elseif order == 8
      for k = 1:nz
         dz[:,k,:] = (-3*ϕt[:, (k+3)%nz + 1,:] + 32*ϕt[:, (k+2)%nz + 1,:] - 168*ϕt[:, (k+1)%nz + 1,:] + 672*ϕt[:, k%nz + 1,:]
            - 672*ϕt[:, mod(k%nz-2,nz) + 1,:] + 168*ϕt[:, mod(k%nz-3,nz) + 1,:] - 32*ϕt[:, mod(k%nz-4,nz) + 1,:] + 3*ϕt[:, mod(k%nz-5,nz) + 1,:]) / (840*n.f.g.Δz)
         ddz[:,k,:] = (-14350 * ϕt[:,k,:] - 9*ϕt[:, (k+3)%nz + 1,:] + 128*ϕt[:, (k+2)%nz + 1,:] - 1008*ϕt[:, (k+1)%nz + 1,:] + 8064*ϕt[:, k%nz + 1,:]
            + 8064*ϕt[:, mod(k%nz-2,nz) + 1,:] - 1008*ϕt[:, mod(k%nz-3,nz) + 1,:] + 128*ϕt[:, mod(k%nz-4,nz) + 1,:] - 9*ϕt[:, mod(k%nz-5,nz) + 1,:]) / (5040*n.f.g.Δz^2)
      end
   end
   return nothing
end