function computedxddx(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField, P<:AbstractParameters, Plan<:AbstractFFTPlan}
   ϕthat_x = n.ϕ_hat
   ξx = n.plan.ξx
   plan_x = n.plan.plan_x
   mul!(ϕthat_x, plan_x, ϕt)
   dx = plan_x \ ( im .* ξx .* ϕthat_x )
   ddx = plan_x \ ( - ξx.^2 .* ϕthat_x )
   return dx, ddx
end

function computedyddy(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField, P<:GrossPitaevskiiParameters, Plan<:AbstractFFTPlan}
   ϕthat_y = n.ϕ_hat
   ξy = n.plan.ξy
   plan_y = n.plan.plan_y
   mul!(ϕthat_y, plan_y, ϕt)
   dy = plan_y \ ( im .* ξy .* ϕthat_y )
   ddy = plan_y \ ( - ξy.^2 .* ϕthat_y )
   return dy, ddy
end

function computedzddz(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField, P<:AbstractParameters, Plan<:AbstractFFTPlan}
   ϕthat_z = n.ϕ_hat
   ξz = n.plan.ξz
   plan_z = n.plan.plan_z
   mul!(ϕthat_z, plan_z, ϕt)
   dz = plan_z \ ( im .* ξz .* ϕthat_z )
   ddz = plan_z \ ( - ξz.^2 .* ϕthat_z )
   return dz, ddz
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

function computedxddx(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   nx,_= size(ϕt)
   dx = similar(ϕt)
   ddx = similar(ϕt)
   for i = 1:nx
      #dx[i,:] = (ϕt[ i%nx + 1,:] - ϕt[ mod(i%nx-2,nx) + 1,:]) / (2*n.f.g.Δx)
      #dx[i,:] = (-ϕt[ (i+1)%nx + 1,:] + 8*ϕt[ i%nx + 1,:] - 8*ϕt[ mod(i%nx-2,nx) + 1,:] + ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx)
      #dx[i,:] = (ϕt[ (i+2)%nx + 1,:] - 9*ϕt[ (i+1)%nx + 1,:] + 45*ϕt[ i%nx + 1,:] - 45*ϕt[ mod(i%nx-2,nx) + 1,:] + 9*ϕt[ mod(i%nx-3,nx) + 1,:] - ϕt[ mod(i%nx-4,nx) + 1,:]) / (60*n.f.g.Δx)
      dx[i,:] = (-3*ϕt[ (i+3)%nx + 1,:] + 32*ϕt[ (i+2)%nx + 1,:] - 168*ϕt[ (i+1)%nx + 1,:] + 672*ϕt[ i%nx + 1,:]
      - 672*ϕt[ mod(i%nx-2,nx) + 1,:] + 168*ϕt[ mod(i%nx-3,nx) + 1,:] - 32*ϕt[ mod(i%nx-4,nx) + 1,:] + 3*ϕt[ mod(i%nx-5,nx) + 1,:]) / (840*n.f.g.Δx)
      #ddx[i,:] = (-2 * ϕt[i,:] + ϕt[ i%nx + 1,:] + ϕt[ mod(i%nx-2,nx) + 1,:]) / n.f.g.Δx^2
      # ddx[i,:] = (-30 * ϕt[i,:] - ϕt[ (i+1)%nx + 1,:] + 16*ϕt[ i%nx + 1,:] + 16*ϕt[ mod(i%nx-2,nx) + 1,:] - ϕt[ mod(i%nx-3,nx) + 1,:]) / (12*n.f.g.Δx^2)
      #ddx[i,:] = (-490 * ϕt[i,:] + 2*ϕt[ (i+2)%nx + 1,:] - 27*ϕt[ (i+1)%nx + 1,:] + 270*ϕt[ i%nx + 1,:] + 270*ϕt[ mod(i%nx-2,nx) + 1,:] - 27*ϕt[ mod(i%nx-3,nx) + 1,:] + 2*ϕt[ mod(i%nx-4,nx) + 1,:]) / (180*n.f.g.Δx^2)
      ddx[i,:] = (-14350 * ϕt[i,:] - 9*ϕt[ (i+3)%nx + 1,:] + 128*ϕt[ (i+2)%nx + 1,:] - 1008*ϕt[ (i+1)%nx + 1,:] + 8064*ϕt[ i%nx + 1,:]
      + 8064*ϕt[ mod(i%nx-2,nx) + 1,:] - 1008*ϕt[ mod(i%nx-3,nx) + 1,:] + 128*ϕt[ mod(i%nx-4,nx) + 1,:] - 9*ϕt[ mod(i%nx-5,nx) + 1,:]) / (5040*n.f.g.Δx^2)
   end
   return dx, ddx
end

function computedyddy(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
   _,ny = size(ϕt)
   dy = similar(ϕt)
   ddy = similar(ϕt)
   for j = 1:ny
      #dy[:,j] = (ϕt[ :, j%ny + 1] - ϕt[ :, mod(j%ny-2,ny) + 1]) / (2*n.f.g.Δy)
      #dy[:,j] = (-ϕt[:, (j+1)%ny + 1] + 8*ϕt[:, j%ny + 1] - 8*ϕt[:, mod(j%ny-2,ny) + 1] + ϕt[:, mod(j%ny-3,ny) + 1]) / (12*n.f.g.Δy)
      # dy[:,j] = (ϕt[:, (j+2)%ny + 1] - 9*ϕt[:, (j+1)%ny + 1] + 45*ϕt[:, j%ny + 1] - 45*ϕt[:, mod(j%ny-2,ny) + 1] + 9*ϕt[:, mod(j%ny-3,ny) + 1] - ϕt[:, mod(j%ny-4,ny) + 1]) / (60*n.f.g.Δy)
      dy[:,j] = (-3*ϕt[:, (j+3)%ny + 1] + 32*ϕt[:, (j+2)%ny + 1] - 168*ϕt[:, (j+1)%ny + 1] + 672*ϕt[:, j%ny + 1]
      - 672*ϕt[:, mod(j%ny-2,ny) + 1] + 168*ϕt[:, mod(j%ny-3,ny) + 1] - 32*ϕt[:, mod(j%ny-4,ny) + 1] + 3*ϕt[:, mod(j%ny-5,ny) + 1]) / (840*n.f.g.Δy)
      #ddy[:,j] = (-2 * ϕt[:,j] + ϕt[ :, j%ny + 1] + ϕt[ :, mod(j%ny-2,ny) + 1]) / n.f.g.Δy^2
      #ddy[:,j] = (-30 * ϕt[:,j] - ϕt[:, (j+1)%ny + 1] + 16*ϕt[:, j%ny + 1] + 16*ϕt[:, mod(j%ny-2,ny) + 1] - ϕt[:, mod(j%ny-3,ny) + 1]) / (12*n.f.g.Δy^2)
      #ddy[:,j] = (-490 * ϕt[:,j] + 2*ϕt[:, (j+2)%ny + 1] - 27*ϕt[:, (j+1)%ny + 1] + 270*ϕt[:, j%ny + 1] + 270*ϕt[:, mod(j%ny-2,ny) + 1] - 27*ϕt[:, mod(j%ny-3,ny) + 1] + 2*ϕt[:, mod(j%ny-4,ny) + 1]) / (180*n.f.g.Δy^2)
      ddy[:,j] = (-14350 * ϕt[:,j] - 9*ϕt[:, (j+3)%ny + 1] + 128*ϕt[:, (j+2)%ny + 1] - 1008*ϕt[:, (j+1)%ny + 1] + 8064*ϕt[:, j%ny + 1]
      + 8064*ϕt[:, mod(j%ny-2,ny) + 1] - 1008*ϕt[:, mod(j%ny-3,ny) + 1] + 128*ϕt[:, mod(j%ny-4,ny) + 1] - 9*ϕt[:, mod(j%ny-5,ny) + 1]) / (5040*n.f.g.Δy^2)
   end
   return dy, ddy
end