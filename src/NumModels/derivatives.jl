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

 function computedxddx(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
    nx,_= size(ϕt)
    dx = similar(ϕt)
    ddx = similar(ϕt)
    for i = 1:nx
       dx[i,:] = (ϕt[ i%nx + 1,:] - ϕt[ mod(i%nx-2,nx) + 1,:]) / (2*n.f.g.Δx)
       ddx[i,:] = (-2 * ϕt[i,:] + ϕt[ i%nx + 1,:] + ϕt[ mod(i%nx-2,nx) + 1,:]) / n.f.g.Δx^2
    end
    return dx, ddx
 end

 function computedyddy(n::AbstractNumModel{F,P, Plan}, ϕt) where {F<:AbstractField2D, P<:AbstractParameters, Plan<:AbstractFDPlan}
    _,ny = size(ϕt)
    dy = similar(ϕt)
    ddy = similar(ϕt)
    for j = 1:ny
       dy[:,j] = (ϕt[ :, j%ny + 1] - ϕt[ :, mod(j%ny-2,ny) + 1]) / (2*n.f.g.Δy)
       ddy[:,j] = (-2 * ϕt[:,j] + ϕt[ :, j%ny + 1] + ϕt[ :, mod(j%ny-2,ny) + 1]) / n.f.g.Δy^2
    end
    return dy, ddy
 end