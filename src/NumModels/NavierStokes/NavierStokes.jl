export taylor_green!

function taylor_green!(f, x, y, z; θ::Real=0)
    @. f.ux[:, :, :] = 2 / √3 * sin(θ + 2π / 3) * sin(x) * cos(y) * cos(z)
    @. f.uy[:, :, :] = 2 / √3 * sin(θ - 2π / 3) * cos(x) * sin(y) * cos(z)
    @. f.uz[:, :, :] = 2 / √3 * sin(θ) * cos(x) * cos(y) * sin(z)
    return nothing
end

function rhs(n::AbstractNumModel{F,P}, u;
             lap_explicit::Bool=false) where {F<:AbstractField3D,P<:NavierStokesParameters}
    # references
    ξx = n.plan.ξx
    ξy = n.plan.ξy
    ξz = n.plan.ξz
    gridξ = localgrid(n.plan.pen_z, (ξx, ξy, ξz))
    # compute ω
    computeDerivatives!(n.gf, n.plan, u)
    # dU = u ∧ ω
    dU = cross(u, n.gf.ω)
    # dU_hat = F(u ∧ ω)
    uhat = n.plan.uz_hat
    uxtmphat = n.plan.uxtmp_hat
    uytmphat = n.plan.uytmp_hat
    uztmphat = n.plan.uztmp_hat
    uytmp2hat = n.plan.uytmp2_hat
    uztmp2hat = n.plan.uztmp2_hat
    dU_hat = n.plan.uztmp_hat
    for i in 1:3
        mul!(parent(uxtmphat[i]), n.plan.plan_x, parent(dU[i]))
        transpose!(uytmp2hat[i], uxtmphat[i])
        mul!(parent(uytmphat[i]), n.plan.plan_y, parent(uytmp2hat[i]))
        transpose!(uztmp2hat[i], uytmphat[i])
        mul!(parent(dU_hat[i]), n.plan.plan_z, parent(uztmp2hat[i]))
    end
    #dealias dU_hat
    dealias!(dU_hat, ξx, ξy, ξz)
    # compute ∇⋅(u∧ω)
    @. uztmphat[1] = gridξ.x * dU_hat[1] + gridξ.y * dU_hat[2] + gridξ.z * dU_hat[3]
    # dU = (u∧ω) - P(u∧ω)
    for i in 1:3
        @. dU_hat[i] = dU_hat[i] -
                       gridξ[i] * uztmphat[1] / ξsquared(gridξ.x, gridξ.y, gridξ.z)
    end
    # dU -= νΔu
    if lap_explicit
        for i in 1:3
            @. dU_hat[i] -= n.param.ν * (gridξ.x^2 + gridξ.y^2 + gridξ.z^2) * uhat[i]
        end
    end
    return dU_hat
end

function energy(n::AbstractNumModel{F,P},
                showEnergy=false) where {F<:AbstractField2D,P<:NavierStokesParameters}
    E = sum(0.5 .* real.(n.f.ux .^ 2 .+
                         n.f.uy .^ 2 .+
                         n.f.uz .^ 2) .* (n.f.g.Δx * n.f.g.Δy * n.f.g.Δz))
    if (showEnergy)
        println("E = $(E)")
    end
    return 0.0, E, 0.0, E
end

function energy(n::AbstractNumModel{F,P},
                showEnergy=false) where {F<:AbstractField3D,P<:NavierStokesParameters}
    E = sum(0.5 .*
            real.(n.f.ux[:, :, :] .^ 2 .+
                  n.f.uy[:, :, :] .^ 2 .+
                  n.f.uz[:, :, :] .^ 2) .* (n.f.g.Δx * n.f.g.Δy * n.f.g.Δz))
    if (showEnergy)
        println("E = $(E)")
    end
    return 0.0, E, 0.0, E
end
