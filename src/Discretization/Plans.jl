"Abstract supertype for FFT plans."
abstract type AbstractPlan{F} end

#TODO add a new dimension for plans (F,PT}
# it could be FFT
# also CS for compact scheme

struct Plan2D{F} <: AbstractPlan{F}
   "plan in all directions"
   plan :: AbstractFFTs.Plan
   "plan in x direction only"
   plan_x :: AbstractFFTs.Plan
   "plan in y direction only"
   plan_y :: AbstractFFTs.Plan
   "ξx x frequencies"
   ξx :: Any
   "ξy y frequencies"
   ξy :: Any
end

struct Plan3D{F} <: AbstractPlan{F}
   "plan in all directions"
   plan :: AbstractFFTs.Plan
   "plan in x direction only"
   plan_x :: AbstractFFTs.Plan
   "plan in y direction only"
   plan_y :: AbstractFFTs.Plan
   "plan in z direction only"
   plan_z :: AbstractFFTs.Plan
   "ξx x frequencies"
   ξx :: Any
   "ξy y frequencies"
   ξy :: Any
   "ξz z frequencies"
   ξz :: Any
end

function Plan(f::F) where {F<:AbstractField2D}
   ξx = fftfreq(f.g.nx,2π/f.g.Δx)
   ξy = fftfreq(f.g.ny,2π/f.g.Δy)
   # if GPU do something
   Ξx = reshape(ξx,f.g.nx,1)
   Ξy = reshape(ξy,1,f.g.ny)
   return Plan2D{F}(plan_fft(f.ϕ,(1,2)),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2),
                    Ξx, Ξy)
end

function Plan(f::F) where {F<:AbstractField3D}
   ξx = fftfreq(f.g.nx,2π/f.g.Δx)
   ξy = fftfreq(f.g.ny,2π/f.g.Δy)
   ξz = fftfreq(f.g.nz,2π/f.g.Δz)
   # if GPU do something
   Ξx = reshape(ξx,f.g.nx,1,1)
   Ξy = reshape(ξy,1,f.g.ny,1)
   Ξz = reshape(ξz,1,1,f.g.nz)
   return Plan3D{F}(plan_fft(f.ϕ,(1,2,3)),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2),
                    plan_fft(f.ϕ,3),
                    Ξx, Ξy, Ξz)
end
