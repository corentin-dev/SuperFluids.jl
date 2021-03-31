"Abstract supertype for FFT plans."
abstract type AbstractPlan{F} end

struct Plan2D{F} <: AbstractPlan{F}
   plan :: AbstractFFTs.Plan
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
end

struct Plan3D{F} <: AbstractPlan{F}
   plan :: AbstractFFTs.Plan
   plan_x :: AbstractFFTs.Plan
   plan_y :: AbstractFFTs.Plan
   plan_z :: AbstractFFTs.Plan
end

function Plan(f::F) where {F<:AbstractField2D}
   return Plan2D{F}(plan_fft(f.ϕ),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2))
end

function Plan(f::F) where {F<:AbstractField3D}
   return Plan3D{F}(plan_fft(f.ϕ),
                    plan_fft(f.ϕ,1),
                    plan_fft(f.ϕ,2),
                    plan_fft(f.ϕ,3))
end
