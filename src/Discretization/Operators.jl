function cross(a, b)
   c = similar_data(a)
   @. c[1] = a[2] * b[3] - a[3] * b[2]
   @. c[2] = a[3] * b[1] - a[1] * b[3]
   @. c[3] = a[1] * b[2] - a[2] * b[1]
   return c
end

function dealias!(u_hat, ξx, ξy, ξz)
   func=x->x^2
   ξmax = 4/9 * minimum((mapreduce(func,max,ξx), mapreduce(func,max,ξy), mapreduce(func,max,ξz)))
   @. u_hat[1] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
   @. u_hat[2] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
   @. u_hat[3] *= (ξx^2 + ξy^2 + ξz^2) < ξmax
   return nothing
end