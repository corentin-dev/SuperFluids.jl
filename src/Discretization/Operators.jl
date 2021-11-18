function curl_hat(v_hat, ξx, ξy, ξz)
   v_curl_hat = similar(v_hat)
   @. v_curl_hat[:,:,:,1] = im * (ξy * v_hat[:,:,:,3] - ξz * v_hat[:,:,:,2])
   @. v_curl_hat[:,:,:,2] = im * (ξz * v_hat[:,:,:,1] - ξx * v_hat[:,:,:,3])
   @. v_curl_hat[:,:,:,3] = im * (ξx * v_hat[:,:,:,2] - ξy * v_hat[:,:,:,1])
   return v_curl_hat
end

function cross(a, b)
   @assert size(a) == size(b)
   c = similar(a)
   @. c[:,:,:,1] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   @. c[:,:,:,2] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   @. c[:,:,:,3] = a[:,:,:,2] * b[:,:,:,3] - a[:,:,:,3] * b[:,:,:,2]
   return c
end

function dealias!(u_hat, ξx, ξy, ξz)
   ξmax = 4/9 * minimum((maximum(ξx.^2),maximum(ξy.^2),maximum(ξz.^2)))
   u_hat[:,:,:,1] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   u_hat[:,:,:,2] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   u_hat[:,:,:,3] .*= (ξx.^2 .+ ξy.^2 .+ ξz.^2) .< ξmax
   return nothing
end