function krylov!(n::AbstractNumModel, ϕ)
   normb=sqrt(real(sum(n.b.*conj.(n.b))))
   r = n.b - prodA(n,ϕ)
   r₂ = copy(r)
   p = copy(r)
   for i = 1:n.nkrylov
      Ap = prodA(n,p)
      α = real(sum(r.*conj.(r₂))/sum(Ap.*conj.(r₂)))
      s = r - α * Ap
      As = prodA(n,s)
      ω = sum(As.*conj.(s))/sum(As.*conj.(As))
      @. ϕ = ϕ + α*p + ω*s
      rr₂ = sum(r.*conj.(r₂))
      r = s - ω*As
      if sqrt(real(sum(r.*conj.(r))))/normb < n.tolkrylov
         break
      end
      β = sum(r.*conj.(r₂)) / rr₂ * α / ω
      p = r + β * (p - ω*Ap)
   end
   return nothing
end

function krylovPreCond!(n::AbstractNumModel, ϕ)
   normb=sqrt(real(sum( n.M .* n.b .* conj.(n.M .* n.b) )))
   # println("normb $(normb)")
   r = n.M .* (n.b .- prodA(n,ϕ))
   r₂ = copy(r)
   p = copy(r)
   for i = 1:n.nkrylov
      Ap = n.M .* prodA(n,p)
      α = real(sum(r.*conj.(r₂))/sum(Ap.*conj.(r₂)))
      s = r - α * Ap
      As = n.M .* prodA(n,s)
      ω = sum(As.*conj.(s))/sum(As.*conj.(As))
      @. ϕ = ϕ + α*p + ω*s
      rr₂ = sum(r.*conj.(r₂))
      r = s - ω*As
      if sqrt(real(sum(r.*conj.(r))))/normb < n.tolkrylov
         break
      end
      β = sum(r.*conj.(r₂)) / rr₂ * α / ω
      p = r + β * (p - ω*Ap)
   end
   return nothing
end
