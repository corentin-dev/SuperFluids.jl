function krylov!(n::AbstractNumModel, ϕ)
   normb = sqrt( mapreduce((x)->real(x*conj(x)),+,n.b) )
   r = n.b - prodA(n,ϕ)
   r₂ = copy(r)
   p = copy(r)
   niter = 0
   for i = 1:n.nkrylov
      Ap = prodA(n,p)
      rr₂ = mapreduce((x,y)->real(x*conj(y)),+,r,r₂)
      Apr₂ = mapreduce((x,y)->real(x*conj(y)),+,Ap,r₂)
      α = rr₂/Apr₂
      s = r - α * Ap
      As = prodA(n,s)
      Ass = mapreduce((x,y)->real(x*conj(y)),+,As,s)
      AsAs = mapreduce((x)->real(x*conj(x)),+,As)
      ω = Ass/AsAs
      @. ϕ = ϕ + α*p + ω*s
      rr₂old = mapreduce((x,y)->real(x*conj(y)),+,r,r₂)
      r = s - ω*As
      rr = mapreduce((x)->real(x*conj(x)),+,r)
      if sqrt(rr)/normb < n.tolkrylov
         println("Number of Krylov iterations: $(i)")
         break
      elseif i == n.nkrylov
         println("warning: Krylov solver did not converge")
      end
      niter = i
      rr₂ = mapreduce((x,y)->real(x*conj(y)),+,r,r₂)
      β = rr₂ / rr₂old * α / ω
      p = r + β * (p - ω*Ap)
   end
   return niter
end

function krylovPreCond!(n::AbstractNumModel, ϕ)
   normb=sqrt(real(sum( n.M .* n.b .* conj.(n.M .* n.b) )))
   r = n.M .* (n.b .- prodA(n,ϕ))
   r₂ = copy(r)
   p = copy(r)
   niter = 0
   for i = 1:n.nkrylov
      Ap = n.M .* prodA(n,p)
      α = real(sum(r.*conj.(r₂))/sum(Ap.*conj.(r₂)))
      s = r - α * Ap
      As = n.M .* prodA(n,s)
      ω = sum(As.*conj.(s))/sum(abs2.(As))
      @. ϕ = ϕ + α*p + ω*s
      rr₂ = sum(r.*conj.(r₂))
      r = s - ω*As
      if sqrt(sum(abs2.(r)))/normb < n.tolkrylov
         println("Number of Krylov iterations: $(i)")
         break
      elseif i == n.nkrylov
         println("warning: Krylov solver did not converge")
      end
      niter = i
      β = sum(r.*conj.(r₂)) / rr₂ * α / ω
      p = r + β * (p - ω*Ap)
   end
   return niter
end
