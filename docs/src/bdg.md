# Bogoliubov-de Gennes

Linearized excitations of a stationary Gross-Pitaevskii state.

The Gross-Pitaevskii equation admits stationary states ``$\psi_0$`` with a
chemical potential ``$\mu$``. The small-amplitude excitations about such a
state, ``(u, v)``, obey the Bogoliubov-de Gennes (BdG) eigenproblem

$$\begin{pmatrix} H & -K \\ K^* & -H \end{pmatrix}\begin{pmatrix} u \\ v \end{pmatrix} = \omega\begin{pmatrix} u \\ v \end{pmatrix},$$

with

$$H = c\,\nabla^2 + V + 2\beta|\psi_0|^2 - \mu - i\Omega\,\mathcal{L}_z, \qquad K = \beta\,\psi_0^2,$$

where `c` is the `coeffΔ` parameter (default `-0.5`, so ``$c\nabla^2$`` is the
positive kinetic operator ``$-|c|\,\nabla^2$``), `V` the trapping potential, `β`
the interaction coefficient and ``$\mathcal{L}_z$`` the rotation operator.

Because of the bosonic symmetries the spectrum is symmetric,
``$\omega \leftrightarrow -\omega$``, and the zero mode ``$\omega = 0$``,
``(u, v) = (\psi_0, \psi_0^*)``, is a spurious Goldstone mode. Each physical
mode is returned with the **symplectic normalization**
``$\|u\|^2 - \|v\|^2 = 1$``; the positive-norm member of every
``$\omega \leftrightarrow -\omega$`` pair is kept, and the zero mode is rejected.

## Numerical method

The ``$2N \times 2N$`` operator is applied **matrix-free** through an ARPACK
Krylov solve (``which = :SM``, smallest ``$|\omega|$``), reusing the package's
derivative machinery (`computeDerivatives!` with the model's plan), so no dense
matrix is ever formed and the memory footprint is ``$O(N)$``. The operator is
therefore compatible with every derivative scheme (periodic, homogeneous-Dirichlet
and homogeneous-Neumann compact plans).

The zero mode is removed *after* the solve, by identifying it with the known
vector ``(ψ₀, ψ₀*)``: a column whose overlap with that vector exceeds a
threshold is discarded. This structure-based rejection stays robust even when
``ψ₀`` is not an exact discrete eigenstate — in that case the zero mode shows up
as a small-residual pair at ``$\omega \approx \pm\varepsilon$`` rather than
exactly at zero, so a frequency-based cutoff would not be reliable.

## Example

The matrix-free solver is validated against the analytic spectrum of an
anisotropic 2D harmonic oscillator (``$\gamma_x = 1, \gamma_y = 2$``, hence
``$\omega_x = 1, \omega_y = \sqrt{2}$``), whose lowest BdG modes are the
non-degenerate frequencies ``$1, \sqrt{2}, 2$``:

```julia
grid  = Grid((20, 20), ((-6.0, 6.0), (-6.0, 6.0)))
field = Field(grid, ComplexField())
ωx, ωy = 1.0, sqrt(2.0)

# exact ground state  ψ ∝ exp(-ωx x²/2 - ωy y²/2)
X = reshape(vec(field.x), :, 1);  Y = reshape(vec(field.y), 1, :)
field.ϕ .= exp.(-ωx * X.^2 / 2 .- ωy * Y.^2 / 2)
field.ϕ ./= sqrt(sum(abs2.(parent(field.ϕ))) * grid.Δx * grid.Δy)

pot   = PotentialQuadratic(field; γx = ωx, γy = ωy^2)
param = BdGParameters(coeffΔ = -0.5, β = 0.0, pot = pot, Ω = 0.0)

n = NumModelBdG(field, param, 1, 1; nev = 3)   # three lowest modes
timeStep!(n)

n.mu   # ≈ (ωx + ωy)/2 = 1.2071
n.ωs   # ≈ [1.0, √2, 2.0]
```

`n.us` and `n.vs` hold the normalized `u` and `v` of each mode. Because
`timeStep!` runs the whole eigensolve, the standard driver also works:
`solve!(n)` with `niter = 1`.

## Model

```@docs
NumModelBdG
BdGParameters
bdg_mu
bdg_apply!
```
