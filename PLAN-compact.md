# Plan : schémas compacts (FD implicites) portés de GPS

Réf. GPS : `gps-gs/src/GPS_compute_compact.f90`
Note GPS : « all derivative operators defined with 6 order compact scheme ».
Croisé-contrôlé avec Xcompact3d (`schemes.f90`, `ifirstder==4` / `isecondder==4`) : **mêmes coefficients**.

## Statut : ✅ FAIT (intégrations + tests)

## Objectif (atteint)
Dérivées 1ʳᵉ et 2ⁿᵈ **compacts d'ordre 6**, périodiques (`cdl*==0` GPS), branchables sur
`computeDerivatives!` (alternative FD/FDD au FFT) via `Plan(..., t=CompactPlan())`.

## Réalisations

### Étape 1 — Prototype + validation (hors package) ✅
- Noyaux 1D + solve cyclique portés mot à mot du Fortran.
- **Solveur validé à la machine precision** : la matrice effective inversée par
  `prepare_compact` + balayages avant/arrière + correction `sx` est strictement égale à la
  matrice circulaire compacte périodique (`max|M_eff − A_circ| = 2.2e-16`).
- **Bogues évités** : `prepare_compact` ne modifie **pas** `f` ; la sur-diagonale du balayage
  arrière est `fvec[i]` (pas `ffx(i+1)` ni `-b[i]`) ; copier `f` avant le solve.
- **Convergence mesurée** (f = sin x + 0.5 cos 2y, Δx = 2π/N) :
  - 1ʳᵉ dérivée : **ordre 6** (6.02 / 6.01 / 6.00 / 5.99 pour N = 16→128)
  - 2ⁿᵈ dérivée : **ordre 6** (6.01 / 6.00 / 5.90 pour N = 16→128)
  - déclin apparent N ≥ 256 = plancher d'arrondi machine (~1e-13).
- **GO** : les deux dérivées sont bien d'ordre 6 (cohérent header GPS + Xcompact3d).

### Étape 2 — Intégration : `PlanCompact` ✅
- `src/Discretization/Plans.jl` :
  - `struct CompactPlan <: PlanType` (exporté).
  - `abstract type AbstractCompactPlan{N}` + `PlanCompact2D` / `PlanCompact3D`.
  - `struct CompactAxis{R}` : multiplieurs précalculés pour un axe
    (n, order, alpha, a, b, s, w, f, t, denom).
  - `compact_setup(n, Δ, order)` : construit la tridiagonale cyclique + **vecteur de
    correction précalculé** (indépendant du second membre — validé exact, diff = 0 sur 50 RHS).
  - `Plan(f; t=CompactPlan())` branché pour 2D et 3D (crée `pen_y`/`pen_z` + scratch).

### Étape 3 — Dispatch `computeDerivatives!` ✅
- `src/NumModels/derivatives.jl` : section "Compact Finite Difference".
  - Noyaux ligne `compact1line!`/`compact2line!` (2D) et `compact1line3!`/`compact2line3!` (3D).
  - `_compact_solve!` : 2 balayages Thomas + correction cyclique (in-place).
  - 4 méthodes `computeDerivatives!(gf, plan::AbstractCompactPlan, ϕt)` :
    `GradientField2D/3D` + `GradientRotField2D/3D` (rx = y·dx, ry = −x·dy).
  - Layout : x sur dim 1 (ligne locale pleine, **aucune comms MPI**) ; y/z via `transpose!`
    (pattern identique au FD : z → y → x, une permutation à la fois).

### Étape 4 — Tests package ✅
- `test/runtests.jl` : 4 testsets compact (même protocole Fourier-mode que FFT/FD, `TOL_FD`) :
  - 2D rotation=true (6 tests), 2D rotation=false (4), 3D rotation=true (8), 3D rotation=false (6).
- **Suite complète : 0 échec.** Erreurs mesurées (mode k=(2,3[,4])) :
  - 2D : dx/ddy ~ 4e-9/3e-9, rx/ry exacts.
  - 3D : dx/ddx ~ 2.7e-8/1.7e-8, dz/ddz ~ 1.8e-6/1.1e-6 (6ᵉ ordre).

## Décisions
- **Fidélité GPS** : port exact de l'algorithme (pas de Sherman-Morrison générique ni de
  forme fermée) ; le solveur est prouvé équivalent à la matrice circulaire compacte.
- **Convention de signe** : opérateurs = dérivées pures (`∂x`, `∂xx`), **sans** le `−alpha`
  de GPS (échelle physique GP, appliquée par le modèle dans `lapRot`).
- **Optimisation** : correction cyclique précalculée par axe (1 seul `CompactAxis`/ordre/axe),
  pas recalculée à chaque appel.
- **Périodique uniquement** : cas `cdl*==2` (bord non périodique, one-sided) reporté.

## Restant (optionnel, reporté)
- Docs : `docs/src/...` — documenter l'option `t=CompactPlan()`, ordre 6, limites.
- Brancher explicitement une option `discretization=` dans `NumModelGP`/`NumModelHVBK`
  (actuellement l'utilisateur construit le plan s'il le souhaite ; le default reste FFT).
- Cas non périodique (`cdl*==2`).

## Coefficients GPS (périodique, cdl*==0) — portés dans `compact_setup`
- 1ʳᵉ : `alpha=1/3`, `a=(7/9)/Δx`, `b=(1/36)/Δx`
- 2ⁿᵈ : `alpha=2/11`, `a=(12/11)/Δx²`, `b=(3/44)/Δx²`

## Solve cyclique (port exact)
- Tridiagonale : `diag=1` (interne), `diag(1)=2`, `diag(n)=1+α²`, `sous=sur=α` (`sur(n)=0`).
- `prepare_compact` : Thomas non cyclique (w=c ; s(i)=b(i-1)/w(i-1) ; w(i)=w(i)-f(i-1)s(i) ;
  w=1/w) — **f inchangé**.
- Forward : `r(i) -= r(i-1)s(i)`  |  Backward : `r(i) = (r(i) - f(i)r(i+1))w(i)` (f = sur-diag).
- Correction : `sx = (r(1) - α r(n)) / (1 + t(1) - α t(n))` ; `r(i) -= sx t(i)`.
- `t` (correction) précalculé : `t(1)=-1, t(n)=α`, puis mêmes balayages (indépendant du 2ᵉ membre).

## Fichiers
- `src/Discretization/Plans.jl` — `CompactPlan`, `CompactAxis`, `compact_setup`, plans compact.
- `src/NumModels/derivatives.jl` — noyaux + dispatch `computeDerivatives!`.
- `test/runtests.jl` — 4 testsets compact.
