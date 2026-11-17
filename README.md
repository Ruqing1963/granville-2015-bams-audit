# A note on the variational and exponential-sum estimates in Granville's "Primes in intervals of bounded length"

Ruqing Chen (2026). doi:[10.5281/zenodo.24296047](https://doi.org/10.5281/zenodo.24296047)

A. Granville, *Primes in intervals of bounded length*, Bull. Amer. Math. Soc. (N.S.) 52 (2015), 171–222
(doi:[10.1090/S0273-0979-2015-01480-1](https://doi.org/10.1090/S0273-0979-2015-01480-1)), gives in Sections 6.8–6.9
a self-contained lower bound for Maynard's ratio `rho(F)`, leading to Theorem 6.4 and Corollary 6.5 (`B_m = e^(8m+5)`).
This note checks that argument and the exposition of Zhang's theorem in Part 2.

* **Section 6.8 (p. 200).** The numerator is rewritten as `L_k - M_k`, but `k ∫(A-B)^2 <= L_k - k M_k` (strictly):
  this is an upper bound, used as a lower bound in (6.7). The factor `k` is also missing from the definition of `M_k`.
* **Section 6.9 (p. 201).** `eta = (1-(A+1)e^-A)/(A(1-e^-A))`, not `(1-(A-1)e^-A)/(...)`. The final bound of (6.8)
  survives, by the identity `(1-(A+1)x)^2 - (1-x)^2(1-2Ax-x^2) = x^2(A-1+x)^2`.
* **Repairs.** Young's inequality gives `rho(F) >= (1/2) log k - 2` (Theorem 6.4 under `k > e^(8m+4)`,
  `B_m = (16m+9) e^(8m+4)`). Truncating the support in `k-1` variables gives `rho(F) >= log k - 2 log log k - 1/2`
  for `k >= e^7`, which recovers Theorem 6.4 and Corollary 6.5 exactly as stated.
* **Exact values.** `rho(F)` at Granville's choice of `A` is computed by FFT convolution for `5 <= k <= 1000`;
  it exceeds the bound claimed in (6.8), so the gap is in the proof, not the conclusion.
* **Part 2.** The Fourier cutoff `H_j` in Section 9.3 should be `2^j r g l1 l2 / M`; the CRT factorisation in
  Section 11.2 is correct only with `/` read as modular division; misprints on pp. 197 and 207.
  Two further suspected slips (the variance line on p. 200, and the Section 11.2 identity under the modular
  reading) were checked and are correct as printed.

The exact values of `rho(F)` come from floating-point FFT convolution on a fine grid, not interval arithmetic.

| path | content |
|---|---|
| `granville_bounded_intervals_audit.tex`, `.pdf` | the note (4 pages) |
| `code/verify_granville_2015.py` | symbolic checks (sympy), FFT numerics, threshold checks, CRT tests, and the figure (about 10 s) |
| `data/verify_granville_2015.txt` | output of the script |
| `figures/fig_granville_2015_audit.pdf` | Figure 1 |

Requirements: Python 3 with numpy, sympy, matplotlib. Run `python code/verify_granville_2015.py`.
