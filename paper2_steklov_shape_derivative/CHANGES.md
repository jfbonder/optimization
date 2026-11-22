# Differences with the published version

**Published as:** J. Fernández Bonder, P. Groisman, J. D. Rossi, *Optimization of the first Steklov eigenvalue in domains with holes: a shape derivative approach*, Ann. Mat. Pura Appl. (4) **186** (2007), no. 2, 341–358, DOI 10.1007/s10231-006-0009-y.
**Base text:** final manuscript `aunum-final.pdf`.

The same list appears, with cross-references, at the beginning of the Introduction of `main.pdf`. Numbers refer to the revised version unless stated otherwise. The published version numbered lemmas together with the equations, so for example Lemma 2.7 and Corollary 2.13 there correspond to Lemma 2.2 and Lemma 2.4 here.

1. **Theorem 2 (1)(b) is corrected, and (1)(c) is new.**
   - *Published claim:* the centred ball $B_r$ is never optimal in $B_R$.
   - *Why it fails:* the proof needs $\lambda_1^2+\frac{N-1}{R}\lambda_1>1$, but Corollary 2.13 only proves $\lambda_1^2+\frac{N-1}{r}\lambda_1>1$, which is weaker. The needed inequality holds **if and only if** $R<R^*(r)$, the unique minimum point of $\lambda_1(\cdot,r)$ (new Lemma 2.4).
   - *New statement:*
     - (b) for $R<R^*(r)$ the centred ball is a strict local maximum among translated balls, hence not optimal;
     - (c) for $R>R^*(r)$ it is a strict local minimum among translated balls, so the published claim is false in that regime.
   - Corollary 2.13 is replaced by Lemma 2.4. The abstract and the introduction were changed accordingly.
2. **Explicit formula (Lemma 2.3).** The published text says that $\lambda_1(R,r)$ cannot be found explicitly. In fact
   $$\lambda_1(R,r)=\frac{K_\nu(r)I_{\nu+1}(R)+I_\nu(r)K_{\nu+1}(R)}{K_\nu(r)I_\nu(R)-I_\nu(r)K_\nu(R)},\qquad\nu=\tfrac{N-2}{2}.$$
3. **Second variation at the centred ball (new Proposition 2.6).**
   - For volume preserving perturbations $\rho=r+\varepsilon h$:
     $$\lambda_1(A_\varepsilon)=\lambda_1(B_r)+\varepsilon^2\varphi'(r)^2\sum_{\ell\ge1}c_\ell h_\ell^2+o(\varepsilon^2),\qquad c_\ell=-\mu_\ell-\tfrac{N-1}{r},$$
     with $\mu_\ell$ given explicitly by Bessel functions.
   - $c_1$ has the sign of $\partial_R\lambda_1(R,r)$ (via the Wronskian of $\varphi'$ and the $\ell=1$ mode), so the threshold $R^*(r)$ is sharp.
   - $c_\ell=(\ell-1)/r+o(1)$ as $\ell\to\infty$.
   - New Conjecture 2.8: for $R>R^*(r)$ the centred ball is the optimal hole. It is supported by the numerics, and a possible strategy is sketched.
4. **Proof of Lemma 2.2 (differentiability; Lemma 2.7 in the published numbering).**
   - *Problem:* the continuity of $t\mapsto(\lambda_1(A_t),\bar u_t)$ was attributed to Theorem 1.6 of [11] (Hausdorff continuity), which is false in general.
   - *Change:* it is now proved directly, using that the deformation is smooth. The Fredholm argument is also written out.
5. **Proof of Theorem 3 (FEM).**
   - *Problem:* the construction of the discrete competitors relied on an interpolation estimate not justified near the free boundary.
   - *Change:* it now uses $(u_\alpha-\delta)^+$, the global Lipschitz bound on $u_\alpha$ and the $W^{1,p}$-stability of the interpolant ($p>N$). The convergence of the holes is deduced explicitly from Lemma 3.1.
   - The statements do not change.
6. **Section 3 is rewritten.**
   - The original algorithm is kept as Algorithm 1, with the improvements suggested there: several swaps per step at the beginning, and a tabu rule.
   - New Algorithm 2: monotone penalized rearrangement (Proposition 3.2).
   - New Algorithm 3: shape-gradient flow with moving mesh and adaptive remeshing near the hole.
   - The eigenvalue problems are solved with Lanczos instead of `fmincon`.
   - The implementation of Algorithm 3 was optimized after profiling, which showed that remeshing dominated its cost. The changes are a size function evaluated on a background grid, a cap on the DistMesh iterations, and a stopping rule based on stalls right after remeshing; together they make the flow 3 to 10 times faster.
   - New experiments on an annulus (Steklov condition on both circles) and on the balls of $\mathbb R^3$, computed through an exact axisymmetric reduction justified by Theorem 1.4 of [11]. In the annulus the optimal holes are sectors spanning the ring. In $B_1$ they are polar caps. In $B_3$ with a hole of 1 % of the volume the optimum is the centred ball, which supports the conjecture in dimension 3.
   - The code is publicly available at https://github.com/jfbonder/optimization. It is cited in the Introduction and in Section 3.3, and described in a new "Code availability" paragraph before the acknowledgements.
   - All the numerical results are new (Python implementation in `steklov_holes/`). Figures 1–5 and Table 1 of the published version are replaced by new figures and tables, including the translation of the centred ball and the regime diagram.
7. **Minor points.**
   - Typos were corrected ("Sobloev", "design and algorithm", "Lema", "looses").
   - [11], [12] and Henrot–Pierre are cited in their published form; Dambrine–Pierre, Flucher, Kao–Su, Persson–Strang and Reichel were added.
