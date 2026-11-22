# Differences with the published version

**Published as:** J. Fernández Bonder, J. D. Rossi, N. Wolanski, *Regularity of the free boundary in an optimization problem related to the best Sobolev trace constant*, SIAM J. Control Optim. **44** (2006), no. 5, 1614–1635.
**Base text:** final manuscript `bonder.pdf`.

The same list appears, with cross-references, at the beginning of the Introduction of `main.pdf`. The results and the regularity theory (à la Alt–Caffarelli) are those of the published version.

1. **Sign condition (a clarification of hypotheses).**
   - *Change:* the condition $v\ge0$ was added to the admissible classes $\mathcal A_\alpha$, $K$, $K_1$, $K_2$ and $K_\alpha$.
   - *Why:* it was implicit in the published version, which uses it in several places (subharmonicity of minimizers, Lemmas 2.1–2.2, Theorem 2.2), but it is not part of the definitions there. Without it, $v=-u_\emptyset$ minimizes $J_\varepsilon$ in $K_1$ with $|\{v>0\}|=0$ for every $\varepsilon$, so Theorems 5.2 and 5.3 are false as stated (new Remark 1.1).
   - Equivalently, one can measure $|\{v\ne0\}|$, since $J(|v|)=J(v)$.
   - With the sign condition all the proofs go through. The only argument that has to be adapted is the one-sided perturbation of Remark 2.1, which now uses $(u-t\varphi)^+$ (valid because $J(w^+)\le J(w)$).
2. **Lemma 2.2.**
   - The smallness condition is now $C_2C_3\gamma<\min(1,\varepsilon)$ instead of $C_2C_3\gamma<1$, because the left hand side of (2.11) contains $\varepsilon|B_\kappa\cap\{\bar u>0\}|$. Only constants that already depended on $\varepsilon$ change.
   - The proof of Theorem 2.1(2) was written out in more detail.
3. **Theorem 2.5.**
   - Near a free boundary point the Lipschitz bound gives $u\le C\rho_k$, not $C\rho_k^N$ (a misprint).
   - The difference of the $L^2$ norms is therefore $O(\rho_k^{N+2})$, and the argument is unchanged.
4. **Section 4.**
   - Lemma 4.1: the uniform bound on $\|u_\varepsilon\|_{H^1}$ is justified.
   - Lemma 4.2: the set $D$ is taken connected (needed for the relative isoperimetric inequality), with $|\Omega\setminus D|<c/2$, and a sign is corrected ($\lambda(\varphi)=-\int\nabla u\nabla\varphi+u\varphi$).
   - Lemma 4.3:
     - $\partial_\nu v_r=-\lambda_\varepsilon+o(1)$ instead of $o(\delta_r)$;
     - $B_r(x_0)$ instead of $B_\kappa(x_0)$;
     - $D_{\delta_r}$ instead of $D_r$;
     - final constant $c/K$.
5. **Section 5.**
   - Theorem 5.2: the reference for the strict monotonicity of $S$ is explicit (in [11] it is not "Remark 2.2"). We point out that here $\alpha$ is the measure of the positivity set, while in [11] it is the measure of the hole.
   - The second case of Theorem 5.1 is written out.
   - Corollary 5.1 refers to the precise symmetry result of [11].
6. **Minor points.**
   - In Theorem 2.7 we added the justification of $\Delta|\nabla u|\ge|\nabla u|$.
   - The constant $\alpha(N)$ was renamed $a_N$ and the Bessel factor $J(r)$ became $\mathbb J(r)$, to avoid clashes with $\alpha$ and $\mathcal J$.
   - Typos were corrected ("reduce boundary", "Lispchitz", "ellptic", "comvexity", "de").
   - [11] is cited in its published form, and the volume of [5] (Alt–Caffarelli–Friedman) was corrected.
