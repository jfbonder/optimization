# Differences with the published version

**Published as:** J. Fernández Bonder, J. D. Rossi, N. Wolanski, *On the best Sobolev trace constant and extremals in domains with holes*, Bull. Sci. Math. **130** (2006), no. 7, 565–579.
**Base text:** final manuscript `bon-ros-wol5.pdf`, whose title was *Behavior of the best Sobolev trace constant and extremals in domains with holes*.

The same list appears, with cross-references, at the beginning of the Introduction of `main.pdf`.

1. **Theorem 1.6 (Hausdorff continuity) is corrected.**
   - *Published claim:* for every $p$, if $d_H(A_n,A)\to0$ then $\mathcal S_{A_n}\to\mathcal S_A$ and the extremals converge.
   - *Why it fails:* the claim is false for $1<p\le N$. Take $A$ a closed ball and $A_n$ finite $1/n$-nets of $A$. Each $A_n$ has zero capacity, so $\mathcal S_{A_n}=S_\emptyset$, while $\mathcal S_A>S_\emptyset$ (new Remark 1.1). The gap in the published proof is the step "by symmetry": approximating an extremal of $\mathcal S_{A_n}$ by functions vanishing near $A$ cannot be done uniformly in $n$.
   - *New statement:*
     - (a) $\limsup\mathcal S_{A_n}\le\mathcal S_A$ for every $p$, with the published proof;
     - (b) continuity and convergence of extremals if $p>N$ (via Morrey's embedding and spectral synthesis) or if $A\subset A_n$ (closed subspaces), with a new proof.
   - The abstract and the introduction were changed accordingly.
2. **Capacity.**
   - *Problem:* with the homogeneous capacity $\inf\int|\nabla\phi|^p$ used in (1.6), every bounded set has zero capacity when $p=N$, so Theorem 1.5 fails for $p=N$. In addition, the proof (via the Gagliardo–Nirenberg–Sobolev inequality) required $p<N$.
   - *Change:* we now use the Sobolev capacity $\inf\|\phi\|^p_{W^{1,p}(\mathbb R^N)}$. For $p<N$ it has the same null sets, so the statements do not change in that range.
   - The proof of Theorem 1.5 becomes simpler, and the proof of Corollary 1.1 now also controls $\|\phi_n\|_{L^p}$.
3. **Proof of Theorem 1.4 (symmetry).**
   - *Problem:* the zero set of the spherical symmetrization $u^*$ consists of caps centred at $-re_N$, not at $re_N$, so $(u^*,A^*)$ is not directly admissible as the published proof claimed.
   - *Change:* the symmetric hole is defined as $\{u^*=0\}$ (after a reflection), and the fact that it has measure exactly $\alpha$ is deduced from Theorem 2.1.
   - The statement does not change.
4. **Proof of Theorem 2.1 (existence).**
   - The densities $\phi_n$ converge weakly-$*$ in $L^\infty$ instead of weakly in $L^{p'}$; this is what preserves $\int\phi\ge\alpha$.
   - In the final chain of inequalities, $S(\alpha_1)$ (undefined there) is replaced by $S(|D_1|)$.
5. **Minor points.**
   - $1<p<\infty$ and $p_*=\infty$ for $p\ge N$ are now explicit.
   - We clarified that $C^\infty_0(\overline\Omega\setminus A)$ means functions smooth up to $\partial\Omega$ that vanish near $A$.
   - The notation $\mathcal S_A$ for the capacitary constant is now distinct from $S_A$.
   - Remark 3.1 was made precise.
   - Typos ("Haussdorff", "definded", "comvexity", "dependance") and references were corrected.
