"""Translation of the centred hole in a ball (Theorem 2 (1)(b) of paper 2).

For Omega = B_R and A_t = B_r(t e_1) we compute lambda_{1,h}(A_t) on a fixed
conforming mesh deformed by x -> x + t eta(|x|) e_1, with eta = 1 near the hole
and eta = 0 near dOmega (same connectivity for every t, so that the small
differences lambda(t) - lambda(0) are not polluted by remeshing noise).
We compare with the exact lambda_1(R, r) and with the sign of
Theta''(0) = (2/N) lambda_1 d lambda_1/dR.
"""
import json
import numpy as np
from common import FIG, RES
import steklov_holes as sh
from steklov_holes import radial
from steklov_holes.plotting import plt


def eta(s, r, R):
    a, b = r + 0.15 * (R - r), r + 0.75 * (R - r)
    x = np.clip((s - a) / (b - a), 0, 1)
    return 1 - x * x * (3 - 2 * x)      # smoothstep


def lam_translated(R, r, h, ts, seed=0):
    m = sh.annulus(h, R, r, seed=seed)
    rad = np.hypot(*m.p.T)
    mid = np.hypot(*m.p[m.bedges].mean(1).T)
    outer = m.bedges[mid > 0.5 * (R + r)]
    inner = np.unique(m.bedges[mid <= 0.5 * (R + r)])
    out = []
    for t in ts:
        D = np.zeros_like(m.p)
        D[:, 0] = t * eta(rad, r, R)
        mt = m.deform(D)
        S = sh.P1System(mt, outer)
        fixed = np.zeros(mt.n_nodes, bool)
        fixed[inner] = True
        lam, _ = sh.eig_dirichlet(S, fixed)
        out.append(lam)
    return m, np.array(out)


cases = [
    # (R, r, h)
    (1.0, 0.2, 0.02),
    (1.0, 0.5, 0.02),
    (3.0, 0.5, 0.05),
    (3.0, 1.0, 0.05),
    (5.0, 0.5, 0.08),
]
results = []
fig, axs = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.0))
for ax, (R, r, h) in zip(axs, cases):
    tmax = 0.5 * (R - r) * 0.3
    ts = np.linspace(0, tmax, 7)
    m, lam = lam_translated(R, r, h, ts)
    ex = float(radial.lambda1(R, r))
    th2 = float(radial.theta2(R, r))
    Rs = radial.R_star(r)
    # quadratic fit lambda(t) = a + b t^2 (even function)
    c = np.polyfit(ts ** 2, lam, 1)
    results.append(dict(R=R, r=r, h=h, nodes=int(m.n_nodes), lambda_exact=ex,
                        lambda_h0=float(lam[0]), R_star=Rs, theta2=th2,
                        lambda_h_second_derivative=float(2 * c[0]),
                        t=ts.tolist(), lambda_h=lam.tolist()))
    print(f"R={R} r={r}: R*={Rs:.4f}  exact={ex:.6f} h0={lam[0]:.6f}  "
          f"Theta''(0)={th2:+.4e}  lambda_h''(0)~{2*c[0]:+.4e}")
    ax.plot(ts, lam - lam[0], "o-", ms=3, label=r"$\lambda_{1,h}(B_r(te_1))-\lambda_{1,h}(B_r)$")
    ax.plot(ts, 0.5 * th2 * ts ** 2, "--", label=r"$\frac{1}{2}\Theta''(0)t^2$")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title(f"$R={R:g}$, $r={r:g}$ ($R^*={Rs:.2f}$)", fontsize=9)
    ax.set_xlabel("$t$")
    ax.grid(alpha=0.3)
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/translation.pdf")
fig.savefig(f"{FIG}/translation.png", dpi=150)
json.dump(results, open(f"{RES}/translation.json", "w"), indent=1)
