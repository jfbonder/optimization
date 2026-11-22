"""Second variation of lambda_1 at the centred ball B_r in B_R (N = 2).

Produces the regime diagram in the (r, R) plane:
  * R < R*(r): d lambda_1/dR < 0, c_1 < 0: the centred ball is a strict local
    maximum among translated balls, hence not optimal (corrected Theorem 2);
  * R > R*(r): c_ell > 0 for all ell >= 1 (checked for ell <= 40): the centred
    ball is a non-degenerate critical point with positive second variation.
"""
import json
import numpy as np
from common import FIG, RES
from steklov_holes import radial
from steklov_holes.plotting import plt

rs = np.linspace(0.05, 2.5, 60)
Rstar = np.array([radial.R_star(r) for r in rs])

# minimal c_ell, ell=1..40 on a grid above and below R*
grid = []
for r in rs[::6]:
    for fac in [1.02, 1.1, 1.5, 2.0, 3.0]:
        R = radial.R_star(r) * fac
        cs = [radial.c_ell(l, R, r) for l in range(1, 41)]
        grid.append(dict(r=float(r), R=float(R), min_c=float(min(cs)), argmin=int(np.argmin(cs) + 1)))
    for fac in [0.98, 0.9, 0.7]:
        R = max(r * 1.05, radial.R_star(r) * fac)
        cs = [radial.c_ell(l, R, r) for l in range(1, 41)]
        grid.append(dict(r=float(r), R=float(R), c1=float(cs[0]), min_c_ell_ge2=float(min(cs[1:]))))
above = [g for g in grid if "min_c" in g]
print("above R*: min over grid of min_ell c_ell =", min(g["min_c"] for g in above),
      " attained at ell =", {g["argmin"] for g in above})

fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
ax = axs[0]
h1 = ax.fill_between(rs, rs, Rstar, color="C3", alpha=0.15, label=r"$R<R^*(r)$: centred ball not optimal")
h2 = ax.fill_between(rs, Rstar, 3 * Rstar.max(), color="C0", alpha=0.12,
                label=r"$R>R^*(r)$: $c_\ell>0$ for all $\ell\geq1$")
h3, = ax.plot(rs, Rstar, "k", lw=1.5, label=r"$R^*(r)$")
ax.plot(rs, rs, "k:", lw=0.8)
for (R, r, mk) in [(1, 0.2, "s"), (1, 0.5, "s"), (3, 1, "s"), (3, 0.5, "o"), (5, 0.5, "o"), (1, np.sqrt(0.5), "^")]:
    ax.plot(r, R, mk, color="k", ms=5)
ax.set_xlim(0, rs.max())
ax.set_ylim(0, 6)
ax.set_xlabel("$r$ (radius of the hole)")
ax.set_ylabel("$R$ (radius of $\\Omega$)")
ax.legend(handles=[h1, h2, h3], fontsize=7, loc="upper left")
ax.set_title("regimes for the centred hole ($N=2$)", fontsize=10)

ax = axs[1]
for r in [0.2, 0.5, 1.0]:
    R = np.linspace(r * 1.02, 8, 400)
    ax.plot(R, radial.lambda1(R, r), label=f"$r={r}$")
    Rs = radial.R_star(r)
    ax.plot(Rs, radial.lambda1(Rs, r), "ko", ms=3)
ax.axhline(1, color="0.5", lw=0.6, ls="--")
ax.set_ylim(0.4, 2.5)
ax.set_xlabel("$R$")
ax.set_ylabel(r"$\lambda_1(R,r)$")
ax.legend(fontsize=8)
ax.set_title(r"$\lambda_1(R,r)$ and its minimum at $R^*(r)$", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG}/regimes.pdf")
fig.savefig(f"{FIG}/regimes.png", dpi=150)

table = [dict(r=r, R_star=radial.R_star(r)) for r in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]]
json.dump(dict(grid=grid, R_star=table), open(f"{RES}/second_variation.json", "w"), indent=1)
for t in table:
    print(f"r={t['r']:.2f}  R*={t['R_star']:.6f}")
