"""Assemble the figures of the annulus and ball experiments for paper 2."""
import numpy as np
from common import FIG, RES
from steklov_holes import plotting as pl
from steklov_holes.plotting import plt

cases = [("annulus10", "annulus, 10%"), ("annulus25", "annulus, 25%"), ("annulus50", "annulus, 50%"),
         ("ball10", r"ball $B_1$, 10%"), ("ball25", r"ball $B_1$, 25%"), ("ball50", r"ball $B_1$, 50%"),
         ("ball3_1", r"ball $B_3$, 1%")]
# plane sections (meridian sections reflected for the balls)
fig, axs = plt.subplots(2, len(cases), figsize=(2.5 * len(cases), 5.2))
for j, (name, ttl) in enumerate(cases):
    d = np.load(f"{RES}/{name}_final.npz")
    mir = bool(d["axisymmetric"])
    pl.draw_hole(axs[0, j], d["p"], d["t"], d["hole"], mirror=mir)
    axs[0, j].set_title(f"{ttl}\n" + rf"$\lambda_{{1,h}}={float(d['lam']):.5f}$", fontsize=9)
    pl.draw_eigenfunction(axs[1, j], d["p"], d["t"], d["u"], mirror=mir)
fig.tight_layout()
fig.savefig(f"{FIG}/annulus_ball.png", dpi=200)
fig.savefig(f"{FIG}/annulus_ball.pdf")
plt.close(fig)

# 3D views of the optimal holes in the balls
balls = [c for c in cases if c[0].startswith("ball")]
fig = plt.figure(figsize=(4.0 * len(balls), 4.8))
for j, (name, ttl) in enumerate(balls):
    d = np.load(f"{RES}/{name}_final.npz")
    ax = fig.add_subplot(1, len(balls), j + 1, projection="3d")
    R = 3.0 if name == "ball3_1" else 1.0
    pl.plot_axisym_hole_3d(d["p"], d["t"], d["hole"], ax=ax, radius=R,
                           title=f"{ttl}\n" + rf"$\lambda_{{1,h}}={float(d['lam']):.5f}$")
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.86, wspace=0.0)
fig.savefig(f"{FIG}/balls_3d.png", dpi=200)
plt.close(fig)
print("figures written")
