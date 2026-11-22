"""Optimal holes: original algorithm vs. the new pipeline, with movies.

For each domain:
  1. original element-swapping algorithm of the paper (one swap per step,
     tabu list), as a baseline;
  2. penalised rearrangement (monotone) -> topology and rough shape;
  3. shape-gradient flow with moving mesh and adaptive remeshing near the
     interface -> smooth, boundary-fitted hole.

Usage:  python optimal_holes.py [disk|lshape|handle|bigdisk ...]
"""
import json
import sys
import time

import numpy as np
from common import FIG, MOV, RES
import steklov_holes as sh
from steklov_holes import algorithms as al
from steklov_holes import plotting as pl
from steklov_holes.mesh import d_circle, d_difference, d_rectangle, d_union


def config(name):
    if name == "disk":
        fd = lambda p: d_circle(p, (0, 0), 1.0)
        return dict(title=r"unit disk, $\alpha=\pi/2$", mesh=lambda h: sh.disk(h, 1.0),
                    fd=fd, bbox=((-1, -1), (1, 1)), alpha=np.pi / 2, h=0.04, h_fine=0.015,
                    init=dict(center=(0.25, 0.1)), corners=None,
                    extra_starts=[(0.0, 0.0), (-0.5, 0.0)])
    if name == "lshape":
        fd = lambda p: d_difference(d_rectangle(p, -1, 1, -1, 1), d_rectangle(p, 0, 2, 0, 2))
        corners = [[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]]
        return dict(title=r"L-shaped domain, $\alpha=1/2$", mesh=lambda h: sh.lshape(h),
                    fd=fd, bbox=((-1, -1), (1, 1)), alpha=0.5, h=0.04, h_fine=0.015,
                    init=dict(center=(-0.5, -0.5)), corners=corners,
                    extra_starts=[(0.5, -0.5), (-0.5, 0.5), (0.0, 0.0)])
    if name == "handle":
        R, gap, w = 1.0, 1.0, 0.5
        cx = R + gap / 2
        fd = lambda p: d_union(d_circle(p, (-cx, 0), R), d_circle(p, (cx, 0), R),
                               d_rectangle(p, -cx, cx, -w / 2, w / 2))
        xr = cx - np.sqrt(R ** 2 - (w / 2) ** 2)
        corners = [[-xr, w / 2], [-xr, -w / 2], [xr, w / 2], [xr, -w / 2]]
        return dict(title=r"two disks with a handle, $\alpha=\pi$",
                    mesh=lambda h: sh.two_disks_with_handle(h, R, gap, w),
                    fd=fd, bbox=((-cx - R, -R), (cx + R, R)), alpha=np.pi, h=0.06, h_fine=0.025,
                    init=dict(center=(-1.2, 0.1)), corners=corners,
                    extra_starts=[(0.0, 0.0), (1.5, 0.0)])
    if name == "bigdisk":
        fd = lambda p: d_circle(p, (0, 0), 3.0)
        return dict(title=r"disk $B_3$, $\alpha=\pi/4$ (small hole, $R>R^*(r)$)",
                    mesh=lambda h: sh.disk(h, 3.0), fd=fd, bbox=((-3, -3), (3, 3)),
                    alpha=np.pi / 4, h=0.08, h_fine=0.03, init=dict(center=(1.5, 0.5)), corners=None,
                    extra_starts=[(0.0, 0.0), (-2.5, 0.0)])
    raise ValueError(name)


def run(name, make_movies=True):
    c = config(name)
    alpha = c["alpha"]
    mesh = c["mesh"](c["h"])
    S = sh.P1System(mesh)
    h0 = al.initial_hole(mesh, alpha, **c["init"])
    lam0, u0 = sh.eig_hard(S, h0)
    out = dict(domain=name, alpha=alpha, nodes=int(mesh.n_nodes), elements=int(mesh.n_elements),
               h=float(mesh.h), lambda_initial=float(lam0))

    t0 = time.time()
    # decreasing number of swaps per step, as suggested in Section 3.4 of the paper
    r_swap = al.swap_algorithm(S, h0, alpha, n_iter=300, n_swap=lambda k: max(1, 10 - k // 20),
                               tabu=10, record=True)
    out["swap"] = dict(lam=r_swap.lam, area=r_swap.area, iters=len(r_swap.history) - 1,
                       time=time.time() - t0, history=r_swap.history)

    t0 = time.time()
    # the starting hole of the original algorithm is one of the starts
    starts = [c["init"]["center"]] + c.get("extra_starts", [])
    r_re = al.multistart(S, alpha, centers=starts, record=True)
    out["rearrangement"] = dict(lam=r_re.lam, area=r_re.area, frames=len(r_re.info["frames"]),
                                time=time.time() - t0, hard_per_beta=r_re.info["hard_per_beta"])

    t0 = time.time()
    r_fl = al.shape_gradient_flow(mesh, r_re.hole, alpha, n_iter=150, smooth=25, patience=15,
                                  record=True,
                                  remesh_args=dict(fd=c["fd"], bbox=c["bbox"], h=c["h"],
                                                   h_fine=c["h_fine"], corners=c["corners"]))
    out["flow"] = dict(lam=r_fl.lam, area=r_fl.area, iters=len(r_fl.history) - 1,
                       remeshes=r_fl.info["remeshes"], nodes=int(r_fl.mesh.n_nodes),
                       time=time.time() - t0, history=r_fl.history)

    print(f"[{name}] initial {lam0:.5f} | swap {r_swap.lam:.5f} | rearr {r_re.lam:.5f} | "
          f"flow {r_fl.lam:.5f} (remeshes {r_fl.info['remeshes']})")

    states = [(mesh.p, mesh.t, h0, u0), (mesh.p, mesh.t, r_swap.hole, r_swap.u),
              (mesh.p, mesh.t, r_re.hole, r_re.u), (r_fl.mesh.p, r_fl.mesh.t, r_fl.hole, r_fl.u)]
    titles = [rf"initial: $\lambda={lam0:.4f}$", rf"original alg.: $\lambda={r_swap.lam:.4f}$",
              rf"rearrangement: $\lambda={r_re.lam:.4f}$", rf"shape flow: $\lambda={r_fl.lam:.4f}$"]
    pl.plot_holes_panel(states, titles, f"{FIG}/holes_{name}.pdf")
    pl.plot_holes_panel(states, titles, f"{FIG}/holes_{name}.png")

    # convergence histories
    fig, ax = pl.plt.subplots(figsize=(5, 3.2))
    ax.plot(r_swap.history, lw=1, label="original algorithm")
    lam_re = [f["lam"] for f in r_re.info["frames"]]
    ax.plot(lam_re, "o-", ms=3, lw=1, label="rearrangement")
    ax.plot(np.arange(len(r_fl.history)) + len(lam_re) - 1, r_fl.history, lw=1,
            label="rearrangement + shape flow")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\lambda_{1,h}$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/history_{name}.pdf")
    pl.plt.close(fig)

    import gzip, pickle
    with gzip.open(f"{RES}/frames_{name}.pkl.gz", "wb") as fh:
        pickle.dump(dict(original=r_swap.info["frames"],
                         new=r_re.info["frames"] + r_fl.info["frames"], title=c["title"]), fh)
    if make_movies:
        pl.make_movie(r_swap.info["frames"], f"{MOV}/{name}_original_algorithm.mp4", fps=15,
                      title=c["title"] + " — original algorithm")
        frames = r_re.info["frames"] + r_fl.info["frames"]
        pl.make_movie(frames, f"{MOV}/{name}_rearrangement_and_flow.mp4", fps=4,
                      title=c["title"] + " — new method")
    json.dump(out, open(f"{RES}/holes_{name}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["disk", "lshape", "handle", "bigdisk"]
    for n in names:
        run(n)
