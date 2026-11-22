"""Experiments on the annulus (R^2) and on the ball of R^3 (axisymmetric).

* Annulus Omega = {0.5 < |x| < 1}, Steklov condition on the whole boundary
  (both circles); hole volumes 10%, 25% and 50% of |Omega|.
* Ball B_1 in R^3: by Theorem 1.4 of [FBRW1] (p = 2) every optimal hole is
  spherically symmetric with respect to an axis, so the problem reduces to an
  axisymmetric one on the meridian half-disk {(r, z): r >= 0, r^2 + z^2 < 1},
  with all integrals weighted by 2 pi r (trace on the meridian semicircle).
  Volumes 10%, 25% and 50% of |B_1|.
* Ball B_3 in R^3 with a small hole (1% of the volume): here R > R*(r), the
  regime of Conjecture 2.8, and the centred ball is expected to be optimal.

For each case both discrete algorithms are run from several initial holes, the
best hole goes to the shape flow, and a pipeline movie (Alg. 1 -> 3 vs
Alg. 2 -> 3, normalized time) is made from one common initial hole.

Usage:  python annulus_ball.py [case ...]      (cases: see CASES below)
"""
import json
import sys

import numpy as np
from common import FIG, MOV, RES
import steklov_holes as sh
from steklov_holes import algorithms as al
from steklov_holes import plotting as pl
from steklov_holes.mesh import d_circle, d_difference, d_intersection

N_SWAP = lambda k: max(1, 10 - k // 20)


def annulus_case(frac):
    R, rho = 1.0, 0.5
    vol = np.pi * (R ** 2 - rho ** 2)
    fd = lambda p: d_difference(d_circle(p, (0, 0), R), d_circle(p, (0, 0), rho))
    return dict(title=rf"annulus $0.5<|x|<1$, $\alpha={int(100 * frac)}\%$ of $|\Omega|$",
                mesh=lambda h: sh.annulus(h, R, rho, graded=False), fd=fd, bbox=((-R, -R), (R, R)),
                alpha=frac * vol, h=0.035, h_fine=0.015, corners=None, mirror=False,
                starts=[(0.75, 0.0), (0.55, 0.0), (0.95, 0.0), (0.0, 0.75)], movie_start=(0.0, 0.75))


def ball_case(frac, R=1.0, h=0.04, h_fine=0.015, starts=None, movie_start=None):
    vol = 4 / 3 * np.pi * R ** 3
    fd = lambda p: d_intersection(d_circle(p, (0, 0), R), -p[:, 0])
    starts = starts or [(0.0, 0.0), (0.0, 0.8 * R), (0.5 * R, 0.0), (0.0, -0.5 * R)]
    return dict(title=rf"ball $B_{{{R:g}}}\subset\mathbb{{R}}^3$, $\alpha={frac * 100:g}\%$ of $|B|$ (meridian section)",
                mesh=lambda hh: sh.half_disk(hh, R), fd=fd, bbox=((0, -R), (R, R)), alpha=frac * vol,
                h=h, h_fine=h_fine, corners=[[0.0, -R], [0.0, R]], mirror=True,
                starts=starts, movie_start=movie_start or starts[2])


CASES = {
    "annulus10": annulus_case(0.10), "annulus25": annulus_case(0.25), "annulus50": annulus_case(0.50),
    "ball10": ball_case(0.10), "ball25": ball_case(0.25), "ball50": ball_case(0.50),
    "ball3_1": ball_case(0.01, R=3.0, h=0.1, h_fine=0.04,
                         starts=[(0.0, 0.0), (0.0, 2.0), (1.5, 0.0)], movie_start=(0.0, 2.0)),
}


def run(name):
    c = CASES[name]
    alpha = c["alpha"]
    mesh = c["mesh"](c["h"])
    S = sh.P1System(mesh)
    remesh_args = dict(fd=c["fd"], bbox=c["bbox"], h=c["h"], h_fine=c["h_fine"], corners=c["corners"])
    flow_kw = dict(n_iter=150, smooth=25, patience=15, remesh_args=remesh_args)

    rows, best = [], None
    for st in c["starts"]:
        h0 = al.initial_hole(mesh, alpha, center=st)
        lam0, _ = sh.eig_hard(S, h0)
        sw = al.swap_algorithm(S, h0, alpha, n_iter=300, n_swap=N_SWAP, tabu=10, patience=40)
        re = al.rearrangement(S, h0, alpha)
        rows.append(dict(start=list(st), lam_initial=float(lam0), lam_swap=float(sw.lam),
                         time_swap=sw.info["time"], lam_rearr=float(re.lam), time_rearr=re.info["time"]))
        for res, meth, t in ((sw, "swap", sw.info["time"]), (re, "rearrangement", re.info["time"])):
            if best is None or res.lam < best[0].lam:
                best = (res, meth, st, t)
        print(f"[{name}] start {st}: initial {lam0:.5f} | Alg.1 {sw.lam:.5f} ({sw.info['time']:.1f}s) "
              f"| Alg.2 {re.lam:.5f} ({re.info['time']:.1f}s)")
    fl = al.shape_gradient_flow(mesh, best[0].hole, alpha, **flow_kw)
    out = dict(case=name, alpha=alpha, nodes=int(mesh.n_nodes), runs=rows, best_discrete=float(best[0].lam),
               best_method=best[1], best_start=list(best[2]), flow=float(fl.lam), flow_area=fl.area,
               time_flow=fl.info["time"], remeshes=fl.info["remeshes"])
    print(f"[{name}] best discrete {best[0].lam:.5f} ({best[1]}, start {best[2]}) -> flow {fl.lam:.5f} "
          f"(area {fl.area:.5f} vs {alpha:.5f}, {fl.info['time']:.1f}s)")
    pl.plot_holes_panel([(fl.mesh.p, fl.mesh.t, fl.hole, fl.u)], [rf"$\lambda_{{1,h}}={fl.lam:.5f}$"],
                        f"{FIG}/{name}.png", mirror=c["mirror"], size=3.2)
    np.savez_compressed(f"{RES}/{name}_final.npz", p=fl.mesh.p, t=fl.mesh.t, hole=fl.hole, u=fl.u,
                        lam=fl.lam, axisymmetric=c["mirror"])
    if c["mirror"]:
        R = c["bbox"][1][0]
        pl.plot_axisym_hole_3d(fl.mesh.p, fl.mesh.t, fl.hole, f"{FIG}/{name}_3d.png", radius=R,
                               title=rf"$\lambda_{{1,h}}={fl.lam:.5f}$")

    # pipeline movie from a common initial hole
    h0 = al.initial_hole(mesh, alpha, center=c["movie_start"])
    sw = al.swap_algorithm(S, h0, alpha, n_iter=300, n_swap=N_SWAP, tabu=10, patience=40, record=True)
    T1 = sw.info["time"]
    fa = sw.info["frames"] + [dict(sw.info["frames"][-1], hole=sw.hole.copy(), u=sw.u.copy(),
                                   lam=float(sw.lam), label="swap (best hole)", time=T1)]
    fla = al.shape_gradient_flow(mesh, sw.hole, alpha, record=True, **flow_kw)
    fa += [dict(f, time=f["time"] + T1) for f in fla.info["frames"]]
    re = al.rearrangement(S, h0, alpha, record=True)
    T2 = re.info["time"]
    flb = al.shape_gradient_flow(mesh, re.hole, alpha, record=True, **flow_kw)
    fb = re.info["frames"] + [dict(f, time=f["time"] + T2) for f in flb.info["frames"]]
    out["movie"] = dict(start=list(c["movie_start"]), alg1_flow=float(fla.lam), alg1_total=T1 + fla.info["time"],
                        alg2_flow=float(flb.lam), alg2_total=T2 + flb.info["time"])
    pl.make_pipeline_movie(fa, fb, f"{MOV}/{name}_pipelines.mp4",
                           totals=(T1 + fla.info["time"], T2 + flb.info["time"]),
                           title=c["title"] + "  —  same initial hole", duration=20.0)
    json.dump(out, open(f"{RES}/{name}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or list(CASES):
        run(n)
