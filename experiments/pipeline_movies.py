"""Movies comparing the two complete pipelines from the same initial hole:

    Alg. 1 (element swapping) -> Alg. 3 (shape flow), and
    Alg. 2 (penalised rearrangement) -> Alg. 3 (shape flow).

Alg. 1 is stopped when the best eigenvalue has not improved during 40
iterations; Alg. 2 stops by itself.  Alg. 3 then starts from the best hole.
Three panels: hole (pipeline 1), hole (pipeline 2), both eigenvalue curves.

Usage:  python pipeline_movies.py [disk|lshape|handle|bigdisk ...]
"""
import json
import sys

from common import MOV, RES
import steklov_holes as sh
from steklov_holes import algorithms as al
from steklov_holes import plotting as pl
from optimal_holes import config

N_SWAP = lambda k: max(1, 10 - k // 20)
MOVIE_START = {"handle": (0.0, 0.0)}   # the default start already fills one disk


def run(name):
    c = config(name)
    alpha = c["alpha"]
    mesh = c["mesh"](c["h"])
    S = sh.P1System(mesh)
    # start of the comparison movies, except where a more informative one is chosen
    start = MOVIE_START.get(name) or tuple(json.load(open(f"{RES}/holes_{name}.json"))["start_movie"])
    h0 = al.initial_hole(mesh, alpha, center=start)
    remesh_args = dict(fd=c["fd"], bbox=c["bbox"], h=c["h"], h_fine=c["h_fine"], corners=c["corners"])
    flow_kw = dict(n_iter=150, smooth=25, patience=15, record=True, remesh_args=remesh_args)

    sw = al.swap_algorithm(S, h0, alpha, n_iter=300, n_swap=N_SWAP, tabu=10, patience=40, record=True)
    T1 = sw.info["time"]
    fa = sw.info["frames"]
    fa.append(dict(fa[-1], hole=sw.hole.copy(), u=sw.u.copy(), lam=float(sw.lam),
                   area=float(mesh.areas[sw.hole].sum()), label="swap (best hole)", time=T1))
    fl_a = al.shape_gradient_flow(mesh, sw.hole, alpha, **flow_kw)
    fa = fa + [dict(f, time=f["time"] + T1) for f in fl_a.info["frames"]]
    total_a = T1 + fl_a.info["time"]

    re = al.rearrangement(S, h0, alpha, record=True)
    T2 = re.info["time"]
    fl_b = al.shape_gradient_flow(mesh, re.hole, alpha, **flow_kw)
    fb = re.info["frames"] + [dict(f, time=f["time"] + T2) for f in fl_b.info["frames"]]
    total_b = T2 + fl_b.info["time"]

    summary = dict(domain=name, start=list(start),
                   alg1_iters=len(sw.history) - 1, alg1=float(sw.lam), alg1_time=T1,
                   alg1_flow=float(fl_a.lam), alg1_flow_time=fl_a.info["time"], alg1_total_time=total_a,
                   alg2_iters=len(re.info["frames"]) - 1, alg2=float(re.lam), alg2_time=T2,
                   alg2_flow=float(fl_b.lam), alg2_flow_time=fl_b.info["time"], alg2_total_time=total_b)
    print(summary)
    json.dump(summary, open(f"{RES}/pipeline_{name}.json", "w"), indent=1)
    pl.make_pipeline_movie(fa, fb, f"{MOV}/{name}_pipelines.mp4", totals=(total_a, total_b),
                           title=c["title"] + "  —  same initial hole", duration=20.0)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["lshape", "handle", "disk", "bigdisk"]:
        run(n)
