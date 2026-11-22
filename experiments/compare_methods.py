"""Original algorithm vs. new method, run from the same initial holes.

For each domain and each initial hole we run
  * Algorithm 1 (element swapping, the algorithm of the original paper),
  * Algorithm 2 (penalised rearrangement),
and then Algorithm 3 (shape-gradient flow) from the best discrete hole.
Times are wall-clock times of the computations (no frames are recorded).
The movies are produced by pipeline_movies.py.

Usage:  python compare_methods.py [disk|lshape|handle|bigdisk ...]
"""
import gzip
import json
import pickle
import sys
import time

import numpy as np
from common import FIG, MOV, RES
import steklov_holes as sh
from steklov_holes import algorithms as al
from steklov_holes import plotting as pl
from optimal_holes import config

N_SWAP = lambda k: max(1, 10 - k // 20)


def run(name):
    c = config(name)
    alpha = c["alpha"]
    mesh = c["mesh"](c["h"])
    S = sh.P1System(mesh)
    starts = [c["init"]["center"]] + c.get("extra_starts", [])
    rows, runs = [], []
    for st in starts:
        h0 = al.initial_hole(mesh, alpha, center=st)
        lam0, u0 = sh.eig_hard(S, h0)
        sw = al.swap_algorithm(S, h0, alpha, n_iter=300, n_swap=N_SWAP, tabu=10, patience=40)
        t_sw = sw.info["time"]
        re = al.rearrangement(S, h0, alpha)
        t_re = re.info["time"]
        rows.append(dict(start=list(map(float, st)), lam_initial=float(lam0), lam_swap=float(sw.lam),
                         iters_swap=len(sw.history) - 1, time_swap=t_sw, lam_rearr=float(re.lam),
                         iters_rearr=len(re.history), time_rearr=t_re))
        runs.append((h0, u0, lam0, sw, re))
        print(f"[{name}] start {st}: initial {lam0:.5f} | original {sw.lam:.5f} ({t_sw:.1f}s) "
              f"| rearrangement {re.lam:.5f} ({t_re:.1f}s)")
    best_new = min(r["lam_rearr"] for r in rows)
    # among the starts where the new method attains its best value, show the one
    # with the largest initial eigenvalue (the most informative evolution)
    ties = [i for i, r in enumerate(rows) if r["lam_rearr"] <= best_new * (1 + 1e-6)]
    i_new = max(ties, key=lambda i: rows[i]["lam_initial"])
    i_sw = int(np.argmin([r["lam_swap"] for r in rows]))
    remesh_args = dict(fd=c["fd"], bbox=c["bbox"], h=c["h"], h_fine=c["h_fine"], corners=c["corners"])
    # shape flow from the best discrete hole of each method
    flows = {}
    for key, i, res in (("new", i_new, runs[i_new][4]), ("original", i_sw, runs[i_sw][3])):
        fl = al.shape_gradient_flow(mesh, res.hole, alpha, n_iter=150, smooth=25, patience=15,
                                    remesh_args=remesh_args)
        flows[key] = (fl, fl.info["time"])
    fl, t_fl = flows["new"]
    out = dict(domain=name, alpha=alpha, nodes=int(mesh.n_nodes), runs=rows,
               best_swap=rows[i_sw]["lam_swap"], best_rearr=rows[i_new]["lam_rearr"],
               flow_from_rearr=float(fl.lam), flow_from_swap=float(flows["original"][0].lam),
               time_flow_rearr=t_fl, time_flow_swap=flows["original"][1],
               total_time_rearr_flow=rows[i_new]["time_rearr"] + t_fl,
               total_time_swap_flow=rows[i_sw]["time_swap"] + flows["original"][1],
               remeshes=fl.info["remeshes"], start_movie=rows[i_new]["start"],
               flow_history=fl.history)
    print(f"[{name}] best original {out['best_swap']:.5f} | best rearrangement {out['best_rearr']:.5f} | "
          f"flow (from rearr) {fl.lam:.5f} | flow (from original) {out['flow_from_swap']:.5f}")

    h0, u0, lam0, sw, re = runs[i_new]
    states = [(mesh.p, mesh.t, h0, u0), (mesh.p, mesh.t, sw.hole, sw.u),
              (mesh.p, mesh.t, re.hole, re.u), (fl.mesh.p, fl.mesh.t, fl.hole, fl.u)]
    titles = [rf"initial: $\lambda={lam0:.4f}$", rf"Alg. 1 (original): $\lambda={sw.lam:.4f}$",
              rf"Alg. 2: $\lambda={re.lam:.4f}$", rf"Alg. 2 + Alg. 3: $\lambda={fl.lam:.4f}$"]
    pl.plot_holes_panel(states, titles, f"{FIG}/holes_{name}.pdf")
    pl.plot_holes_panel(states, titles, f"{FIG}/holes_{name}.png")

    json.dump(out, open(f"{RES}/holes_{name}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or ["disk", "lshape", "handle", "bigdisk"]:
        run(n)
