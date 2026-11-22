"""Mesh convergence on the L-shaped domain, alpha = 1 (replaces Table 1).

For each h we run the rearrangement (structured mesh of size h) and then the
shape-gradient flow with remeshing (h, h_fine = h/2).  The values of the
original paper were 0.58058, 0.57848, 0.54091, 0.50227 for
h = 0.8, 0.5, 0.25, 0.1.
"""
import json
import time
import numpy as np
from common import FIG, RES
import steklov_holes as sh
from steklov_holes import algorithms as al
from steklov_holes import plotting as pl
from steklov_holes.mesh import d_difference, d_rectangle

alpha = 1.0
fd = lambda p: d_difference(d_rectangle(p, -1, 1, -1, 1), d_rectangle(p, 0, 2, 0, 2))
corners = [[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]]
rows, states, titles = [], [], []
for h in [0.2, 0.1, 0.05, 0.025]:
    t0 = time.time()
    m = sh.lshape(h)
    S = sh.P1System(m)
    re = al.multistart(S, alpha, centers=[(-0.5, -0.5), (-0.6, -0.6), (0.5, -0.5), (-0.5, 0.5), (0, 0)])
    t_re = time.time() - t0
    fl = al.shape_gradient_flow(m, re.hole, alpha, n_iter=150, smooth=25, patience=15,
                                remesh_args=dict(fd=fd, bbox=((-1, -1), (1, 1)), h=h,
                                                 h_fine=h / 2, corners=corners))
    rows.append(dict(h=h, nodes=int(m.n_nodes), area_rearr=re.area, lam_rearr=re.lam,
                     nodes_flow=int(fl.mesh.n_nodes), area_flow=fl.area, lam_flow=fl.lam,
                     time_rearr=t_re, time_total=time.time() - t0))
    print(rows[-1])
    states.append((fl.mesh.p, fl.mesh.t, fl.hole, fl.u))
    titles.append(rf"$h={h}$: $\lambda={fl.lam:.4f}$")
json.dump(rows, open(f"{RES}/convergence_lshape.json", "w"), indent=1)
pl.plot_holes_panel(states, titles, f"{FIG}/convergence_lshape.pdf", with_eigenfunction=False)
pl.plot_holes_panel(states, titles, f"{FIG}/convergence_lshape.png", with_eigenfunction=False)
