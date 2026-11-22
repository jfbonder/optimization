"""Algorithms for the discrete optimal hole problem.

    minimise lambda_{1,h}(A_h) over unions of triangles A_h with |A_h| ~ alpha.

Three methods are implemented:

* :func:`swap_algorithm` -- the element-swapping algorithm of the original
  paper (remove hole triangles where |du/dnu| is large, add triangles where it
  is small), with optional tabu list and decreasing number of swaps.

* :func:`rearrangement` -- penalised (fictitious-domain) relaxation
      lambda_beta(chi) = min (|grad u|^2 + u^2 + beta chi u^2)/||u||^2_{dOmega}
  combined with the rearrangement step chi_{k+1} = sublevel set of u_k^2 of
  measure alpha.  For fixed beta the sequence lambda_beta(chi_k) is
  non-increasing (Rayleigh quotient argument), and lambda_beta increases to the
  hard-constrained lambda_{1,h} as beta -> infinity.

* :func:`shape_gradient_flow` -- boundary-fitted descent with the shape
  derivative lambda'(A)[V] = -int_{dA} (du/dnu)^2 <V,nu>: the nodes of the
  hole interface are moved with normal velocity (du/dnu)^2 - mean, the motion
  is extended harmonically to the mesh, and the mesh is periodically
  regenerated with refinement near the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import splu

from .eigen import eig_hard, eig_penalized
from .fem import P1System
from .mesh import Mesh, distmesh2d


@dataclass
class Result:
    mesh: Mesh
    hole: np.ndarray
    lam: float
    u: np.ndarray
    history: list = field(default_factory=list)   # lambda_{1,h} per iteration
    areas: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def area(self):
        return float(self.mesh.areas[self.hole].sum())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frame(mesh, hole, u, lam, label=""):
    """Snapshot used to produce movies of the iterations."""
    return dict(p=mesh.p.copy(), t=mesh.t.copy(), hole=np.asarray(hole).copy(),
                u=np.asarray(u).copy(), lam=float(lam), area=float(mesh.areas[hole].sum()),
                label=label)

def select_by_score(score, areas, alpha, allowed=None):
    """Greedy choice of elements with the smallest score until area >= alpha.

    The result belongs to the class O_h of the paper: |A_h| >= alpha and the
    removal of the last chosen triangle gives |A_h| < alpha.
    """
    score = np.asarray(score, dtype=float).copy()
    if allowed is not None:
        score[~allowed] = np.inf
    order = np.argsort(score, kind="stable")
    cum = np.cumsum(areas[order])
    k = int(np.searchsorted(cum, alpha)) + 1
    k = min(k, order.size)
    hole = np.zeros(score.size, dtype=bool)
    hole[order[:k]] = True
    return hole


def initial_hole(mesh: Mesh, alpha, center=(0.0, 0.0), shape="disk", rng=None, allowed=None):
    """Initial union of triangles of measure ~alpha."""
    c = mesh.centroids
    if shape == "disk":
        score = np.hypot(c[:, 0] - center[0], c[:, 1] - center[1])
    elif shape == "halfplane":
        score = c[:, 0] - center[0]
    elif shape == "random":
        rng = np.random.default_rng(rng)
        score = rng.random(mesh.n_elements)
    else:
        raise ValueError(shape)
    return select_by_score(score, mesh.areas, alpha, allowed)


def interior_elements(mesh: Mesh):
    """Elements without vertices on dOmega (holes must stay inside Omega)."""
    onb = np.zeros(mesh.n_nodes, dtype=bool)
    onb[mesh.boundary_nodes] = True
    return ~np.any(onb[mesh.t], axis=1)


def interface_edges(mesh: Mesh, hole):
    """Interior edges separating hole and non-hole elements.

    Returns (edge indices, element inside the hole, element outside).
    """
    e2t = mesh.e2t
    both = (e2t[:, 0] >= 0) & (e2t[:, 1] >= 0)
    a = np.where(both, e2t[:, 0], 0)
    b = np.where(both, e2t[:, 1], 0)
    mask = both & (hole[a] != hole[b])
    idx = np.flatnonzero(mask)
    ins = np.where(hole[a[idx]], a[idx], b[idx])
    out = np.where(hole[a[idx]], b[idx], a[idx])
    return idx, ins, out


# ---------------------------------------------------------------------------
# 1. the algorithm of the paper
# ---------------------------------------------------------------------------

def swap_algorithm(sys: P1System, hole0, alpha, n_iter=200, n_swap=1, tabu=10,
                   decrease_every=None, verbose=False, record=False, keep_off_boundary=False):
    """Element-swapping algorithm of Section 3.2 of the paper.

    At each step the hole triangles adjacent to the interface faces with the
    largest |du_h/dnu| are removed (n_swap of them) and outside triangles
    adjacent to faces with the smallest |du_h/dnu| are added until |A_h| >=
    alpha.  A triangle removed during the last ``tabu`` steps cannot be added
    back (the anti-cycling strategy described in the paper).
    ``n_swap`` may be an int or a callable k -> int.
    """
    mesh = sys.mesh
    areas = mesh.areas
    allowed = interior_elements(mesh) if keep_off_boundary else np.ones(mesh.n_elements, bool)
    hole = hole0.copy()
    lam, u = eig_hard(sys, hole)
    best = (lam, hole.copy(), u)
    hist, ar = [lam], [areas[hole].sum()]
    removed_at = np.full(mesh.n_elements, -10 ** 9)
    frames = [_frame(mesh, hole, u, lam, "swap")] if record else []
    for k in range(n_iter):
        g2 = np.sum(sys.element_grad(u) ** 2, axis=1)
        idx, ins, out = interface_edges(mesh, hole)
        if idx.size == 0:
            break
        dn = g2[out]                        # (du/dnu)^2 on the face
        # inside triangles: score = largest adjacent normal derivative
        s_in = np.full(mesh.n_elements, -np.inf)
        np.maximum.at(s_in, ins, dn)
        # outside triangles: score = smallest adjacent normal derivative
        s_out = np.full(mesh.n_elements, np.inf)
        np.minimum.at(s_out, out, dn)
        m = n_swap(k) if callable(n_swap) else n_swap
        cand_in = np.flatnonzero(np.isfinite(s_in))
        rem = cand_in[np.argsort(-s_in[cand_in])][:m]
        newhole = hole.copy()
        newhole[rem] = False
        removed_at[rem] = k
        # add triangles with the smallest normal derivative
        okadd = np.isfinite(s_out) & allowed & (k - removed_at > tabu) & ~newhole
        order = np.flatnonzero(okadd)
        order = order[np.argsort(s_out[order])]
        area = areas[newhole].sum()
        for e in order:
            if area >= alpha:
                break
            newhole[e] = True
            area += areas[e]
        # If the old interface does not offer enough candidates (short
        # interfaces, e.g. a hole filling a whole disk), recompute the
        # interface of the current hole and keep adding the adjacent outside
        # triangle with the smallest |grad u_h|^2 (the discrete normal
        # derivative) until |A_h| >= alpha.
        while area < alpha:
            _, _, out2 = interface_edges(mesh, newhole)
            cand = np.unique(out2)
            cand = cand[allowed[cand]]
            ok = cand[k - removed_at[cand] > tabu]
            if ok.size == 0:
                ok = cand
            if ok.size == 0:
                break
            e = ok[np.argmin(g2[ok])]
            newhole[e] = True
            area += areas[e]
        if area < alpha:
            break
        hole = newhole
        lam, u = eig_hard(sys, hole, v0=u)
        hist.append(lam)
        ar.append(area)
        if lam < best[0]:
            best = (lam, hole.copy(), u)
        if record:
            frames.append(_frame(mesh, hole, u, lam, "swap"))
        if verbose and k % 10 == 0:
            print(f"[swap] it {k:4d} lambda={lam:.6f} best={best[0]:.6f}")
    return Result(mesh, best[1], best[0], best[2], hist, ar, {"method": "swap", "frames": frames})


# ---------------------------------------------------------------------------
# 2. penalised rearrangement
# ---------------------------------------------------------------------------

def rearrangement(sys: P1System, hole0, alpha, betas=(1e1, 1e2, 1e3, 1e4, 1e5),
                  max_inner=100, verbose=False, record=False, keep_off_boundary=False):
    """Monotone rearrangement algorithm for the penalised problem.

    For each beta (continuation), iterate
        u_k   = first eigenfunction of  A + beta M_chi_k  (Steklov),
        chi_{k+1} = argmin { int chi u_k^2 : chi in O_h },
    accepting chi_{k+1} only if int chi_{k+1} u_k^2 <= int chi_k u_k^2, which
    guarantees lambda_beta(chi_{k+1}) <= lambda_beta(chi_k).
    The history stores the hard eigenvalue lambda_{1,h}(chi_k) as well.
    """
    mesh = sys.mesh
    areas = mesh.areas
    allowed = interior_elements(mesh) if keep_off_boundary else None
    chi = hole0.copy()
    hist_pen, hist_hard, ar = [], [], []
    frames = []
    if record:
        lh, uh = eig_hard(sys, chi)
        frames.append(_frame(mesh, chi, uh, lh, "initial hole"))
    u = None
    for beta in betas:
        lam_b, u = eig_penalized(sys, chi, beta, v0=u)
        for k in range(max_inner):
            hist_pen.append((beta, lam_b))
            dens = sys.element_l2sq(u)
            new = select_by_score(dens / areas, areas, alpha, allowed)
            if np.array_equal(new, chi):
                break
            if dens[new].sum() > dens[chi].sum() * (1 + 1e-12):
                break
            lam_new, u_new = eig_penalized(sys, new, beta, v0=u)
            if lam_new > lam_b * (1 + 1e-12):    # cannot happen in exact arithmetic
                break
            chi, lam_b, u = new, lam_new, u_new
            ar.append(areas[chi].sum())
            if record:
                lh, uh = eig_hard(sys, chi)
                frames.append(_frame(mesh, chi, uh, lh, f"rearrangement (beta={beta:.0e})"))
        lam_h, _ = eig_hard(sys, chi)
        hist_hard.append((beta, lam_h))
        if verbose:
            print(f"[rearr] beta={beta:8.1e} lambda_beta={lam_b:.6f} lambda_h={lam_h:.6f}")
    lam, uh = eig_hard(sys, chi)
    return Result(mesh, chi, lam, uh, [h for _, h in hist_hard], ar,
                  {"method": "rearrangement", "penalized": hist_pen, "hard_per_beta": hist_hard,
                   "frames": frames})


def local_search(sys: P1System, hole, alpha, max_iter=200, n_cand=6, verbose=False, record=False,
                 keep_off_boundary=False):
    """Monotone polishing of a discrete hole with exact evaluations.

    Candidate moves swap one interface triangle with large |du/dnu| for an
    outside triangle with small |du/dnu|; the best improving swap is accepted.
    """
    mesh = sys.mesh
    areas = mesh.areas
    allowed = interior_elements(mesh) if keep_off_boundary else np.ones(mesh.n_elements, bool)
    lam, u = eig_hard(sys, hole)
    hist = [lam]
    frames = [_frame(mesh, hole, u, lam, "local search")] if record else []
    for it in range(max_iter):
        g2 = np.sum(sys.element_grad(u) ** 2, axis=1)
        idx, ins, out = interface_edges(mesh, hole)
        s_in = np.full(mesh.n_elements, -np.inf)
        np.maximum.at(s_in, ins, g2[out])
        s_out = np.full(mesh.n_elements, np.inf)
        np.minimum.at(s_out, out, g2[out])
        ci = np.flatnonzero(np.isfinite(s_in))
        ci = ci[np.argsort(-s_in[ci])][:n_cand]
        co = np.flatnonzero(np.isfinite(s_out) & allowed)
        co = co[np.argsort(s_out[co])][:n_cand]
        best = None
        for a in ci:
            for b in co:
                new = hole.copy()
                new[a] = False
                new[b] = True
                if new[allowed].sum() == 0:
                    continue
                # keep the class O_h: area >= alpha, minimal
                if areas[new].sum() < alpha:
                    continue
                lam_new, u_new = eig_hard(sys, new, v0=u)
                if lam_new < lam - 1e-12 and (best is None or lam_new < best[0]):
                    best = (lam_new, new, u_new)
        if best is None:
            break
        lam, hole, u = best
        hist.append(lam)
        if record:
            frames.append(_frame(mesh, hole, u, lam, "local search"))
        if verbose:
            print(f"[local] it {it} lambda={lam:.6f}")
    return Result(mesh, hole, lam, u, hist, [areas[hole].sum()], {"method": "local_search", "frames": frames})


# ---------------------------------------------------------------------------
# 3. boundary-fitted shape-gradient flow (moving mesh + adaptive remeshing)
# ---------------------------------------------------------------------------

def _quality(mesh):
    x = mesh.p[mesh.t]
    a = np.linalg.norm(x[:, 1] - x[:, 2], axis=1)
    b = np.linalg.norm(x[:, 2] - x[:, 0], axis=1)
    c = np.linalg.norm(x[:, 0] - x[:, 1], axis=1)
    return 4 * np.sqrt(3) * mesh.areas / (a ** 2 + b ** 2 + c ** 2)


def interface_data(sys: P1System, hole, u):
    """Nodes of the interface, normals pointing into the hole, (du/dnu)^2.

    Returns a dict with node indices, unit normals, nodal values of
    (du/dnu)^2 (length-weighted averages over adjacent faces), nodal lumped
    lengths, and the face data.
    """
    mesh = sys.mesh
    idx, ins, out = interface_edges(mesh, hole)
    e = mesh.edges[idx]
    pa, pb = mesh.p[e[:, 0]], mesh.p[e[:, 1]]
    tvec = pb - pa
    L = np.linalg.norm(tvec, axis=1)
    n = np.column_stack([-tvec[:, 1], tvec[:, 0]]) / L[:, None]
    cin = mesh.centroids[ins]
    flip = np.sum((cin - 0.5 * (pa + pb)) * n, axis=1) < 0
    n[flip] *= -1
    g2 = np.sum(sys.element_grad(u) ** 2, axis=1)[out]
    nodes = np.unique(e)
    pos = -np.ones(mesh.n_nodes, dtype=np.int64)
    pos[nodes] = np.arange(nodes.size)
    k = nodes.size
    w = np.zeros(k)
    gn = np.zeros(k)
    nn = np.zeros((k, 2))
    for j in (0, 1):
        np.add.at(w, pos[e[:, j]], 0.5 * L)
        np.add.at(gn, pos[e[:, j]], 0.5 * L * g2)
        np.add.at(nn, pos[e[:, j]], (0.5 * L)[:, None] * n)
    gn /= w
    nn /= np.linalg.norm(nn, axis=1)[:, None]
    return dict(nodes=nodes, normal=nn, g2=gn, weight=w, edges=e, pos=pos,
                face_len=L, face_g2=g2)


def _smooth_on_interface(values, edges, pos, passes=3):
    """Jacobi smoothing along the interface graph (a discrete H^1 filter)."""
    k = values.shape[0]
    v = values.copy()
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    deg = np.zeros(k)
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    for _ in range(passes):
        s = np.zeros_like(v)
        np.add.at(s, a, v[b])
        np.add.at(s, b, v[a])
        v = 0.5 * v + 0.5 * s / np.maximum(deg, 1)[:, None] if v.ndim == 2 else 0.5 * v + 0.5 * s / np.maximum(deg, 1)
    return v


def harmonic_extension(sys: P1System, nodes, disp):
    """Extend a displacement given on ``nodes`` (zero on dOmega) to the mesh."""
    mesh = sys.mesh
    n = mesh.n_nodes
    fixed = np.zeros(n, dtype=bool)
    fixed[nodes] = True
    fixed[mesh.boundary_nodes] = True
    D = np.zeros((n, 2))
    D[nodes] = disp
    free = np.flatnonzero(~fixed)
    if free.size:
        K = sys.K
        lu = splu(K[free][:, free].tocsc())
        rhs = -(K[free][:, np.flatnonzero(fixed)] @ D[fixed])
        D[free] = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
    return D


def _thin(points, dmin):
    """Greedy thinning: keep points at mutual distance >= dmin."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    keep = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(points[i], dmin):
            if j > i:
                keep[j] = False
    return points[keep]


def remesh(mesh: Mesh, hole, fd, bbox, h, h_fine, width=None, seed=0, smooth_passes=10,
           corners=None):
    """New mesh of Omega refined near the interface of ``hole``.

    The element size grows linearly from h_fine at the interface to h at
    distance ``width``.  The interface polygon is resampled with spacing
    ~h_fine; these points are kept fixed so that the new mesh resolves the
    interface, and the elements of the new mesh are classified by locating
    their centroids in the old mesh.
    """
    from scipy.spatial import cKDTree

    idx, ins, out = interface_edges(mesh, hole)
    e = mesh.edges[idx]
    # Laplacian smoothing of the interface polygon (contact points stay on dOmega)
    nodes = np.unique(e)
    pos = -np.ones(mesh.n_nodes, dtype=np.int64)
    pos[nodes] = np.arange(nodes.size)
    X = mesh.p[nodes].copy()
    if smooth_passes:
        onb = np.isin(nodes, mesh.boundary_nodes)
        Xs = _smooth_on_interface(X, e, pos, smooth_passes)
        Xs[onb] = X[onb]
        X = Xs
    pa, pb = X[pos[e[:, 0]]], X[pos[e[:, 1]]]
    pts = []
    for a, b in zip(pa, pb):
        m = max(1, int(np.ceil(np.linalg.norm(b - a) / (0.5 * h_fine))))
        s = np.arange(m + 1) / m
        pts.append((1 - s)[:, None] * a + s[:, None] * b)
    pts = np.vstack(pts)
    tree = cKDTree(pts)
    width = 6 * h if width is None else width

    def fh(p):
        d, _ = tree.query(p)
        return np.minimum(h_fine + (h - h_fine) * d / width, h) / h_fine

    # contact points with dOmega first (so that they survive the thinning)
    onb = np.abs(fd(pts)) < 1e-9
    order = np.concatenate([np.flatnonzero(onb), np.flatnonzero(~onb)])
    pfix = _thin(pts[order], 0.9 * h_fine)
    pfix = pfix[fd(pfix) < 1e-9]
    if corners is not None:
        corners = np.asarray(corners, dtype=float)
        far = np.min(np.linalg.norm(pfix[:, None] - corners[None], axis=2), axis=1) > 0.5 * h_fine
        pfix = np.vstack([corners, pfix[far]])
    new = distmesh2d(fd, h_fine, bbox, pfix=pfix, fh=fh, seed=seed)
    newhole = hole[locate(mesh, new.centroids)]
    # near the interface, classify by the side of the (smoothed) interface
    seg_a, seg_b = pa, pb
    # normals pointing into the hole, oriented like those of the old mesh
    tv = seg_b - seg_a
    nrm = np.column_stack([-tv[:, 1], tv[:, 0]])
    cin = mesh.centroids[ins]
    flip = np.sum((cin - 0.5 * (mesh.p[e[:, 0]] + mesh.p[e[:, 1]])) * nrm, axis=1) < 0
    nrm[flip] *= -1
    c = new.centroids
    mids = 0.5 * (seg_a + seg_b)
    stree = cKDTree(mids)
    dist, near = stree.query(c, k=min(6, mids.shape[0]))
    near = np.atleast_2d(near)
    best_d = np.full(c.shape[0], np.inf)
    side = np.zeros(c.shape[0])
    for j in range(near.shape[1]):
        k = near[:, j]
        a, b = seg_a[k], seg_b[k]
        ab = b - a
        tpar = np.clip(np.sum((c - a) * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-300), 0, 1)
        proj = a + tpar[:, None] * ab
        d = np.linalg.norm(c - proj, axis=1)
        better = d < best_d
        best_d[better] = d[better]
        side[better] = np.sum((c[better] - proj[better]) * nrm[k[better]], axis=1)
    close = best_d < 3 * h_fine
    newhole[close] = side[close] > 0
    return new, newhole


def locate(mesh: Mesh, x, k=12):
    """Index of an element of ``mesh`` containing each point (nearest if none)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.centroids)
    _, cand = tree.query(x, k=min(k, mesh.n_elements))
    res = cand[:, 0].copy()
    found = np.zeros(x.shape[0], dtype=bool)
    P = mesh.p[mesh.t]
    for j in range(cand.shape[1]):
        c = cand[:, j]
        a, b, d = P[c, 0], P[c, 1], P[c, 2]
        v0, v1, v2 = b - a, d - a, x - a
        den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        l1 = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / den
        inside = (l1 >= -1e-12) & (l2 >= -1e-12) & (l1 + l2 <= 1 + 1e-12) & ~found
        res[inside] = c[inside]
        found |= inside
    return res


def _boundary_projection(fd, p, eps=1e-8):
    d = fd(p)
    gx = (fd(p + [eps, 0]) - d) / eps
    gy = (fd(p + [0, eps]) - d) / eps
    g = np.column_stack([gx, gy])
    g /= np.maximum(np.linalg.norm(g, axis=1), 1e-14)[:, None]
    return d, g


def shape_gradient_flow(mesh: Mesh, hole, alpha, n_iter=100, step=0.3, smooth=3,
                        min_quality=0.2, remesh_args=None, steklov_edges=None,
                        tol=1e-7, patience=8, verbose=False, record=False):
    """Boundary-fitted gradient descent driven by the shape derivative.

    Interface nodes x_i move along the normal nu_i (pointing into the hole) by
        tau * (g_i - mean(g)) + delta,   g = (du/dnu)^2,
    where the uniform shift delta = (|A| - alpha)/|dA| restores the volume.
    By Theorem 1, lambda'(A)[V] = -int (g - mean g) g = -int (g - mean g)^2 <= 0
    for the volume preserving part.  Interface nodes lying on dOmega (contact
    points) slide tangentially along dOmega.  The step tau is chosen so that
    the largest displacement is ``step`` times the local mesh size, with
    backtracking until lambda decreases.  The displacement is extended
    harmonically to the mesh.  When the mesh quality drops below
    ``min_quality`` (or no descent step is found) the domain is regenerated by
    :func:`remesh`, refined near the interface, if ``remesh_args`` =
    dict(fd=..., bbox=..., h=..., h_fine=...) is given.
    """
    fd = None if remesh_args is None else remesh_args["fd"]
    sys = P1System(mesh, steklov_edges)
    lam, u = eig_hard(sys, hole)
    hist, ar = [lam], [mesh.areas[hole].sum()]
    n_remesh = 0
    frames = [_frame(mesh, hole, u, lam, "shape gradient flow")] if record else []
    best = (lam, mesh, hole, u)
    stall = 0
    q_ref = np.min(_quality(mesh))
    for it in range(n_iter):
        data = interface_data(sys, hole, u)
        nodes, nrm, g = data["nodes"], data["normal"], data["g2"]
        onb = np.isin(nodes, mesh.boundary_nodes)
        g = _smooth_on_interface(g, data["edges"], data["pos"], smooth)
        w = data["weight"]
        gbar = np.sum(w * g) / np.sum(w)
        V = g - gbar
        perim = np.sum(w)
        area = mesh.areas[hole].sum()
        delta = (area - alpha) / perim
        hl = np.zeros(nodes.size)
        e = data["edges"]
        np.maximum.at(hl, data["pos"][e[:, 0]], data["face_len"])
        np.maximum.at(hl, data["pos"][e[:, 1]], data["face_len"])
        vmax = max(np.max(np.abs(V)), 1e-14)
        tau = step * np.min(hl) / vmax
        q_cur = np.min(_quality(mesh))
        tangent = None
        if fd is not None and np.any(onb):
            _, gb = _boundary_projection(fd, mesh.p[nodes[onb]])
            tangent = np.column_stack([-gb[:, 1], gb[:, 0]])
        accepted = False
        for _ in range(10):
            disp = (tau * V + delta)[:, None] * nrm
            if np.any(onb):
                if tangent is None:
                    disp[onb] = 0.0
                else:
                    disp[onb] = np.sum(disp[onb] * tangent, axis=1)[:, None] * tangent
            D = harmonic_extension(sys, nodes, disp)
            new_mesh = mesh.deform(D)
            if fd is not None:
                b = new_mesh.boundary_nodes
                d, gb = _boundary_projection(fd, new_mesh.p[b])
                new_mesh.p[b] -= d[:, None] * gb
            if np.min(new_mesh.areas) <= 0 or np.min(_quality(new_mesh)) < 0.5 * q_cur:
                tau *= 0.5
                continue
            new_sys = P1System(new_mesh, steklov_edges)
            lam_new, u_new = eig_hard(new_sys, hole, v0=u)
            # first-order effect of the uniform volume shift: -delta * perim * gbar
            if lam_new <= lam - delta * perim * gbar + 1e-13:
                accepted = True
                break
            tau *= 0.5
        if accepted:
            mesh, sys, lam, u = new_mesh, new_sys, lam_new, u_new
            hist.append(lam)
            ar.append(mesh.areas[hole].sum())
            if record:
                frames.append(_frame(mesh, hole, u, lam, "shape gradient flow"))
        q = np.min(_quality(mesh))
        if remesh_args is not None and ((q < min_quality and q < 0.6 * q_ref) or not accepted):
            mesh, hole = remesh(mesh, hole, seed=n_remesh, **remesh_args)
            n_remesh += 1
            q_ref = np.min(_quality(mesh))
            sys = P1System(mesh, steklov_edges)
            lam, u = eig_hard(sys, hole)
            hist.append(lam)
            ar.append(mesh.areas[hole].sum())
            if record:
                frames.append(_frame(mesh, hole, u, lam, "shape gradient flow (remeshed)"))
        elif not accepted:
            break
        if verbose and it % 5 == 0:
            print(f"[flow] it {it:3d} lambda={lam:.6f} area={ar[-1]:.5f} q={q:.2f} "
                  f"remeshes={n_remesh}")
        if lam < best[0] * (1 - tol) and abs(ar[-1] - alpha) < 0.02 * alpha:
            best = (lam, mesh, hole, u)
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    lam, mesh, hole, u = best
    return Result(mesh, hole, lam, u, hist, ar,
                  {"method": "shape_gradient_flow", "remeshes": n_remesh, "frames": frames})


def multistart(sys: P1System, alpha, centers, betas_list=((1e1, 1e2, 1e3, 1e4, 1e5),
                                                         (1, 3, 1e1, 3e1, 1e2, 1e3, 1e4, 1e5)),
               record=False, shape="disk"):
    """Run :func:`rearrangement` from several initial holes; return the best.

    The penalised problem is not convex, so the rearrangement may stop at a
    local minimiser; a handful of initial holes (and two beta schedules) is
    enough in all our experiments to find the best configuration.
    """
    best = None
    tried = []
    for c in centers:
        for betas in betas_list:
            h0 = initial_hole(sys.mesh, alpha, center=c, shape=shape)
            res = rearrangement(sys, h0, alpha, betas=betas, record=record)
            res.info["start"] = dict(center=tuple(map(float, c)), betas=list(betas))
            tried.append((float(res.lam), tuple(map(float, c)), len(betas)))
            if best is None or res.lam < best.lam:
                best = res
    best.info["multistart"] = tried
    return best
