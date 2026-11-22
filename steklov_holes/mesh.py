"""Unstructured triangular meshes for the numerical experiments.

The generator is a compact implementation of the DistMesh algorithm of
P.-O. Persson and G. Strang ("A simple mesh generator in MATLAB", SIAM Review
46 (2004)), driven by signed distance functions.  Only numpy and scipy are
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Signed distance functions (negative inside)
# ---------------------------------------------------------------------------

def d_circle(p, center=(0.0, 0.0), radius=1.0):
    c = np.asarray(center, dtype=float)
    return np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) - radius


def d_rectangle(p, x0, x1, y0, y1):
    """Exact signed distance to an axis-aligned rectangle."""
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hx, hy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    qx = np.abs(p[:, 0] - cx) - hx
    qy = np.abs(p[:, 1] - cy) - hy
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def d_union(*ds):
    return np.min(np.vstack(ds), axis=0)


def d_difference(d1, d2):
    return np.maximum(d1, -d2)


def d_intersection(*ds):
    return np.max(np.vstack(ds), axis=0)


# ---------------------------------------------------------------------------
# Mesh container
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    p: np.ndarray            # (n, 2) node coordinates
    t: np.ndarray            # (m, 3) triangles, counter-clockwise
    bedges: np.ndarray = field(default=None)   # (k, 2) boundary edges
    axisymmetric: bool = False   # meridian section (r = x >= 0, z = y) of a body of revolution

    def __post_init__(self):
        self.p = np.ascontiguousarray(self.p, dtype=float)
        self.t = np.ascontiguousarray(self.t, dtype=np.int64)
        self._orient()
        if self.bedges is None:
            self.bedges = boundary_edges(self.t)
        self._build_topology()

    # -- geometry ----------------------------------------------------------
    def _orient(self):
        x = self.p[self.t]
        det = ((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1])
               - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0]))
        flip = det < 0
        self.t[flip] = self.t[flip][:, [0, 2, 1]]

    @property
    def n_nodes(self):
        return self.p.shape[0]

    @property
    def n_elements(self):
        return self.t.shape[0]

    @property
    def areas(self):
        x = self.p[self.t]
        return 0.5 * ((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1])
                      - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0]))

    @property
    def centroids(self):
        return self.p[self.t].mean(axis=1)

    @property
    def h(self):
        """Maximal edge length."""
        e = self.edges
        return float(np.max(np.linalg.norm(self.p[e[:, 0]] - self.p[e[:, 1]], axis=1)))

    @property
    def boundary_nodes(self):
        return np.unique(self.bedges)

    # -- topology ----------------------------------------------------------
    def _build_topology(self):
        t = self.t
        # local edge j of triangle k is opposite to vertex j
        loc = np.stack([t[:, [1, 2]], t[:, [2, 0]], t[:, [0, 1]]], axis=1)  # (m,3,2)
        flat = np.sort(loc.reshape(-1, 2), axis=1)
        edges, inv = np.unique(flat, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.edges = edges
        self.t2e = inv.reshape(-1, 3)
        # edge -> up to two elements
        e2t = -np.ones((edges.shape[0], 2), dtype=np.int64)
        elem = np.repeat(np.arange(t.shape[0]), 3)
        order = np.argsort(inv, kind="stable")
        inv_s, elem_s = inv[order], elem[order]
        first = np.ones_like(inv_s, dtype=bool)
        first[1:] = inv_s[1:] != inv_s[:-1]
        e2t[inv_s[first], 0] = elem_s[first]
        e2t[inv_s[~first], 1] = elem_s[~first]
        self.e2t = e2t

    def neighbours(self):
        """(m,3) array: neighbour across local edge j, or -1."""
        nb = -np.ones_like(self.t)
        for j in range(3):
            e = self.t2e[:, j]
            a, b = self.e2t[e, 0], self.e2t[e, 1]
            me = np.arange(self.n_elements)
            nb[:, j] = np.where(a == me, b, a)
        return nb

    # -- utilities ----------------------------------------------------------
    def refine(self):
        """Uniform red refinement (each triangle split into four)."""
        p, t = self.p, self.t
        e = self.edges
        mids = 0.5 * (p[e[:, 0]] + p[e[:, 1]])
        n = p.shape[0]
        newp = np.vstack([p, mids])
        m01 = n + self.t2e[:, 2]
        m12 = n + self.t2e[:, 0]
        m20 = n + self.t2e[:, 1]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        newt = np.vstack([
            np.column_stack([a, m01, m20]),
            np.column_stack([m01, b, m12]),
            np.column_stack([m20, m12, c]),
            np.column_stack([m01, m12, m20]),
        ])
        return Mesh(newp, newt, axisymmetric=self.axisymmetric)

    def deform(self, displacement):
        """Return a mesh with the same connectivity and moved nodes."""
        return Mesh(self.p + displacement, self.t.copy(), self.bedges.copy(), self.axisymmetric)


def boundary_edges(t):
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1]


# ---------------------------------------------------------------------------
# DistMesh
# ---------------------------------------------------------------------------

def distmesh2d(fd, h0, bbox, pfix=None, fh=None, max_iter=600, seed=0):
    """Generate a mesh of {fd < 0} with target edge length ``h0 * fh``."""
    rng = np.random.default_rng(seed)
    dptol, ttol, Fscale, deltat = 1e-3, 0.1, 1.2, 0.2
    geps = 1e-3 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0
    if fh is None:
        def fh(p):
            return np.ones(p.shape[0])
    pfix = np.zeros((0, 2)) if pfix is None else np.asarray(pfix, dtype=float)
    nfix = pfix.shape[0]

    (x0, y0), (x1, y1) = bbox
    x, y = np.meshgrid(np.arange(x0, x1 + h0, h0),
                       np.arange(y0, y1 + h0 * np.sqrt(3) / 2, h0 * np.sqrt(3) / 2))
    x[1::2, :] += h0 / 2
    p = np.column_stack([x.ravel(), y.ravel()])
    p = p[fd(p) < geps]
    r0 = 1.0 / fh(p) ** 2
    p = p[rng.random(p.shape[0]) < r0 / r0.max()]
    if nfix:
        keep = np.min(np.linalg.norm(p[:, None, :] - pfix[None, :, :], axis=2), axis=1) > 0.5 * h0
        p = np.vstack([pfix, p[keep]])

    pold = np.full_like(p, np.inf)
    t = None
    for _ in range(max_iter):
        if np.max(np.linalg.norm(p - pold, axis=1)) / h0 > ttol:
            pold = p.copy()
            t = Delaunay(p).simplices
            pmid = p[t].mean(axis=1)
            t = t[fd(pmid) < -geps]
            bars = np.unique(np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        barvec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.linalg.norm(barvec, axis=1)
        hbars = fh(0.5 * (p[bars[:, 0]] + p[bars[:, 1]]))
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0.0)
        Fvec = (F / L)[:, None] * barvec
        Ftot = np.zeros_like(p)
        np.add.at(Ftot, bars[:, 0], Fvec)
        np.add.at(Ftot, bars[:, 1], -Fvec)
        Ftot[:nfix] = 0.0
        p = p + deltat * Ftot
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            q = p[ix]
            dgx = (fd(q + [deps, 0]) - d[ix]) / deps
            dgy = (fd(q + [0, deps]) - d[ix]) / deps
            p[ix] -= np.column_stack([d[ix] * dgx, d[ix] * dgy])
        move = np.linalg.norm(deltat * Ftot[d < -geps], axis=1)
        if move.size and np.max(move) / h0 < dptol:
            break

    t = Delaunay(p).simplices
    t = t[fd(p[t].mean(axis=1)) < -geps]
    return _cleanup(p, t)


def _cleanup(p, t, min_quality=0.05):
    used = np.unique(t)
    remap = -np.ones(p.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    p, t = p[used], remap[t]
    x = p[t]
    a = np.linalg.norm(x[:, 1] - x[:, 2], axis=1)
    b = np.linalg.norm(x[:, 2] - x[:, 0], axis=1)
    c = np.linalg.norm(x[:, 0] - x[:, 1], axis=1)
    area = 0.5 * np.abs((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1])
                        - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0]))
    q = 4 * np.sqrt(3) * area / (a ** 2 + b ** 2 + c ** 2)
    t = t[q > min_quality]
    used = np.unique(t)
    remap = -np.ones(p.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return Mesh(p[used], remap[t])


# ---------------------------------------------------------------------------
# Domains used in the papers
# ---------------------------------------------------------------------------

def disk(h, radius=1.0, center=(0.0, 0.0), seed=0):
    cx, cy = center
    fd = lambda p: d_circle(p, center, radius)
    return distmesh2d(fd, h, ((cx - radius, cy - radius), (cx + radius, cy + radius)), seed=seed)


def annulus(h, R=1.0, r=0.5, hole_center=(0.0, 0.0), seed=0, graded=True):
    """Disk of radius R with a circular hole of radius r (a conforming mesh)."""
    fd = lambda p: d_difference(d_circle(p, (0, 0), R), d_circle(p, hole_center, r))
    fh = None
    if graded:
        # slightly finer near the hole, where the eigenfunction has a kink
        fh = lambda p: 0.6 + 0.4 * np.clip(np.abs(d_circle(p, hole_center, r)) / max(R - r, 1e-12), 0, 1)
    m = distmesh2d(fd, h, ((-R, -R), (R, R)), fh=fh, seed=seed)
    return m


def lshape(h, side=1.0):
    """L-shaped domain made of three unit squares (as in the paper).

    Built as a structured, exactly conforming triangulation.
    """
    n = max(1, int(round(side / h)))
    xs = np.linspace(-side, side, 2 * n + 1)
    X, Y = np.meshgrid(xs, xs)
    p = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange(p.shape[0]).reshape(2 * n + 1, 2 * n + 1)
    tris = []
    for i in range(2 * n):
        for j in range(2 * n):
            xc = 0.5 * (xs[j] + xs[j + 1])
            yc = 0.5 * (xs[i] + xs[i + 1])
            if xc > 0 and yc > 0:      # remove the upper-right square
                continue
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]
            # alternate the diagonal to avoid a directional bias
            if (i + j) % 2 == 0:
                tris += [[a, b, c], [a, c, d]]
            else:
                tris += [[a, b, d], [b, c, d]]
    t = np.array(tris)
    used = np.unique(t)
    remap = -np.ones(p.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    return Mesh(p[used], remap[t])


def two_disks_with_handle(h, radius=1.0, gap=1.0, handle_width=0.5, seed=0):
    """Two disks of the given radius joined by a rectangular handle."""
    cx = radius + 0.5 * gap
    fd = lambda p: d_union(d_circle(p, (-cx, 0), radius),
                           d_circle(p, (cx, 0), radius),
                           d_rectangle(p, -cx, cx, -0.5 * handle_width, 0.5 * handle_width))
    w = 0.5 * handle_width
    xr = cx - np.sqrt(radius ** 2 - w ** 2)
    pfix = [[-xr, w], [-xr, -w], [xr, w], [xr, -w]]
    return distmesh2d(fd, h, ((-cx - radius, -radius), (cx + radius, radius)), pfix=pfix, seed=seed)


def half_disk(h, radius=1.0, seed=0):
    """Meridian half-disk {r >= 0, r^2 + z^2 < R^2} of the ball B_R in R^3
    (axisymmetric mesh: coordinates (r, z))."""
    fd = lambda p: d_intersection(d_circle(p, (0, 0), radius), -p[:, 0])
    pfix = [[0.0, -radius], [0.0, radius]]
    m = distmesh2d(fd, h, ((0, -radius), (radius, radius)), pfix=pfix, seed=seed)
    m.axisymmetric = True
    return m
