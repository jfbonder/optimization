"""P1 finite elements for the Steklov problem with a hole.

Continuous problem (Omega subset R^2, hole A subset Omega):

    -Delta u + u = 0   in Omega \\ A,
    du/dnu = lambda u  on dOmega,
    u = 0              in A.

The first eigenvalue lambda_1(A) is the best constant in the trace inequality
S ||u||^2_{L^2(dOmega)} <= ||u||^2_{H^1(Omega)} for u vanishing on A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh

_MLOC = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0


def gradients(mesh: Mesh):
    """Gradients of the barycentric basis functions, shape (m, 3, 2)."""
    x = mesh.p[mesh.t]
    det = ((x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1])
           - (x[:, 1, 1] - x[:, 0, 1]) * (x[:, 2, 0] - x[:, 0, 0]))
    G = np.empty((mesh.n_elements, 3, 2))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        e = x[:, k] - x[:, j]          # edge opposite to vertex i
        G[:, i, 0] = -e[:, 1] / det
        G[:, i, 1] = e[:, 0] / det
    return G


def _assemble(mesh: Mesh, local):
    t = mesh.t
    rows = np.repeat(t, 3, axis=1).ravel()
    cols = np.tile(t, (1, 3)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))


# 6-point quadrature on triangles, exact for polynomials of degree 4
_QA, _QB = 0.445948490915965, 0.091576213509771
_QW1, _QW2 = 0.223381589678011, 0.109951743655322
_QBARY = np.array([[_QA, _QA, 1 - 2 * _QA], [_QA, 1 - 2 * _QA, _QA], [1 - 2 * _QA, _QA, _QA],
                   [_QB, _QB, 1 - 2 * _QB], [_QB, 1 - 2 * _QB, _QB], [1 - 2 * _QB, _QB, _QB]])
_QW = np.array([_QW1] * 3 + [_QW2] * 3)


@dataclass
class P1System:
    """Matrices of the P1 discretisation on a fixed mesh.

    If ``mesh.axisymmetric`` is True, the mesh is the meridian section
    (r, z), r >= 0, of a body of revolution in R^3 and all the integrals carry
    the weight 2*pi*r (axisymmetric functions); the Steklov condition acts on
    the boundary edges that do not lie on the axis r = 0.
    """

    mesh: Mesh
    steklov_edges: np.ndarray | None = None   # boundary edges carrying the Steklov condition

    def __post_init__(self):
        m = self.mesh
        self.axisymmetric = bool(getattr(m, "axisymmetric", False))
        self.area = m.areas
        self.grad = gradients(m)
        GG = np.einsum("kid,kjd->kij", self.grad, self.grad)
        if self.axisymmetric:
            rq = np.einsum("qi,kid->kqd", _QBARY, m.p[m.t])[:, :, 0]          # r at quadrature points
            self._wq = 2 * np.pi * self.area[:, None] * _QW[None, :] * rq      # weights (m, 6)
            rbar = m.p[m.t][:, :, 0].mean(axis=1)
            Kloc = GG * (2 * np.pi * self.area * rbar)[:, None, None]
            self._Mloc = np.einsum("kq,qi,qj->kij", self._wq, _QBARY, _QBARY)
            self.measure = 2 * np.pi * self.area * rbar                       # volume of revolution
        else:
            Kloc = GG * self.area[:, None, None]
            self._Mloc = _MLOC[None] * self.area[:, None, None]
            self.measure = self.area
        self.K = _assemble(m, Kloc)                       # stiffness
        self.M = _assemble(m, self._Mloc)                 # mass
        self.A = (self.K + self.M).tocsr()               # H^1 form
        if self.steklov_edges is None:
            e = m.bedges
            if self.axisymmetric:
                scale = np.ptp(m.p, axis=0).max()
                on_axis = np.all(np.abs(m.p[e][:, :, 0]) < 1e-9 * scale, axis=1)
                e = e[~on_axis]
            self.steklov_edges = e
        self.B = boundary_mass(m, self.steklov_edges, self.axisymmetric)
        self.steklov_nodes = np.unique(self.steklov_edges)

    # -- element-wise quantities ---------------------------------------------
    def weighted_mass(self, w):
        """Mass matrix with element-wise constant weight w (length m)."""
        return _assemble(self.mesh, self._Mloc * np.asarray(w, dtype=float)[:, None, None])

    def element_l2sq(self, u):
        """int_T u^2 for each element (exact for P1; weight 2 pi r if axisymmetric)."""
        ut = u[self.mesh.t]
        if self.axisymmetric:
            uq = ut @ _QBARY.T            # (m, 6)
            return np.sum(self._wq * uq ** 2, axis=1)
        return self.area / 12.0 * (np.sum(ut ** 2, axis=1) + np.sum(ut, axis=1) ** 2)

    def element_grad(self, u):
        """Constant gradient of the P1 function u on each element, (m, 2)."""
        return np.einsum("kid,ki->kd", self.grad, u[self.mesh.t])

    def h1_norm_sq(self, u):
        return float(u @ (self.A @ u))

    def trace_norm_sq(self, u):
        return float(u @ (self.B @ u))


def boundary_mass(mesh: Mesh, edges, axisymmetric=False):
    p = mesh.p
    L = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    if axisymmetric:
        # 2-point Gauss rule (exact for the cubic integrands 2 pi r phi_i phi_j)
        g = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
        phi = np.array([[1 - g[0], g[0]], [1 - g[1], g[1]]])          # (q, i)
        r = np.einsum("qi,ki->kq", phi, p[edges][:, :, 0])
        w = 2 * np.pi * 0.5 * L[:, None] * r                              # (k, q)
        loc = np.einsum("kq,qi,qj->kij", w, phi, phi)
        vals = loc.ravel()
    else:
        loc = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
        vals = (L[:, None, None] * loc[None]).ravel()
    rows = np.repeat(edges, 2, axis=1).ravel()
    cols = np.tile(edges, (1, 2)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
