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


@dataclass
class P1System:
    """Matrices of the P1 discretisation on a fixed mesh."""

    mesh: Mesh
    steklov_edges: np.ndarray | None = None   # boundary edges carrying the Steklov condition

    def __post_init__(self):
        m = self.mesh
        self.area = m.areas
        self.grad = gradients(m)
        Kloc = np.einsum("kid,kjd->kij", self.grad, self.grad) * self.area[:, None, None]
        self.K = _assemble(m, Kloc)                       # stiffness
        self.M = _assemble(m, _MLOC[None] * self.area[:, None, None])   # mass
        self.A = (self.K + self.M).tocsr()               # H^1 form
        if self.steklov_edges is None:
            self.steklov_edges = m.bedges
        self.B = boundary_mass(m, self.steklov_edges)
        self.steklov_nodes = np.unique(self.steklov_edges)

    # -- element-wise quantities ---------------------------------------------
    def weighted_mass(self, w):
        """Mass matrix with element-wise constant weight w (length m)."""
        return _assemble(self.mesh, _MLOC[None] * (w * self.area)[:, None, None])

    def element_l2sq(self, u):
        """int_T u^2 for each element (exact for P1)."""
        ut = u[self.mesh.t]
        return self.area / 12.0 * (np.sum(ut ** 2, axis=1) + np.sum(ut, axis=1) ** 2)

    def element_grad(self, u):
        """Constant gradient of the P1 function u on each element, (m, 2)."""
        return np.einsum("kid,ki->kd", self.grad, u[self.mesh.t])

    def h1_norm_sq(self, u):
        return float(u @ (self.A @ u))

    def trace_norm_sq(self, u):
        return float(u @ (self.B @ u))


def boundary_mass(mesh: Mesh, edges):
    p = mesh.p
    L = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    loc = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    vals = (L[:, None, None] * loc[None]).ravel()
    rows = np.repeat(edges, 2, axis=1).ravel()
    cols = np.tile(edges, (1, 2)).ravel()
    n = mesh.n_nodes
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
