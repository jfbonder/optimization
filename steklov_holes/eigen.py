"""First Steklov eigenpair for the discrete problems."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .fem import P1System


def _smallest_steklov(A, B, v0=None, tol=1e-12):
    """Smallest lambda with A u = lambda B u, A s.p.d., B s.p.semidefinite.

    We compute the largest eigenvalue mu = 1/lambda of B u = mu A u (ARPACK in
    generalized mode with M = A).  Nodes not carrying boundary mass correspond
    to mu = 0 and do not interfere.
    """
    A = A.tocsc()
    lu = splu(A)
    Minv = LinearOperator(A.shape, matvec=lu.solve, dtype=float)
    n = A.shape[0]
    if v0 is not None and (v0.shape[0] != n or not np.any(v0)):
        v0 = None
    ncv = min(n, 20)
    mu, vec = eigsh(B.tocsr(), k=1, M=A, Minv=Minv, which="LA", v0=v0, tol=tol, ncv=ncv)
    lam = 1.0 / mu[0]
    u = vec[:, 0]
    nrm = np.sqrt(u @ (B @ u))
    u = u / nrm
    if u.sum() < 0:
        u = -u
    return lam, u


def eig_hard(sys: P1System, hole, v0=None):
    """lambda_{1,h}(A_h): P1 functions vanishing on the union of hole triangles.

    ``hole`` is a boolean mask over elements.  Returns (lambda, u) with u a
    nodal vector (zero on the hole) normalised by ||u||_{L^2(dOmega)} = 1.
    """
    hole = np.asarray(hole, dtype=bool)
    fixed = np.zeros(sys.mesh.n_nodes, dtype=bool)
    fixed[np.unique(sys.mesh.t[hole])] = True
    return eig_dirichlet(sys, fixed, v0=v0)


def eig_dirichlet(sys: P1System, fixed, v0=None):
    """Eigenpair with homogeneous Dirichlet conditions on the nodes ``fixed``."""
    free = ~np.asarray(fixed, dtype=bool)
    idx = np.flatnonzero(free)
    if not np.any(free[sys.steklov_nodes]):
        raise ValueError("every Steklov node is constrained")
    A = sys.A[idx][:, idx]
    B = sys.B[idx][:, idx]
    lam, uf = _smallest_steklov(A, B, None if v0 is None else v0[idx])
    u = np.zeros(sys.mesh.n_nodes)
    u[idx] = uf
    return lam, u


def eig_penalized(sys: P1System, chi, beta, v0=None):
    """lambda_beta(chi) = min (|grad u|^2 + u^2 + beta chi u^2) / ||u||^2_{dOmega}."""
    A = sys.A + beta * sys.weighted_mass(np.asarray(chi, dtype=float))
    return _smallest_steklov(A, sys.B, v0)


def rayleigh(sys: P1System, u, chi=None, beta=0.0):
    num = u @ (sys.A @ u)
    if chi is not None and beta:
        num += beta * float(np.sum(chi * sys.element_l2sq(u)))
    return num / (u @ (sys.B @ u))


def dense_check(sys: P1System, fixed):
    """Reference value via the dense Dirichlet-to-Neumann (Schur) problem."""
    import scipy.linalg as sla
    free = ~np.asarray(fixed, dtype=bool)
    bnd = np.zeros(sys.mesh.n_nodes, dtype=bool)
    bnd[sys.steklov_nodes] = True
    b = np.flatnonzero(free & bnd)
    i = np.flatnonzero(free & ~bnd)
    A = sys.A
    Abb = A[b][:, b].toarray()
    Abi = A[b][:, i]
    Aii = A[i][:, i].tocsc()
    X = splu(Aii).solve(Abi.T.toarray())
    S = Abb - Abi @ X
    Bbb = sys.B[b][:, b].toarray()
    w = sla.eigh(S, Bbb, eigvals_only=True, subset_by_index=[0, 0])
    return float(w[0])
