"""Numerical approximation of optimal holes for the first Steklov eigenvalue.

Companion code to

  J. Fernández Bonder, P. Groisman, J. D. Rossi, Optimization of the first
  Steklov eigenvalue in domains with holes: a shape derivative approach,
  Ann. Mat. Pura Appl. (4) 186 (2007), 341-358.
"""

from .mesh import Mesh, disk, annulus, lshape, two_disks_with_handle, distmesh2d
from .fem import P1System
from .eigen import eig_hard, eig_dirichlet, eig_penalized
from . import radial, algorithms

__all__ = [
    "Mesh", "disk", "annulus", "lshape", "two_disks_with_handle", "distmesh2d",
    "P1System", "eig_hard", "eig_dirichlet", "eig_penalized", "radial", "algorithms",
]
