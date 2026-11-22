"""Basic tests: exact formulas, FEM convergence and algorithm properties."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import steklov_holes as sh                       # noqa: E402
from steklov_holes import algorithms as al, radial  # noqa: E402
from steklov_holes.eigen import dense_check      # noqa: E402


def rk4_lambda(N, r, R, n=20000):
    """lambda_1(R, r) from the radial ODE (independent of the Bessel formula)."""
    h = (R - r) / n
    s, y = r, np.array([0.0, 1.0])
    f = lambda s, y: np.array([y[1], y[0] - (N - 1) / s * y[1]])
    for _ in range(n):
        k1 = f(s, y); k2 = f(s + h / 2, y + h / 2 * k1)
        k3 = f(s + h / 2, y + h / 2 * k2); k4 = f(s + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); s += h
    return y[1] / y[0]


@pytest.mark.parametrize("N,r,R", [(2, 0.5, 1.0), (2, 1.0, 3.0), (3, 0.3, 2.0)])
def test_bessel_formula(N, r, R):
    assert abs(radial.lambda1(R, r, N) - rk4_lambda(N, r, R)) < 1e-8


@pytest.mark.parametrize("r", [0.2, 0.5, 1.0])
def test_c1_vanishes_at_Rstar(r):
    Rs = radial.R_star(r)
    assert abs(radial.c_ell(1, Rs, r)) < 1e-7
    assert radial.c_ell(1, 0.9 * Rs, r) < 0 < radial.c_ell(1, 1.2 * Rs, r)


def _annulus_eig(h, R=1.0, r=0.5):
    m = sh.annulus(h, R, r)
    mid = np.hypot(*m.p[m.bedges].mean(1).T)
    outer = m.bedges[mid > 0.5 * (R + r)]
    inner = np.unique(m.bedges[mid <= 0.5 * (R + r)])
    S = sh.P1System(m, outer)
    fixed = np.zeros(m.n_nodes, bool)
    fixed[inner] = True
    return S, fixed, sh.eig_dirichlet(S, fixed)[0]


def test_fem_quadratic_convergence():
    ex = radial.lambda1(1.0, 0.5)
    e1 = abs(_annulus_eig(0.1)[2] - ex)
    e2 = abs(_annulus_eig(0.05)[2] - ex)
    assert e1 / e2 > 3.0          # O(h^2) (with mesh noise)
    assert e2 / ex < 2e-3


def test_iterative_vs_dense():
    S, fixed, lam = _annulus_eig(0.1)
    assert abs(lam - dense_check(S, fixed)) < 1e-9


def test_rearrangement_monotone_and_penalty_limit():
    m = sh.disk(0.08)
    S = sh.P1System(m)
    alpha = np.pi / 2
    h0 = al.initial_hole(m, alpha, center=(0.3, 0.0))
    res = al.rearrangement(S, h0, alpha, betas=(1e2,), max_inner=50)
    pen = [l for _, l in res.info["penalized"]]
    assert np.all(np.diff(pen) <= 1e-12)
    lam_h, _ = sh.eig_hard(S, res.hole)
    lam_b = [sh.eig_penalized(S, res.hole.astype(float), b)[0] for b in (1e2, 1e4, 1e6)]
    assert lam_b[0] <= lam_b[1] <= lam_b[2] <= lam_h + 1e-10
    assert abs(lam_b[2] - lam_h) / lam_h < 1e-3


def test_hole_class_Oh():
    m = sh.disk(0.08)
    alpha = 1.0
    h0 = al.initial_hole(m, alpha)
    a = m.areas
    assert a[h0].sum() >= alpha
    assert (a[h0].sum() - a[h0].min()) < alpha + a.max()


def test_translation_second_derivative_sign():
    # R < R*: local maximum, R > R*: local minimum
    assert radial.translation_second_derivative(1.0, 0.2) < 0
    assert radial.translation_second_derivative(3.0, 0.5) > 0
