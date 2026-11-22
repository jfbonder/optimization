"""Exact formulas for the centred configuration Omega = B_R, A = B_r.

The radial eigenfunction is phi(|x|) with phi'' + (N-1)/s phi' = phi,
phi(r) = 0.  Writing nu = (N-2)/2,

    phi(s) = s^{-nu} [K_nu(r) I_nu(s) - I_nu(r) K_nu(s)],

and, since (s^{-nu} I_nu)' = s^{-nu} I_{nu+1}, (s^{-nu} K_nu)' = -s^{-nu} K_{nu+1},

    lambda_1(R, r) = phi'(R)/phi(R)
                   = [K_nu(r) I_{nu+1}(R) + I_nu(r) K_{nu+1}(R)]
                     / [K_nu(r) I_nu(R) - I_nu(r) K_nu(R)].

lambda_1 solves the Riccati equation d lambda/dR = 1 - (N-1)/R lambda - lambda^2.
For the translation of the centred hole, Theta''(0) = (2/N) lambda dlambda/dR
(see the corrected Theorem 2 in paper 2).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive, kve


def lambda1(R, r, N=2):
    """First Steklov eigenvalue of B_R with the centred hole B_r (R > r > 0)."""
    nu = 0.5 * (N - 2)
    R = np.asarray(R, dtype=float)
    # exponentially scaled Bessel functions: I(x)=ive(x)e^x, K(x)=kve(x)e^{-x}
    # numerator and denominator are multiplied by e^{r-R}
    Kr, Ir = kve(nu, r), ive(nu, r)
    e = np.exp(-2.0 * (R - r))
    num = Kr * ive(nu + 1, R) + Ir * kve(nu + 1, R) * e
    den = Kr * ive(nu, R) - Ir * kve(nu, R) * e
    return num / den


def dlambda_dR(R, r, N=2):
    lam = lambda1(R, r, N)
    return 1.0 - (N - 1) / R * lam - lam ** 2


def theta2(R, r, N=2):
    """Second derivative at t=0 of the translated Rayleigh quotient Theta(t)."""
    return 2.0 / N * lambda1(R, r, N) * dlambda_dR(R, r, N)


def R_star(r, N=2, Rmax=200.0):
    """Unique R > r where lambda_1(., r) attains its minimum (dlambda/dR = 0).

    For r < R < R_star the translation argument of the paper applies
    (Theta''(0) < 0); for R > R_star one has Theta''(0) > 0.
    """
    f = lambda R: dlambda_dR(R, r, N)
    a = r * (1 + 1e-9) + 1e-9
    # f -> -infinity as R -> r+ ; f > 0 for R large
    b = r + 1.0
    while f(b) < 0:
        b = r + 2 * (b - r)
        if b > Rmax:
            raise RuntimeError("no sign change found")
    return brentq(f, a, b, xtol=1e-13)


def phi(s, R, r, N=2):
    """Radial eigenfunction normalised by phi(R) = 1."""
    from scipy.special import iv, kv
    nu = 0.5 * (N - 2)
    f = lambda x: x ** (-nu) * (kv(nu, r) * iv(nu, x) - iv(nu, r) * kv(nu, x))
    return f(np.asarray(s, dtype=float)) / f(R)


# ---------------------------------------------------------------------------
# Second shape variation at the centred ball
# ---------------------------------------------------------------------------

def mu_ell(ell, R, r, N=2):
    """mu_ell = g'(r)/g(r) for the mode g = s^{-nu}(A I_{ell+nu} + B K_{ell+nu})
    solving g'' + (N-1)/s g' - ell(ell+N-2)/s^2 g = g with g'(R) = lambda_1 g(R)."""
    from scipy.special import iv, ivp, kv, kvp
    nu = 0.5 * (N - 2)
    m = ell + nu
    lam = float(lambda1(R, r, N))

    def f(A, B, x):
        return x ** (-nu) * (A * iv(m, x) + B * kv(m, x))

    def df(A, B, x):
        return (x ** (-nu) * (A * ivp(m, x) + B * kvp(m, x))
                - nu * x ** (-nu - 1) * (A * iv(m, x) + B * kv(m, x)))

    a1 = df(1, 0, R) - lam * f(1, 0, R)
    b1 = df(0, 1, R) - lam * f(0, 1, R)
    A, B = b1, -a1
    return df(A, B, r) / f(A, B, r)


def c_ell(ell, R, r, N=2):
    """Coefficient of |h_ell|^2 in the second variation of lambda_1 at B_r.

    For volume preserving perturbations rho = r + eps h of the hole,
        lambda_1(A_eps) = lambda_1(B_r) + eps^2 phi'(r)^2 sum_{ell>=1} c_ell |h_ell|^2 + o(eps^2),
    with c_ell = -mu_ell - (N-1)/r and phi normalised in L^2(dB_R).
    """
    return -mu_ell(ell, R, r, N) - (N - 1) / r


def phi_prime_r(R, r, N=2):
    """phi'(r) for the radial eigenfunction normalised by ||phi||_{L^2(dB_R)} = 1."""
    from scipy.special import gamma, iv, ivp, kv, kvp
    nu = 0.5 * (N - 2)
    f = lambda x: x ** (-nu) * (kv(nu, r) * iv(nu, x) - iv(nu, r) * kv(nu, x))
    df_r = r ** (-nu) * (kv(nu, r) * ivp(nu, r) - iv(nu, r) * kvp(nu, r))
    area = 2 * np.pi ** (N / 2) / gamma(N / 2) * R ** (N - 1)
    return df_r / (abs(f(R)) * np.sqrt(area))


def translation_second_derivative(R, r, N=2):
    """d^2/dt^2 lambda_1(B_r(t e_1)) at t = 0 (exact)."""
    from scipy.special import gamma
    # int_{S_r} x_1^2/r^2 dS = |S_r| / N
    Sr = 2 * np.pi ** (N / 2) / gamma(N / 2) * r ** (N - 1)
    return 2 * phi_prime_r(R, r, N) ** 2 * c_ell(1, R, r, N) * Sr / N


def R_crit(r, ell=1, N=2, Rmax=200.0):
    """Radius R where c_ell(R, r) changes sign (R_crit >= R_star for ell = 1)."""
    f = lambda R: c_ell(ell, R, r, N)
    a = r * 1.0001 + 1e-6
    b = r + 0.5
    if f(a) > 0:
        return None
    while f(b) < 0:
        b = r + 2 * (b - r)
        if b > Rmax:
            return None
    return brentq(f, a, b, xtol=1e-12)
