"""
Physics: hydrogen-like wavefunctions
=====================================

    psi_nlm(r, theta, phi) = R_nl(r) * Y_l^m(theta, phi)

    R_nl(r) = sqrt( (2/(n a0))^3 (n-l-1)! / (2n (n+l)!) )
              * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho),   rho = 2r/(n a0)

SciPy's `eval_genlaguerre` and `sph_harm_y` already implement the associated
Laguerre polynomials and (Condon-Shortley phase) spherical harmonics needed
here, so no hand-rolled recurrences are required. Verified numerically
(Monte-Carlo integral of |psi|^2 dV ~= 1).

This module has no Qt / plotting dependencies -- it is pure NumPy/SciPy math,
importable and testable on its own.
"""

import numpy as np
from scipy import special

N_MAX = 7
A0 = 1.0  # Bohr radius, in the same "natural units" the original used
Z_MAX = 10

# Nuclear charge Z (hydrogen-LIKE ions: one electron, charge +Z nucleus).
# Bumping Z shrinks the effective Bohr radius: a0_eff = A0 / Z. This is what
# "changing the atom" means throughout this project -- everything stays a
# one-electron hydrogenic system, just with a different, more tightly bound
# nucleus.
ELEMENT_SYMBOLS = {
    1: "H", 2: "He+", 3: "Li2+", 4: "Be3+", 5: "B4+",
    6: "C5+", 7: "N6+", 8: "O7+", 9: "F8+", 10: "Ne9+",
}


def a0_for_z(z):
    return A0 / z


def element_label(z):
    return f"Z = {z} ({ELEMENT_SYMBOLS.get(z, '?')})"


def energy_level(n, z=1):
    """Exact hydrogen-like energy E_n = -Z^2 / (2 n^2), in the same atomic
    (Hartree) units this whole module is already implicitly written in:
    A0=1 means hbar = m_e = e = 1, so E_n comes out in Hartree with no
    other constants needed. This is what makes the time-evolution phase
    exp(-i E_n tau) meaningful with tau in plain "natural time units" --
    see the superposition sampler in sampling.py."""
    return -(z ** 2) / (2.0 * n ** 2)


def radial_R(r, n, l, a0=A0):
    """Radial part R_nl(r) of the hydrogen wavefunction."""
    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / (n * a0)
    norm = np.sqrt(
        (2.0 / (n * a0)) ** 3
        * special.factorial(n - l - 1)
        / (2.0 * n * special.factorial(n + l))
    )
    return norm * np.exp(-rho / 2.0) * rho ** l * special.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def psi_nlm(x, y, z, n, l, m, a0=A0):
    """Full complex wavefunction psi_nlm evaluated at Cartesian points."""
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r < 1e-12, 1e-12, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    return radial_R(r_safe, n, l, a0) * special.sph_harm_y(l, m, theta, phi)


def valid_orbitals(n_max=N_MAX):
    """All valid (n, l, m) triples, sorted the same way the JS did."""
    out = []
    for n in range(1, n_max + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                out.append((n, l, m))
    out.sort(key=lambda t: t[0] * 100 + t[1] * 10 + t[2])
    return out


ORBITAL_LIST = valid_orbitals()
ORBITAL_INDEX = {t: i for i, t in enumerate(ORBITAL_LIST)}
