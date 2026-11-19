"""
Quantities: what scalar (or hue+brightness) to derive from a complex psi
==========================================================================

Phase (and Complex, which folds phase into hue) are CYCLIC quantities --
angle theta and theta+2*pi are the same physical phase, not two different
values on a line. A regular sequential/diverging colormap (Inferno,
Viridis, Cool Warm, ...) has no way to represent that: its two ends are
different colors, so mapping [-pi, pi] through one always draws a fake seam
right where the phase wraps around, even though nothing physical happens
there. The fix used below is the standard "domain coloring" trick: phase
maps straight to HUE on an HSV wheel, which is inherently periodic (hue=0
and hue=1 are the same red), so -pi and +pi -- and, by construction, 0 and
2*pi -- always render identically. Because of this, Phase (like Complex)
ignores the Colormap dropdown entirely and always draws its own hue wheel;
that dropdown only affects the four genuinely linear quantities (Real,
Imaginary, Modulus, Probability density).
"""

import numpy as np
import matplotlib.colors

from colormaps import COLORMAPS, _color_by_n_base

QUANTITY_FUNCS = {
    "Real part": lambda psi: psi.real,
    "Imaginary part": lambda psi: psi.imag,
    "Modulus": lambda psi: np.abs(psi),
    "Probability density": lambda psi: np.abs(psi) ** 2,
    "Phase": lambda psi: np.angle(psi),  # radians, in (-pi, pi]
    "Complex": None,  # handled specially: hue=phase, brightness=modulus
}
QUANTITY_NAMES = list(QUANTITY_FUNCS.keys())


def _phase_hue(phase, reverse):
    """theta in (-pi, pi] -> hue in [0, 1), wrapping so -pi and +pi (the
    same angle mod 2*pi) land on the identical hue -- and so hue(0) and
    hue(2*pi) would too, since hue is mod-1 by construction."""
    hue = np.mod((phase + np.pi) / (2 * np.pi), 1.0)
    return (1.0 - hue) if reverse else hue


def normalize_quantity(values, quantity):
    """Map raw quantity values to t in [0,1], plus (lo, mid, hi) for the legend."""
    if quantity == "Phase":
        lo, hi = -np.pi, np.pi
    elif quantity in ("Modulus", "Probability density"):
        lo, hi = 0.0, float(values.max()) if values.size else 1.0
    else:  # Real part / Imaginary part: symmetric about 0
        vmax = float(np.abs(values).max()) if values.size else 1.0
        lo, hi = -vmax, vmax
    hi = lo + 1e-12 if hi <= lo else hi
    t = (values - lo) / (hi - lo)
    return np.clip(t, 0.0, 1.0), (lo, (lo + hi) / 2.0, hi)


def colors_for_quantity(psi_vals, quantity, colormap_name, reverse, n):
    """RGB colors (+ raw normalized position t, + (lo, mid, hi) for the legend)."""
    if quantity == "Complex":
        modulus = np.abs(psi_vals)
        mmax = float(modulus.max()) if modulus.size else 1.0
        mmax = mmax if mmax > 0 else 1.0
        hue = _phase_hue(np.angle(psi_vals), reverse)
        val = np.clip(modulus / mmax, 0.0, 1.0)
        hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return rgb, val, (0.0, mmax / 2.0, mmax)

    if quantity == "Phase":
        phase = np.angle(psi_vals)
        hue = _phase_hue(phase, reverse)
        hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        t = np.mod((phase + np.pi) / (2 * np.pi), 1.0)
        return rgb, t, (-np.pi, 0.0, np.pi)

    raw = QUANTITY_FUNCS[quantity](psi_vals)
    t, stats = normalize_quantity(raw, quantity)
    t_disp = 1.0 - t if reverse else t
    if colormap_name == "By n":
        base = _color_by_n_base(n)
        rgb = t_disp[:, None] * base[None, :]
    else:
        rgb = COLORMAPS[colormap_name](t_disp)[:, :3]
    return rgb, t, stats


def colorbar_strip(quantity, colormap_name, reverse, n, stats, width=256):
    """A width x1 RGB strip (uint8) for the legend, plus the (lo, mid, hi) labels."""
    grid = np.linspace(0.0, 1.0, width)
    if quantity in ("Complex", "Phase"):
        # grid already runs 0..1 = one full, seamless trip around the hue
        # wheel -- draw it directly (reverse just flips direction). The
        # strip only shows the hue axis (full brightness) even for
        # Complex, whose brightness also varies with modulus point-by-point.
        hue = (1.0 - grid) if reverse else grid
        hsv = np.stack([hue, np.ones(width), np.ones(width)], axis=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
    else:
        t_disp = 1.0 - grid if reverse else grid
        if colormap_name == "By n":
            base = _color_by_n_base(n)
            rgb = t_disp[:, None] * base[None, :]
        else:
            rgb = COLORMAPS[colormap_name](t_disp)[:, :3]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
