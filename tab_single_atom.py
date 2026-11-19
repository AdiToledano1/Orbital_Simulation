"""
Tab 1: single atom -- one hydrogen-like orbital (the original viewer).
"""

import numpy as np

from physics import ELEMENT_SYMBOLS, a0_for_z
from sampling import sample_orbital_cached
from base_tab import OrbitalTabBase
from nlm_controls import NLMZControlsMixin


class SingleAtomTab(NLMZControlsMixin, OrbitalTabBase):
    MAX_POINTS = 150_000
    DEFAULT_SAMPLES = 20_000
    DEFAULT_QUANTITY = "Modulus"
    DEFAULT_COLORMAP = "Inferno"
    DEFAULT_CUTAWAY = True

    def _init_extra_state(self):
        self.n, self.l, self.m = 2, 1, 1
        self.Z = 1

    def _build_top_controls(self, form):
        self._build_nlm_z_controls(form)

    # -- data pipeline --------------------------------------------------------
    def _resample(self):
        """The only step that re-evaluates the wavefunction."""
        a0 = a0_for_z(self.Z)
        self.points_full, self.psi_full = sample_orbital_cached(self.n, self.l, self.m, self.MAX_POINTS, a0=a0)
        self.half_size = float(np.max(np.abs(self.points_full))) * 1.05 if self.points_full.size else 1.0

    def _info_text(self, shown, total):
        a0 = a0_for_z(self.Z)
        return (
            f"orbital (n,l,m) = ({self.n}, {self.l}, {self.m})\n"
            f"nucleus: {ELEMENT_SYMBOLS.get(self.Z, '?')} (Z={self.Z}), a0_eff = {a0:.3f}\n"
            f"quantity: {self.quantity}\n"
            f"colormap: {self.colormap}{' (reversed)' if self.reverse else ''}\n"
            f"points shown: {shown:,} / {total:,}"
        )

    def _hint_text(self):
        return (
            "←/→ step orbitals\n"
            "drag: orbit  ·  ctrl+drag / middle-drag: pan  ·  scroll: zoom"
        )

    def step_orbital(self, direction):
        """direction = +1 or -1. Used by the main window's arrow keys."""
        self.step_nlm(direction)
        self._on_structural_change()
