"""
NLMZControlsMixin: shared (n, l, m, Z) sliders and callbacks
================================================================

Both the single-atom tab and the LCAO tab carry a self.n/self.l/self.m/
self.Z and treat all four the same way: any change is "structural" (it
changes which wavefunction gets sampled), so it ends by calling
self._on_structural_change() (provided by OrbitalTabBase). This mixin
builds the four sliders once and implements their callbacks once, instead
of duplicating this block in every tab that needs it.

Requires the class it's mixed into to also provide (from OrbitalTabBase):
self._building, self._clamp_lm(), self._on_structural_change().
"""

import numpy as np

from physics import N_MAX, ORBITAL_LIST, ORBITAL_INDEX
from widgets import add_int_slider, add_z_slider, sync_int_slider


class NLMZControlsMixin:
    def _build_nlm_z_controls(self, form):
        self.s_n, self.l_n = add_int_slider(form, "n", 1, N_MAX, self.n, self._on_n_change)
        self.s_l, self.l_l = add_int_slider(form, "l", 0, N_MAX - 1, self.l, self._on_l_change)
        self.s_m, self.l_m = add_int_slider(form, "m", -(N_MAX - 1), N_MAX - 1, self.m, self._on_m_change)
        self.s_z, self.l_z = add_z_slider(form, self.Z, self._on_z_change)

    def _on_n_change(self, val):
        if self._building:
            return
        self.n = int(val)
        self._clamp_lm()
        sync_int_slider(self.s_l, self.l_l, "l", self.l)
        sync_int_slider(self.s_m, self.l_m, "m", self.m)
        self._on_structural_change()

    def _on_z_change(self, val):
        if self._building:
            return
        self.Z = int(val)
        self._on_structural_change()

    def _on_l_change(self, val):
        if self._building:
            return
        self.l = min(int(val), self.n - 1)
        self.m = int(np.clip(self.m, -self.l, self.l))
        sync_int_slider(self.s_l, self.l_l, "l", self.l)
        sync_int_slider(self.s_m, self.l_m, "m", self.m)
        self._on_structural_change()

    def _on_m_change(self, val):
        if self._building:
            return
        self.m = int(np.clip(int(val), -self.l, self.l))
        sync_int_slider(self.s_m, self.l_m, "m", self.m)
        self._on_structural_change()

    def step_nlm(self, direction):
        """Advance (n,l,m) to the next/previous valid orbital in the
        canonical ordering and sync the sliders. The caller (the tab's
        step_orbital) is responsible for calling
        self._on_structural_change() afterwards."""
        idx = ORBITAL_INDEX[(self.n, self.l, self.m)]
        idx = (idx + direction) % len(ORBITAL_LIST)
        self.n, self.l, self.m = ORBITAL_LIST[idx]
        sync_int_slider(self.s_n, self.l_n, "n", self.n)
        sync_int_slider(self.s_l, self.l_l, "l", self.l)
        sync_int_slider(self.s_m, self.l_m, "m", self.m)
