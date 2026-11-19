"""
Tab 2: LCAO (2 atoms) -- a second nucleus is added on the z-axis, `distance`
(R) apart from the first, both carrying the same (n, l, m) orbital,
combined bonding (+) or antibonding (-): Psi = psi_A +/- psi_B. This is the
textbook LCAO picture of a homonuclear diatomic sigma/pi bond. See
sampling.sample_lcao() for how this is sampled exactly and quickly by
reusing the single-atom sampler as an envelope for rejection sampling.
"""

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph.opengl as gl

from physics import ELEMENT_SYMBOLS, a0_for_z
from sampling import sample_lcao_cached
from base_tab import OrbitalTabBase
from nlm_controls import NLMZControlsMixin
from widgets import add_float_slider, add_section_label, add_checkbox


class LCAOTab(NLMZControlsMixin, OrbitalTabBase):
    """Two same-(n,l,m) hydrogen orbitals on the z-axis, `distance` apart,
    combined bonding (+) or antibonding (-) via LCAO (see sample_lcao)."""

    MAX_POINTS = 100_000
    DEFAULT_SAMPLES = 15_000
    DEFAULT_QUANTITY =  "Probability density"
    DEFAULT_COLORMAP = "Cool Warm"
    DEFAULT_CUTAWAY = False

    def _init_extra_state(self):
        self.n, self.l, self.m = 2, 1, 0
        self.Z = 1
        self.distance = 4.0
        self.sign = 1  # +1 bonding, -1 antibonding

    def _build_extra_gl_items(self):
        # markers for the two nuclei -- updated only when distance changes
        self.nuclei = gl.GLScatterPlotItem(pos=np.zeros((2, 3)), size=14.0, pxMode=True)
        self.nuclei.setGLOptions("opaque")
        self.gl_view.addItem(self.nuclei)

    def _build_top_controls(self, form):
        add_section_label(form, "Shared orbital on each nucleus")
        self._build_nlm_z_controls(form)
        add_section_label(form, "Same element on both nuclei (homonuclear)")

        form.addSpacing(6)
        self.s_distance, _ = add_float_slider(
            form, "Distance R", 0.0, 30.0, self.distance, self._on_distance_change, fmt="{:.2f}"
        )

        form.addWidget(QtWidgets.QLabel("Combination"))
        self.radio_bonding = QtWidgets.QRadioButton("Bonding  (+)")
        self.radio_antibonding = QtWidgets.QRadioButton("Antibonding  (−)")
        self.radio_bonding.setChecked(True)
        sign_group = QtWidgets.QButtonGroup(self)
        sign_group.addButton(self.radio_bonding)
        sign_group.addButton(self.radio_antibonding)
        self.radio_bonding.toggled.connect(self._on_sign_toggle)
        self.radio_antibonding.toggled.connect(self._on_sign_toggle)
        form.addWidget(self.radio_bonding)
        form.addWidget(self.radio_antibonding)

    def _build_extra_checkboxes(self, form):
        self.cb_nuclei = add_checkbox(form, "Show nuclei markers", True, self._on_nuclei_toggle)

    # -- data pipeline --------------------------------------------------------
    def _resample(self):
        """The only step that re-evaluates the wavefunction: runs when
        n, l, m, Z, the distance, or the bonding sign changes."""
        a0 = a0_for_z(self.Z)
        self.points_full, self.psi_full = sample_lcao_cached(
            self.n, self.l, self.m, float(self.distance), self.sign, self.MAX_POINTS, a0=a0
        )
        self.half_size = float(np.max(np.abs(self.points_full))) * 1.05 if self.points_full.size else max(self.distance, 1.0)
        self._update_nuclei_markers()

    def _update_nuclei_markers(self):
        half = self.distance / 2.0
        self.nuclei.setData(
            pos=np.array([[0.0, 0.0, -half], [0.0, 0.0, half]]),
            color=np.array([[1.0, 1.0, 1.0, 0.9], [1.0, 1.0, 1.0, 0.9]]),
            size=14.0,
        )

    def _info_text(self, shown, total):
        sign_char = "+" if self.sign > 0 else "−"
        sign_txt = "bonding (+)" if self.sign > 0 else "antibonding (−)"
        a0 = a0_for_z(self.Z)
        nlm = f"{self.n},{self.l},{self.m}"
        return (
            f"LCAO combination being plotted:\n"
            f"Ψ(r) = ψ_A(r) {sign_char} ψ_B(r)   [{sign_txt}]\n"
            f"ψ_A(r) = ψ_{{{nlm}}}(r − R_A)\n"
            f"ψ_B(r) = ψ_{{{nlm}}}(r − R_B)\n"
            f"R_A=(0,0,−R/2), R_B=(0,0,+R/2), R={self.distance:.2f}\n"
            f"(coefficients on ψ_A, ψ_B are equal, c_A=c_B=1 --\n"
            f" see the chat for why that's the only physically\n"
            f" meaningful choice for two identical nuclei)\n"
            f"nucleus: {ELEMENT_SYMBOLS.get(self.Z, '?')} (Z={self.Z}), a0_eff={a0:.3f}\n"
            f"quantity: {self.quantity}\n"
            f"colormap: {self.colormap}{' (reversed)' if self.reverse else ''}\n"
            f"points shown: {shown:,} / {total:,}"
        )

    def _hint_text(self):
        return (
            "Psi = psi_A ± psi_B, same (n,l,m) on each nucleus,\n"
            "nuclei on the z-axis.\n"
            "←/→ step orbitals (n,l,m); R and sign stay fixed.\n"
            "drag: orbit  ·  ctrl+drag / middle-drag: pan  ·  scroll: zoom"
        )

    # -- callbacks specific to this tab --------------------------------------
    def _on_distance_change(self, val):
        if self._building:
            return
        self.distance = float(val)
        self._on_structural_change()

    def _on_sign_toggle(self, checked):
        if self._building:
            return
        # radio_bonding.toggled fires for both the newly-checked and the
        # newly-unchecked button; only act on the "checked" transition
        if not checked:
            return
        self.sign = 1 if self.radio_bonding.isChecked() else -1
        self._on_structural_change()

    def _on_nuclei_toggle(self, checked):
        self.nuclei.setVisible(bool(checked))

    def step_orbital(self, direction):
        """direction = +1 or -1. distance/sign stay fixed."""
        self.step_nlm(direction)
        self._on_structural_change()
