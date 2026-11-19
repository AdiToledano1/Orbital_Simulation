"""
Tab 3: Hybridization / time evolution -- a superposition of up to four
exact hydrogen-like eigenstates on ONE nucleus,

    Psi(r, tau) = sum_i c_i * psi_{n_i,l_i,m_i}(r) * exp(-i E_{n_i} tau)

Pick any (n, l, m) per term (not just same-n s/p, as the old "hybridization"
feature did -- that's still reachable by choosing terms that all share one
n). Time (tau) now advances on its own, driven by a QTimer -- the "Speed"
slider sets d(tau)/dt (negative runs it backwards, 0 pauses), it does not
set tau directly. Two terms sharing the same n are exactly degenerate
(E_i = E_j), so their relative phase never changes and the combination is
frozen in time regardless of speed -- that's the honest reason the old
sp/sp2/sp3-style mixes never needed a clock. Two terms with DIFFERENT n
have different energies, so their relative phase sweeps continuously and
the density genuinely beats back and forth between them, at the exact
period 2*pi/|E_i-E_j| -- see the chat for the derivation and sampling.py's
module docstring for how this stays fast even though it's exact, not
approximated.

Where to actually *look* for the beat: it is easy to (wrongly) conclude
nothing is happening, because the two places the eye is drawn to are
exactly the two places that barely move. Right at the nucleus the lowest-n
term totally dominates the density, and far out at the edge of the cloud
whichever term has the largest n dominates -- both regions are governed by
essentially one term, so their *magnitude* barely changes with tau (only a
global phase does, which is invisible in Modulus/Probability). The actual
redistribution happens in the shell in between, where two terms' densities
are comparable in size: probability genuinely sloshes from the top half of
that shell to the bottom half and back, at the beat period. For the default
1s+2p_z preset this was verified numerically: the fraction of displayed
points with z>0 swings from ~71% to ~29% and back over one period -- a
large, real effect -- while the near-nucleus and far-edge densities each
change by only a few percent, exactly matching what you'd see by eye.
"""

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from physics import N_MAX, ELEMENT_SYMBOLS, a0_for_z, energy_level
from sampling import (
    superposition_reservoir_cached, sample_at_tau,
    SUPERPOSITION_PRESETS, SUPERPOSITION_PRESET_NAMES,
)
from base_tab import OrbitalTabBase
from term_row import TermRow
from widgets import NoWheelComboBox, add_z_slider, add_float_slider, add_section_label


class HybridTab(OrbitalTabBase):
    """Up to four (n,l,m,c) terms on a single nucleus at the origin, time-
    evolved via exp(-i E_n tau) -- see the module docstring above."""

    MAX_POINTS = 100_000
    RESERVOIR_SIZE = 500_000
    DEFAULT_SAMPLES = 15_000
    DEFAULT_QUANTITY = "Real part"
    DEFAULT_COLORMAP = "Cool Warm"
    DEFAULT_CUTAWAY = False

    SPEED_MAX = 5.0
    DEFAULT_SPEED = 1.0
    TICK_MS = 33  # ~30 fps; matches the rotate-camera timer's cadence

    _N_TERMS = 4
    _COEFF_RANGE = (-1.2, 1.2)

    def _init_extra_state(self):
        self.Z = 1
        self.tau = 0.0
        self.speed = self.DEFAULT_SPEED  # d(tau)/dt, atomic time units per second; 0 = paused
        # default: a clean two-term beat demo (different n -> genuinely
        # time-dependent), with the other two slots off (c=0)
        default_terms = SUPERPOSITION_PRESETS[0][1]
        self.term_specs = list(default_terms) + [(2, 0, 0, 0.0)] * (self._N_TERMS - len(default_terms))
        self.preset_index = 0
        self._reservoir = None
        self._term_energies = np.empty((0,))
        self.n = self.term_specs[0][0]  # nominal only -- used solely by the "By n" colormap

        self._tau_timer = QtCore.QTimer()
        self._tau_timer.timeout.connect(self._on_tau_tick)
        self._tau_timer.start(self.TICK_MS)

    def _build_extra_gl_items(self):
        # a single marker at the nucleus, for reference
        self.nucleus = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=np.array([[1.0, 1.0, 1.0, 0.9]]), size=14.0, pxMode=True
        )
        self.nucleus.setGLOptions("opaque")
        self.gl_view.addItem(self.nucleus)

    def _build_top_controls(self, form):
        add_section_label(form, "Superposition of exact eigenstates, one nucleus")
        self.s_z, self.l_z = add_z_slider(form, self.Z, self._on_z_change)

        form.addSpacing(6)
        form.addWidget(QtWidgets.QLabel("Preset"))
        self.combo_preset = NoWheelComboBox()
        self.combo_preset.addItems(SUPERPOSITION_PRESET_NAMES)
        self.combo_preset.setCurrentIndex(self.preset_index)
        self.combo_preset.currentIndexChanged.connect(self._on_preset_change)
        form.addWidget(self.combo_preset)

        form.addSpacing(6)
        add_section_label(form, "Terms (n, l, m, coefficient -- set c=0 to turn a term off)")
        self.term_rows = []
        for i in range(self._N_TERMS):
            n0, l0, m0, c0 = self.term_specs[i]
            row = TermRow(
                f"Term {i + 1}", n0, l0, m0, c0, self._make_term_cb(i),
                n_max=N_MAX, c_range=self._COEFF_RANGE,
            )
            form.addWidget(row)
            self.term_rows.append(row)
        self._refresh_energy_labels()

        form.addSpacing(10)
        add_section_label(
            form,
            "Time evolution: Psi(r,tau) = sum c_i*psi_i(r)*exp(-i*E_i*tau), hbar=1 "
            "-- tau now runs on its own; this slider sets its speed d(tau)/dt "
            "(negative runs it backwards, 0 pauses)."
        )
        self.s_speed, self.l_speed = add_float_slider(
            form, "speed", -self.SPEED_MAX, self.SPEED_MAX, self.speed, self._on_speed_change, fmt="{:.2f}"
        )
        self.tau_label = QtWidgets.QLabel(f"tau = {self.tau:.2f}")
        self.tau_label.setStyleSheet("font-family: monospace;")
        form.addWidget(self.tau_label)
        self.b_reset_tau = QtWidgets.QPushButton("Reset tau to 0")
        self.b_reset_tau.clicked.connect(self._on_reset_tau)
        form.addWidget(self.b_reset_tau)

    def _refresh_energy_labels(self):
        for row in self.term_rows:
            n, l, m, c = row.get_term()
            row.set_energy(energy_level(n, self.Z))

    # -- data pipeline --------------------------------------------------------
    def _resample(self):
        """The one expensive step: builds a fresh reservoir. Runs when Z or
        any term's (n,l,m,c) changes -- NOT when tau changes (see
        _on_tau_tick, and sampling.py's module docstring for why that
        split is exact)."""
        a0 = a0_for_z(self.Z)
        terms = [t for t in self.term_specs if abs(t[3]) > 1e-12]
        if not terms:
            self._reservoir = None
            self._term_energies = np.empty((0,))
            self.points_full = np.empty((0, 3))
            self.psi_full = np.empty((0,), dtype=complex)
            self.half_size = 1.0
            return

        terms_tuple = tuple(sorted((n, l, m, round(c, 6)) for (n, l, m, c) in terms))
        self._reservoir = superposition_reservoir_cached(
            terms_tuple, a0=a0, reservoir_size=self.RESERVOIR_SIZE
        )
        self._term_energies = np.array([energy_level(n, self.Z) for (n, l, m, c) in terms_tuple])
        self.n = terms_tuple[0][0]  # nominal, for the "By n" colormap only
        all_pts = self._reservoir["points"]
        self.half_size = float(np.max(np.abs(all_pts))) * 1.05 if all_pts.size else 1.0
        self._apply_tau()

    def _apply_tau(self):
        """The cheap step: re-run the tau-dependent accept/reject over the
        existing reservoir. No psi_nlm evaluation happens here."""
        if self._reservoir is None:
            self.points_full = np.empty((0, 3))
            self.psi_full = np.empty((0,), dtype=complex)
            return
        self.points_full, self.psi_full = sample_at_tau(
            self._reservoir, self._term_energies, self.tau, self.MAX_POINTS
        )

    def _info_text(self, shown, total):
        active = [t for t in self.term_specs if abs(t[3]) > 1e-12]
        lines = [f"nucleus: {ELEMENT_SYMBOLS.get(self.Z, '?')} (Z={self.Z})"]
        for (n, l, m, c) in sorted(active):
            lines.append(f"  ({n},{l},{m}): c={c:+.2f}, E={energy_level(n, self.Z):.4f}")
        if len(active) == 2:
            e0, e1 = energy_level(active[0][0], self.Z), energy_level(active[1][0], self.Z)
            dE = abs(e1 - e0)
            if dE > 1e-12:
                lines.append(f"beat period T=2*pi/|dE| = {2 * np.pi / dE:.3f}")
            else:
                lines.append("same energy -- no beating (frozen density)")
        elif len(active) == 0:
            lines.append("(all coefficients zero)")
        lines.append(f"tau = {self.tau:.2f}   speed = {self.speed:+.2f}")
        if self.points_full is not None and self.points_full.shape[0] > 0:
            z = self.points_full[:, 2]
            frac_pos = float(np.mean(z > 0))
            lines.append(f"points with z>0: {frac_pos * 100:.1f}%  (z<0: {(1 - frac_pos) * 100:.1f}%)")
        lines.append(f"quantity: {self.quantity}")
        lines.append(f"colormap: {self.colormap}{' (reversed)' if self.reverse else ''}")
        lines.append(f"points shown: {shown:,} / {total:,}")
        return "\n".join(lines)

    def _hint_text(self):
        return (
            "Psi(r,tau) = sum c_i * psi_i(r) * exp(-i*E_i*tau), one nucleus.\n"
            "tau advances on its own; 'speed' sets d(tau)/dt (negative\n"
            "reverses it, 0 pauses). Terms sharing the same n are exactly\n"
            "degenerate -- time won't change their density at all (only\n"
            "its complex phase, invisible in Modulus/Probability). Terms\n"
            "with different n genuinely beat: watch the 'points with z>0'\n"
            "line above, or Probability density, rather than the very\n"
            "center or the outer edge of the cloud -- the lowest-n term\n"
            "dominates near the nucleus and the highest-n term dominates\n"
            "the outer edge, so both stay almost static; the beat happens\n"
            "in the shell between them, where two terms are comparable.\n"
            "←/→ step through presets.\n"
            "drag: orbit  ·  ctrl+drag / middle-drag: pan  ·  scroll: zoom"
        )

    # -- callbacks specific to this tab --------------------------------------
    def _on_z_change(self, val):
        if self._building:
            return
        self.Z = int(val)
        self._refresh_energy_labels()
        self._on_structural_change()

    def _make_term_cb(self, i):
        def cb():
            if self._building:
                return
            self.term_specs[i] = self.term_rows[i].get_term()
            self._refresh_energy_labels()
            self._on_structural_change()
        return cb

    def _on_preset_change(self, index):
        if self._building:
            return
        self.preset_index = index
        _, preset_terms = SUPERPOSITION_PRESETS[index]
        padded = list(preset_terms) + [(2, 0, 0, 0.0)] * (self._N_TERMS - len(preset_terms))
        self.term_specs = padded
        self._building = True
        for row, term in zip(self.term_rows, padded):
            row.set_term(*term)
        self._building = False
        self._refresh_energy_labels()
        self._on_structural_change()

    def _on_speed_change(self, val):
        """Just stores d(tau)/dt for the next timer tick -- no resample, no
        redraw here (nothing on screen depends on speed until tau itself
        moves)."""
        if self._building:
            return
        self.speed = float(val)

    def _on_tau_tick(self):
        """Fires every TICK_MS ms via self._tau_timer. Advances tau by
        speed*dt and re-runs the cheap accept/reject step over the existing
        reservoir -- never rebuilds it, never refits the camera (the
        cloud's overall extent doesn't change with tau -- only which points
        within it are lit up). Skips all work while the tab is hidden or
        there's nothing to show, same pattern as the camera's rotate timer
        in gl_viewport.py."""
        if not self.isVisible() or self._reservoir is None or self.speed == 0.0:
            return
        dt = self.TICK_MS / 1000.0
        self.tau += self.speed * dt
        self.tau_label.setText(f"tau = {self.tau:.2f}")
        self._apply_tau()
        self._recompute_view()
        self._update_scatter()

    def _on_reset_tau(self):
        self.tau = 0.0
        self.tau_label.setText(f"tau = {self.tau:.2f}")
        if self._reservoir is not None:
            self._apply_tau()
            self._recompute_view()
            self._update_scatter()

    def step_orbital(self, direction):
        """direction = +1 or -1: cycle through the preset list."""
        new_index = (self.preset_index + direction) % len(SUPERPOSITION_PRESETS)
        self.combo_preset.setCurrentIndex(new_index)  # triggers _on_preset_change
