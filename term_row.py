"""
TermRow: one (n, l, m, c) term in a superposition, as a compact widget
==========================================================================

Used four times by the (generalized) Hybridization tab -- each row picks
one hydrogen-like eigenstate via three small spin boxes (auto-clamped so
l < n and |m| <= l, the same rule the n/l/m sliders elsewhere enforce) plus
a coefficient slider, and shows that state's exact energy E_n = -Z^2/(2n^2)
so the physics of "same n never beats, different n always does" is visible
right next to the controls that produce it.
"""

from PyQt5 import QtCore, QtWidgets

from physics import N_MAX
from widgets import NoWheelSpinBox, add_float_slider, sync_float_slider


class TermRow(QtWidgets.QWidget):
    """A single superposition term. Calls `on_change()` (no arguments)
    whenever n, l, m, or c changes from user interaction; read the current
    term back with `.get_term()` -> (n, l, m, c). `.set_term(...)` and
    `.set_energy(...)` update the widget programmatically without
    re-triggering on_change (matches the sync_*_slider pattern used
    elsewhere in this project)."""

    def __init__(self, title, n_init, l_init, m_init, c_init, on_change,
                 n_max=N_MAX, c_range=(-1.2, 1.2)):
        super().__init__()
        self._on_change = on_change
        self._n_max = n_max
        self._c_range = c_range
        self._c = c_init
        self._building = True

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
        self.energy_label = QtWidgets.QLabel("")
        self.energy_label.setStyleSheet("color: #999; font-size: 10px;")
        header.addStretch(1)
        header.addWidget(self.energy_label)
        outer.addLayout(header)

        nlm_row = QtWidgets.QHBoxLayout()
        self.sb_n = NoWheelSpinBox()
        self.sb_l = NoWheelSpinBox()
        self.sb_m = NoWheelSpinBox()
        self.sb_n.setRange(1, n_max)
        for label, sb in (("n", self.sb_n), ("l", self.sb_l), ("m", self.sb_m)):
            nlm_row.addWidget(QtWidgets.QLabel(label + "="))
            sb.setFixedWidth(48)
            nlm_row.addWidget(sb)
        outer.addLayout(nlm_row)

        self.s_c, self.l_c = add_float_slider(
            outer, "c", c_range[0], c_range[1], c_init, self._on_coeff_change
        )

        self.sb_n.valueChanged.connect(self._on_n_change)
        self.sb_l.valueChanged.connect(self._on_l_change)
        self.sb_m.valueChanged.connect(self._on_m_change)

        self.sb_n.setValue(n_init)
        self._clamp_l_range()
        self.sb_l.setValue(l_init)
        self._clamp_m_range()
        self.sb_m.setValue(m_init)

        self._building = False

    # -- clamping (same rule as the n/l/m sliders elsewhere: 0<=l<n, |m|<=l) --
    def _clamp_l_range(self):
        self.sb_l.setRange(0, max(self.sb_n.value() - 1, 0))

    def _clamp_m_range(self):
        l = self.sb_l.value()
        self.sb_m.setRange(-l, l)

    def _on_n_change(self, _val):
        if self._building:
            return
        self._building = True
        self._clamp_l_range()
        if self.sb_l.value() > self.sb_n.value() - 1:
            self.sb_l.setValue(self.sb_n.value() - 1)
        self._clamp_m_range()
        if abs(self.sb_m.value()) > self.sb_l.value():
            self.sb_m.setValue(max(-self.sb_l.value(), min(self.sb_m.value(), self.sb_l.value())))
        self._building = False
        self._on_change()

    def _on_l_change(self, _val):
        if self._building:
            return
        self._building = True
        self._clamp_m_range()
        if abs(self.sb_m.value()) > self.sb_l.value():
            self.sb_m.setValue(max(-self.sb_l.value(), min(self.sb_m.value(), self.sb_l.value())))
        self._building = False
        self._on_change()

    def _on_m_change(self, _val):
        if self._building:
            return
        self._on_change()

    def _on_coeff_change(self, val):
        self._c = val
        if self._building:
            return
        self._on_change()

    # -- public interface --
    def get_term(self):
        return (self.sb_n.value(), self.sb_l.value(), self.sb_m.value(), self._c)

    def set_term(self, n, l, m, c):
        self._building = True
        self.sb_n.setValue(n)
        self._clamp_l_range()
        self.sb_l.setValue(l)
        self._clamp_m_range()
        self.sb_m.setValue(m)
        lo, hi = self._c_range
        sync_float_slider(self.s_c, self.l_c, "c", c, lo, hi)
        self._c = c
        self._building = False

    def set_energy(self, e):
        self.energy_label.setText(f"E = {e:.4f}")
