"""
Small reusable Qt widgets/helpers, shared by every tab
=========================================================

Nothing tab-specific lives here -- just building blocks: sliders that
ignore the scroll wheel (only click/drag should change a value), the
labeled-slider factory functions, the nuclear-charge (Z) slider, the
legend widget, and the fixed-width scrollable side panel every tab uses.
"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from physics import Z_MAX, element_label
from quantities import colorbar_strip


class NoWheelSlider(QtWidgets.QSlider):
    """A QSlider that ignores mouse-wheel events instead of stepping its
    value. Qt sliders change value on scroll by default, which is easy to
    trigger by accident while scrolling the side panel or the 3D view --
    only click/drag on the handle or track should move these."""

    def wheelEvent(self, event):
        event.ignore()  # let the event bubble up to the scroll area instead


class NoWheelComboBox(QtWidgets.QComboBox):
    """Same idea as NoWheelSlider: a stray scroll shouldn't silently change
    the selected orbital/quantity/colormap/preset."""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QtWidgets.QSpinBox):
    """Same idea again, for the compact n/l/m spin boxes used by TermRow
    (term_row.py) -- a stray scroll over a spin box shouldn't change which
    orbital a superposition term points at."""

    def wheelEvent(self, event):
        event.ignore()


def add_int_slider(form, name, lo, hi, init, callback):
    label = QtWidgets.QLabel(f"{name} = {init}")
    slider = NoWheelSlider(QtCore.Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(init)
    slider.valueChanged.connect(lambda v: (label.setText(f"{name} = {v}"), callback(v)))
    form.addWidget(label)
    form.addWidget(slider)
    return slider, label


def add_float_slider(form, name, lo, hi, init, callback, steps=1000, fmt="{:.2f}"):
    label = QtWidgets.QLabel(f"{name} = {fmt.format(init)}")
    slider = NoWheelSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, steps)
    slider.setValue(int(round((init - lo) / (hi - lo) * steps)))

    def on_change(v):
        val = lo + (hi - lo) * v / steps
        label.setText(f"{name} = {fmt.format(val)}")
        callback(val)

    slider.valueChanged.connect(on_change)
    form.addWidget(label)
    form.addWidget(slider)
    return slider, label


def add_log_slider(form, name, lo, hi, init, callback, steps=1000):
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    label = QtWidgets.QLabel(f"{name} = {init:,}")
    slider = NoWheelSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, steps)
    init_frac = (np.log10(init) - log_lo) / (log_hi - log_lo)
    slider.setValue(int(round(init_frac * steps)))

    def on_change(v):
        val = int(round(10 ** (log_lo + (log_hi - log_lo) * v / steps)))
        label.setText(f"{name} = {val:,}")
        callback(val)

    slider.valueChanged.connect(on_change)
    form.addWidget(label)
    form.addWidget(slider)
    return slider, label


def add_z_slider(form, init, callback):
    """Nuclear charge Z slider -- its label shows the element symbol
    (H, He+, Li2+, ...) rather than a bare number."""
    label = QtWidgets.QLabel(element_label(init))
    slider = NoWheelSlider(QtCore.Qt.Horizontal)
    slider.setRange(1, Z_MAX)
    slider.setValue(init)

    def on_change(v):
        label.setText(element_label(v))
        callback(v)

    slider.valueChanged.connect(on_change)
    form.addWidget(label)
    form.addWidget(slider)
    return slider, label


def sync_z_slider(slider, label, value):
    slider.blockSignals(True)
    slider.setValue(value)
    slider.blockSignals(False)
    label.setText(element_label(value))


def add_section_label(form, text):
    """A word-wrapping QLabel for longer descriptive text (plain single-word
    headers like 'Quantity' don't need this -- only text long enough to
    clip against the 250px-wide side panel)."""
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    form.addWidget(label)
    return label


def add_checkbox(form, name, init, callback):
    cb = QtWidgets.QCheckBox(name)
    cb.setChecked(init)
    cb.toggled.connect(callback)
    form.addWidget(cb)
    return cb


def sync_int_slider(slider, label, name, value):
    slider.blockSignals(True)
    slider.setValue(value)
    slider.blockSignals(False)
    label.setText(f"{name} = {value}")


def sync_float_slider(slider, label, name, value, lo, hi, steps=1000, fmt="{:.2f}"):
    slider.blockSignals(True)
    slider.setValue(int(round((value - lo) / (hi - lo) * steps)))
    slider.blockSignals(False)
    label.setText(f"{name} = {fmt.format(value)}")


class LegendWidget(QtWidgets.QWidget):
    """The colorbar strip + lo/mid/hi tick labels. Only ever redrawn when
    the underlying data/quantity/colormap changes -- never per frame."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Legend"))
        self.strip_label = QtWidgets.QLabel()
        self.strip_label.setFixedHeight(20)
        layout.addWidget(self.strip_label)
        self.ticks_label = QtWidgets.QLabel("0")
        self.ticks_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.ticks_label.setWordWrap(True)
        layout.addWidget(self.ticks_label)

    def update_strip(self, quantity, colormap_name, reverse, n, stats):
        strip = colorbar_strip(quantity, colormap_name, reverse, n, stats, width=256)
        img = QtGui.QImage(strip.tobytes(), strip.shape[0], 1, strip.shape[0] * 3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.strip_label.width() or 240, 20, QtCore.Qt.IgnoreAspectRatio
        )
        self.strip_label.setPixmap(pix)
        lo, mid, hi = stats
        self.ticks_label.setText(f"{lo:.3g}   –   {mid:.3g}   –   {hi:.3g}")


def build_side_panel():
    """A fixed-width scrollable side panel; returns (scroll_area, form_layout)."""
    panel = QtWidgets.QScrollArea()
    panel.setFixedWidth(270)
    panel.setWidgetResizable(True)
    panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    panel_inner = QtWidgets.QWidget()
    panel_inner.setMaximumWidth(250)
    panel.setWidget(panel_inner)
    form = QtWidgets.QVBoxLayout(panel_inner)
    form.setSpacing(6)
    return panel, form
