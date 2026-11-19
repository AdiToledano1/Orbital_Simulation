"""
OrbitalTabBase: the shared "skeleton" every tab is built from
==================================================================

All three tabs (single atom, LCAO, hybridization) share an enormous amount
of plumbing: the Samples/Background sliders, the Reset-camera button, the
Cut away/Rotate/Reverse-colormap checkboxes, the Quantity/Colormap
dropdowns + Legend, the Target/Delta iso-value filter at the bottom, the
mask-and-color "cheap recompute" pipeline, and the resample -> recompute ->
redraw -> refit-camera sequence that runs whenever the physics actually
changes.

This class builds all of that exactly once. Each tab subclass supplies only
what's genuinely different about it, through a small set of hooks:

    Class attributes (must set):
        MAX_POINTS         -- point budget for this tab
        DEFAULT_SAMPLES     -- initial "Samples" slider value
        DEFAULT_QUANTITY    -- initial entry in the Quantity dropdown
        DEFAULT_COLORMAP    -- initial entry in the Colormap dropdown
        DEFAULT_CUTAWAY     -- initial "Cut away" checkbox state

    Methods (must override):
        _init_extra_state()        -- set n/l/m/Z, distance/sign, coeffs, ...
        _build_top_controls(form)  -- the tab-specific sliders/widgets that
                                       appear ABOVE the shared controls
        _resample()                -- the one place physics gets evaluated;
                                       must set self.points_full, self.psi_full,
                                       self.half_size
        _info_text(shown, total)   -> str for the monospace info panel
        _hint_text()               -> str for the small usage-hint label
        step_orbital(direction)    -- arrow-key stepping behaviour

    Methods (override only if needed):
        _build_extra_gl_items()    -- e.g. nucleus marker(s) in the 3D view
        _build_extra_checkboxes(form) -- e.g. LCAO's "Show nuclei markers"

Every control that changes the physical state (n/l/m, Z, distance, bonding
sign, hybrid coefficients, ...) must end its callback by calling
`self._on_structural_change()`. Every control that only changes how the
already-sampled cloud is displayed (Samples shown, Target/Delta, Cut away,
Quantity, Colormap, Reverse) goes through `_recompute_view()` +
`_update_scatter()` only -- no physics re-evaluated. Camera controls (mouse
drag/scroll/pan, Rotate) never call either.
"""

import numpy as np
from PyQt5 import QtWidgets

from quantities import QUANTITY_NAMES, QUANTITY_FUNCS, colors_for_quantity
from colormaps import COLORMAP_NAMES
from gl_viewport import GLViewportMixin
from widgets import (
    NoWheelComboBox, add_log_slider, add_float_slider, add_checkbox,
    add_section_label, LegendWidget, build_side_panel,
)


class OrbitalTabBase(QtWidgets.QWidget, GLViewportMixin):
    # -- subclasses must override these class attributes --
    MAX_POINTS = 100_000
    DEFAULT_SAMPLES = 15_000
    DEFAULT_QUANTITY = "Real part"
    DEFAULT_COLORMAP = "Cool Warm"
    DEFAULT_CUTAWAY = False

    def __init__(self):
        super().__init__()

        # -- shared display state --
        self.n_points = self.DEFAULT_SAMPLES
        self.quantity = self.DEFAULT_QUANTITY
        self.colormap = self.DEFAULT_COLORMAP
        self.reverse = False
        self.target = 0.5
        self.delta = 0.5
        self.cutaway = self.DEFAULT_CUTAWAY

        self.points_full = None
        self.psi_full = None
        self.display_points = np.empty((0, 3))
        self.display_colors = np.empty((0, 4))
        self.last_stats = (0.0, 0.0, 0.0)

        # -- tab-specific state (n/l/m/Z, distance/sign, coeffs, ...) --
        self._init_extra_state()

        self._building = True
        self._init_gl_view()
        self._build_ui()
        self._building = False

        self._resample()
        self._recompute_view()
        self._update_scatter()
        self._reset_camera()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.gl_view, stretch=1)

        self._build_extra_gl_items()

        panel, form = build_side_panel()
        outer.addWidget(panel)

        self._build_top_controls(form)

        form.addSpacing(6)
        self.s_samples, _ = add_log_slider(
            form, "Samples", 500, self.MAX_POINTS, self.n_points, self._on_samples_change
        )
        self.s_bg, _ = add_float_slider(form, "Background", 0.0, 1.0, self.bg, self._on_bg_change)

        self.b_reset = QtWidgets.QPushButton("Reset camera")
        self.b_reset.clicked.connect(self._reset_camera)
        form.addWidget(self.b_reset)

        form.addSpacing(8)
        self.cb_cutaway = add_checkbox(form, "Cut away", self.cutaway, self._on_cutaway_toggle)
        self.cb_rotate = add_checkbox(form, "Rotate", self.rotate, self._on_rotate_toggle)
        self.cb_reverse = add_checkbox(form, "Reverse colormap", self.reverse, self._on_reverse_toggle)
        self._build_extra_checkboxes(form)

        form.addSpacing(8)
        form.addWidget(QtWidgets.QLabel("Quantity"))
        self.combo_quantity = NoWheelComboBox()
        self.combo_quantity.addItems(QUANTITY_NAMES)
        self.combo_quantity.setCurrentText(self.quantity)
        self.combo_quantity.currentTextChanged.connect(self._on_quantity)
        form.addWidget(self.combo_quantity)

        form.addWidget(QtWidgets.QLabel("Colormap"))
        self.combo_colormap = NoWheelComboBox()
        self.combo_colormap.addItems(COLORMAP_NAMES)
        self.combo_colormap.setCurrentText(self.colormap)
        self.combo_colormap.currentTextChanged.connect(self._on_colormap)
        form.addWidget(self.combo_colormap)

        form.addSpacing(8)
        self.legend = LegendWidget()
        form.addWidget(self.legend)

        form.addSpacing(10)
        add_section_label(form, "Iso-value filter (less commonly needed)")
        self.s_target, _ = add_float_slider(form, "Target", 0.0, 1.0, self.target, self._on_target_change)
        self.s_delta, _ = add_float_slider(form, "Delta", 0.0, 1.0, self.delta, self._on_delta_change)

        form.addSpacing(10)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-family: monospace; font-size: 10px; color: #ccc;")
        form.addWidget(self.info_label)

        hint = QtWidgets.QLabel(self._hint_text())
        hint.setStyleSheet("font-size: 10px; color: #888;")
        hint.setWordWrap(True)
        form.addWidget(hint)
        form.addStretch(1)

    # -- hooks with a default no-op (override only if the tab needs them) --
    def _build_extra_gl_items(self):
        """e.g. nucleus marker(s) added to self.gl_view. Called right after
        the GL view is placed in the layout, before the side panel."""
        pass

    def _build_extra_checkboxes(self, form):
        """Extra checkboxes after the standard Cut away/Rotate/Reverse
        trio (e.g. LCAO's 'Show nuclei markers')."""
        pass

    # -- hooks every subclass MUST override --
    def _init_extra_state(self):
        raise NotImplementedError

    def _build_top_controls(self, form):
        raise NotImplementedError

    def _resample(self):
        raise NotImplementedError

    def _info_text(self, shown, total):
        raise NotImplementedError

    def _hint_text(self):
        raise NotImplementedError

    def step_orbital(self, direction):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared data pipeline
    # ------------------------------------------------------------------
    def _recompute_view(self):
        """Cheap: mask/color the already-sampled cloud. No physics here."""
        n_show = min(self.n_points, self.points_full.shape[0])
        pts = self.points_full[:n_show]
        psi = self.psi_full[:n_show]
        rgb, t, stats = colors_for_quantity(psi, self.quantity, self.colormap, self.reverse, self.n)

        mask = np.abs(t - self.target) <= self.delta
        if self.cutaway:
            mask &= ~((pts[:, 0] > 0) & (pts[:, 2] > 0))

        pts_shown = pts[mask]
        rgb_shown = rgb[mask]
        alpha = np.full((rgb_shown.shape[0], 1), 0.85, dtype=float)
        self.display_points = pts_shown
        self.display_colors = np.hstack([rgb_shown, alpha]) if rgb_shown.size else np.empty((0, 4))
        self.last_stats = stats

    def _update_scatter(self):
        self._push_scatter(self.display_points, self.display_colors, self.n_points, self.MAX_POINTS)
        self.legend.update_strip(self.quantity, self.colormap, self.reverse, self.n, self.last_stats)
        shown, total = self.display_points.shape[0], self.n_points
        self.info_label.setText(self._info_text(shown, total))

    def _clamp_lm(self):
        """Shared by any tab that has self.n/self.l/self.m (single atom,
        LCAO). Unused by the hybridization tab."""
        self.l = int(np.clip(self.l, 0, self.n - 1))
        self.m = int(np.clip(self.m, -self.l, self.l))

    def _on_structural_change(self):
        """Physics actually changed: resample (expensive), recompute,
        redraw, refit camera. Every callback that touches n/l/m/Z/distance/
        sign/coefficients ends by calling this."""
        self._resample()
        self._recompute_view()
        self._update_scatter()
        self._fit_camera()

    # ------------------------------------------------------------------
    # Shared display-only callbacks (never touch the sampled physics)
    # ------------------------------------------------------------------
    def _on_samples_change(self, val):
        if self._building:
            return
        self.n_points = int(val)
        self._recompute_view()
        self._update_scatter()

    def _on_target_change(self, val):
        if self._building:
            return
        self.target = float(val)
        self._recompute_view()
        self._update_scatter()

    def _on_delta_change(self, val):
        if self._building:
            return
        self.delta = float(val)
        self._recompute_view()
        self._update_scatter()

    def _on_cutaway_toggle(self, checked):
        if self._building:
            return
        self.cutaway = bool(checked)
        self._recompute_view()
        self._update_scatter()

    def _on_rotate_toggle(self, checked):
        self.rotate = bool(checked)

    def _on_reverse_toggle(self, checked):
        if self._building:
            return
        self.reverse = bool(checked)
        self._recompute_view()
        self._update_scatter()

    def _on_quantity(self, label):
        if self._building:
            return
        self.quantity = label
        self._recompute_view()
        self._update_scatter()

    def _on_colormap(self, label):
        if self._building:
            return
        self.colormap = label
        self._recompute_view()
        self._update_scatter()
