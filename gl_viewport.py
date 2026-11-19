"""
GLViewportMixin: the 3D GL view, camera helpers, and auto-rotate timer
=========================================================================

Shared by every tab. Mixed into a QWidget subclass (call _init_gl_view()
from __init__). Nothing here ever re-evaluates the wavefunction -- rotate/
zoom/pan are pure camera transforms handled by the GPU.
"""

import numpy as np
from PyQt5 import QtCore, QtGui
import pyqtgraph.opengl as gl


class GLViewportMixin:
    DEFAULT_ELEV = 20
    DEFAULT_AZIM = 45

    def _init_gl_view(self, bg=0.15):
        self.rotate = True
        self.bg = bg  # grayscale brightness, 0 (black) .. 1 (white)
        self.half_size = 1.0

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(QtGui.QColor.fromRgbF(self.bg, self.bg, self.bg))
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=3.0, pxMode=True)
        self.scatter.setGLOptions("translucent")
        self.gl_view.addItem(self.scatter)

        self.rotate_timer = QtCore.QTimer()
        self.rotate_timer.timeout.connect(self._on_rotate_tick)
        self.rotate_timer.start(30)  # camera-only tick; never touches point data

    def _on_bg_change(self, val):
        """Background brightness slider. Purely cosmetic -- never touches
        the point data."""
        if getattr(self, "_building", False):
            return
        self.bg = float(val)
        self.gl_view.setBackgroundColor(QtGui.QColor.fromRgbF(self.bg, self.bg, self.bg))

    def _on_rotate_tick(self):
        # isVisible() is False while this tab is in the background, so a
        # hidden tab costs nothing -- not even a camera update.
        if self.rotate and self.isVisible():
            az = self.gl_view.opts.get("azimuth", self.DEFAULT_AZIM)
            self.gl_view.opts["azimuth"] = (az + 0.3) % 360
            self.gl_view.update()

    def _reset_camera(self):
        d = max(self.half_size * 2.6, 1e-3)
        self.gl_view.setCameraPosition(distance=d, elevation=self.DEFAULT_ELEV, azimuth=self.DEFAULT_AZIM)
        self.gl_view.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.gl_view.update()

    def _fit_camera(self):
        """Refit zoom distance to the current cloud's extent, keeping the
        user's viewing angle. Call only when the sampled geometry changed."""
        d = max(self.half_size * 2.6, 1e-3)
        self.gl_view.opts["distance"] = d
        self.gl_view.update()

    def _push_scatter(self, pts, cols, n_points, max_points):
        if pts.shape[0] > 0:
            size = float(np.interp(n_points, [500, max_points], [6.0, 1.5]))
            self.scatter.setData(pos=pts, color=cols, size=size)
        else:
            self.scatter.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)), size=1.0)
