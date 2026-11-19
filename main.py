"""
Hydrogen-like Atomic Orbital Viewer (GPU-accelerated)
======================================================

Every (n, l, m) up to n=7 is computed analytically at runtime with
NumPy/SciPy -- there is no baked dataset to ship.

Three tabs:
    "Single atom"    -- one hydrogen-like orbital (the original viewer).
    "LCAO (2 atoms)" -- a second nucleus is added on the z-axis, `distance`
                         (R) apart from the first, both carrying the same
                         (n, l, m) orbital, combined bonding (+) or
                         antibonding (-): Psi = psi_A +/- psi_B.
    "Hybridization"  -- same-atom mixing instead of two-atom mixing: up to
                         four terms Psi(r,tau) = sum c_i*psi_{n_i,l_i,m_i}(r)
                         *exp(-i*E_i*tau) on ONE nucleus. Time (tau) advances
                         on its own; a Speed slider sets d(tau)/dt (negative
                         reverses it, 0 pauses). Terms sharing one n are
                         exactly degenerate and never beat (the old
                         sp/sp2/sp3-style mixes are a special case of this);
                         terms with different n genuinely oscillate at the
                         exact period 2*pi/|E_i-E_j|. See sampling.py for how
                         each tick stays cheap despite this being exact, not
                         approximated.

Every tab's "Background" control is a single grayscale brightness slider
(0 = black, 1 = white) -- no color tint.

Rendering uses PyQt5 + pyqtgraph's OpenGL view (pyqtgraph.opengl), instead
of matplotlib's software 3D renderer. This matters for responsiveness:

  * The wavefunction is only ever resampled when n, l or m changes (or you
    step orbitals with the arrow keys, or -- LCAO/Hybrid tabs -- Z,
    distance, bonding sign, or a coefficient changes). That is the one
    "expensive" step, and it is precomputed once, up front, for the full
    point budget (MAX_POINTS) -- everything else (samples-shown slider,
    quantity, colormap, target/delta, cut-away) just re-masks/re-colors
    that already-computed cloud, no new physics is evaluated.
  * Rotating, zooming and panning the view is handled entirely by the
    GPU/OpenGL camera. It never touches the point data or re-runs any
    Python-side computation.
  * There is no bounding box / wireframe cube / axes panes drawn around
    the point cloud -- just the points on a plain background.

Module layout
-------------
    physics.py        -- radial_R, psi_nlm, valid orbital list, Z/element helpers
    sampling.py        -- sample_orbital / sample_lcao / sample_hybrid (+ caches)
    colormaps.py        -- colormap lookup tables
    quantities.py       -- Real/Imaginary/Modulus/Probability/Phase/Complex
    widgets.py           -- small reusable Qt widgets (no-wheel slider/combo,
                              slider factories, legend, side panel)
    gl_viewport.py       -- GLViewportMixin (3D view, camera, auto-rotate)
    base_tab.py           -- OrbitalTabBase, the shared tab "skeleton"
    nlm_controls.py        -- NLMZControlsMixin, shared by tabs 1 and 2
    tab_single_atom.py      -- SingleAtomTab
    tab_lcao.py               -- LCAOTab
    tab_hybrid.py              -- HybridTab
    main.py                     -- this file: MainWindow + entry point

Dependencies
------------
    pip install numpy scipy matplotlib pyqtgraph PyQt5 PyOpenGL

Run
---
    python main.py

Controls
--------
    n, l, m sliders        -> choose the orbital (l and m auto-clamp to
                               valid ranges). This is the ONLY thing that
                               triggers a new wavefunction sample (on the
                               LCAO tab, Distance, Bonding/Antibonding, and
                               Z also do; on the Hybridization tab, Z and
                               any term's (n,l,m,c) do -- all for the same
                               reason: they change the geometry).
    Z (nuclear charge)      -> hydrogen-LIKE ion; shrinks a0_eff = a0/Z.
    Distance R (LCAO tab)   -> separation between the two nuclei, in Bohr
                               radii. Triggers a resample.
    Bonding / Antibonding    -> +psi_B or -psi_B combination. Triggers a
      (LCAO tab)               resample.
    Speed / tau (Hybrid tab) -> tau (atomic units, hbar=1) advances on its
                               own; Speed sets d(tau)/dt (negative reverses,
                               0 pauses); "Reset tau to 0" jumps back. Never
                               resamples -- each tick only re-runs a cheap
                               accept/reject step over an already-computed
                               reservoir (see sampling.py). Terms sharing
                               one n are degenerate and won't visibly move;
                               terms with different n genuinely beat, most
                               visibly in the info panel's "points with
                               z>0" readout and in the shell of points
                               between the nucleus and the cloud's edge.
    Samples (10^x)          -> how many of the precomputed points are shown
                               (no resample -- just takes a slice)
    Quantity                -> Real / Imaginary / Modulus / Probability
                               density / Phase / Complex (phase=hue,
                               modulus=brightness)
    Colormap                -> choice of colormaps
    Reverse colormap        -> flips the colormap direction
    Target / Delta          -> keep only points whose normalized quantity
                               value falls within [target-delta,
                               target+delta] (an "iso-shell" filter).
                               Defaults show everything.
    Cut away                -> hides the octant where x>0 and z>0 to
                               reveal the inside of the point cloud
    Rotate                  -> slow automatic camera spin (camera only)
    Reset camera             -> restores the default view
    Left/Right arrow keys    -> step through all valid orbitals in
                               (n,l,m) order (or presets, on the
                               Hybridization tab)
    Mouse: left-drag         = orbit the camera
           ctrl+left-drag /
           middle-drag        = pan (move around)
           scroll wheel       = zoom
    (all mouse navigation is native GPU camera control -- smooth at any
    point count, and never recomputes anything. Sliders/dropdowns ignore
    the scroll wheel entirely -- only click/drag changes their values.)
"""

import sys

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from tab_single_atom import SingleAtomTab
from tab_lcao import LCAOTab
from tab_hybrid import HybridTab


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydrogen Orbital Viewer")
        self.resize(1400, 860)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.single_tab = SingleAtomTab()
        self.lcao_tab = LCAOTab()
        self.hybrid_tab = HybridTab()
        self.tabs.addTab(self.single_tab, "Single atom")
        self.tabs.addTab(self.lcao_tab, "LCAO (2 atoms)")
        self.tabs.addTab(self.hybrid_tab, "Hybridization")

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, event):
        if event.key() not in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            super().keyPressEvent(event)
            return
        direction = 1 if event.key() == QtCore.Qt.Key_Right else -1
        current = self.tabs.currentWidget()
        if hasattr(current, "step_orbital"):
            current.step_orbital(direction)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
