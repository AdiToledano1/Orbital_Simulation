"""
Colormaps
=========

Most map directly onto Matplotlib's built-in colormaps (used here purely as
lookup tables -- no matplotlib plotting/GUI is involved). A few of the
original's hand-rolled ones (HSL wheel, blue/black/red diverging cycle,
"color by n") are recreated here since Matplotlib doesn't ship them.
"""

import colorsys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # we only use matplotlib for its colormap data, never for plotting
import matplotlib.colors
from matplotlib import colormaps as mpl_colormaps

_PARULA_ANCHORS = [
    (0.2081, 0.1663, 0.5292), (0.0116, 0.3875, 0.8820), (0.0059, 0.4086, 0.8828),
    (0.0265, 0.6137, 0.8135), (0.0384, 0.6743, 0.7436), (0.1834, 0.7177, 0.6424),
    (0.5301, 0.7491, 0.4661), (0.7525, 0.7384, 0.3768), (0.9139, 0.7258, 0.3063),
    (0.9908, 0.7653, 0.2164), (0.9683, 0.9514, 0.0755), (0.9763, 0.9831, 0.0538),
]
_parula_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("parula", _PARULA_ANCHORS)

_blueyellow_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blueYellow", [(0, 0, 1), (0, 0, 0), (1, 1, 0)]
)

_redbluecycle_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "redBlueCycle",
    [(0.00, (0, 0, 1)), (0.25, (0, 0, 0)), (0.50, (1, 0, 0)), (0.75, (0, 0, 0)), (1.00, (0, 0, 1))],
)

_N_HUES = {1: 0, 2: 45, 3: 135, 4: 90, 5: 180, 6: 225, 7: 270}
_N_LIGHT = {n: (0.6 if n == 6 else 0.3) for n in range(1, 8)}


def _color_by_n_base(n):
    h = _N_HUES.get(n, 0) / 360.0
    return np.array(colorsys.hls_to_rgb(h, _N_LIGHT.get(n, 0.3), 1.0))


COLORMAPS = {
    "Inferno": mpl_colormaps["inferno"],
    "Viridis": mpl_colormaps["viridis"],
    "Parula": _parula_cmap,
    "Grey": mpl_colormaps["gray"],
    "Cool Warm": mpl_colormaps["coolwarm"],
    "Turbo": mpl_colormaps["turbo"],
    "blueYellow": _blueyellow_cmap,
    "HSL": mpl_colormaps["hsv"],
    "Twilight": mpl_colormaps["twilight"],
    "Twilight Shifted": mpl_colormaps["twilight_shifted"],
    "Red Blue Cycle": _redbluecycle_cmap,
    "By n": None,
}
COLORMAP_NAMES = list(COLORMAPS.keys())
