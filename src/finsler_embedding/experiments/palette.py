"""Colours shared by the figures and the notebooks of the experiments.

No matplotlib backend is selected here (several plotting scripts select ``Agg``), so a notebook can import it.

* ``SPEED``: the one scale for every continuous quantity (speed of the currents, angle of the Swiss roll, height of the
  Matsumoto surface, community of the DiSBM). It is the thesis palette (blue, purple, pink, orange) with the dark blue
  end replaced by a pale tint of the same blue, so that the low values recede, the high ones stand out, and the ink
  arrows stay readable on every colour.
* ``shade(Z)``: the colours of a height map lit from one side, which keep the hues of the scale.
* ``categories(n)``: the same hues, each in two tones, for classes (cell types).
* ``INK``: arrows and axes. ``OURS``, ``PJM``, ``MDS``: one colour per method.
"""

import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize, to_hex, to_rgb

INK = "#0b0b0b"
THESIS = ["#072AC8", "#9A44C5", "#ff459c", "#F96C39"]  # blue, purple, pink, orange

SPEED = LinearSegmentedColormap.from_list("speed", ["#cfd6f5", *THESIS[1:]])

OURS, MDS = "#2a78d6", "#eb6834"  # as in the DiSBM comparison
PJM = THESIS[1]


def tint(color: str, amount: float = 0.5) -> str:
    """``color`` mixed with white (``amount`` of white)."""
    return to_hex([c + amount * (1 - c) for c in to_rgb(color)])


_blue, _purple, _pink, _orange = THESIS
CATEGORIES = [_blue, _orange, _purple, _pink, tint(_blue), tint(_orange), tint(_purple), tint(_pink), "#52514e", "#bdbcb7"]


def categories(n: int) -> list[str]:
    """``n`` colours for classes: the four hues of the thesis palette in two tones, then two greys (cycled after 10)."""
    return [CATEGORIES[i % len(CATEGORIES)] for i in range(n)]


def shade(Z, dx=1.0, dy=1.0, cmap=SPEED, vert_exag=1.0, strength=0.45, azdeg=315, altdeg=45):
    """RGB colours (shape of ``Z`` plus 3) of the height map ``Z`` on ``cmap``, lit from one side.

    The light only scales the brightness of each colour, so the hues stay those of the scale (the soft-light blend
    of ``LightSource.shade`` saturates them). Flat ground keeps the colour of the scale, slopes facing the light
    are brighter and the others darker, by up to ``strength``. ``dx`` and ``dy`` are the spacings of the grid.
    """
    Z = np.asarray(Z, dtype=float)
    rgb = cmap(Normalize(Z.min(), Z.max())(Z))[..., :3]
    light = LightSource(azdeg=azdeg, altdeg=altdeg).hillshade(Z, vert_exag=vert_exag, dx=dx, dy=dy)
    light = light / np.sin(np.radians(altdeg))  # 1 on flat ground
    return np.clip(rgb * (1 - strength + strength * light[..., None]), 0, 1)
