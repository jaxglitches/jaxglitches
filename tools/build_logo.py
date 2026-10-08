"""Export the two-pulse logo as portable SVG and high-resolution PNG assets."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as Curve
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

DESTINATION = Path(__file__).resolve().parents[1] / "docs/assets/branding"
BLUE = "#076BB3"
AMBER = "#E59319"
INK = "#20242C"

# Smooth opposing transient pulses, based on the selected logo concept.
BLUE_POINTS = [(0, 0), (35, 0), (45, 0), (65, 17),
               (90, 44), (108, 45), (126, 44),
               (148, 44), (150, -48), (186, -48),
               (227, -52), (222, 0), (280, 0)]
AMBER_POINTS = [(0, 0), (35, 0), (45, 0), (65, -18),
                (84, -39), (97, -77), (116, -73),
                (135, -68), (160, 72), (182, 74),
                (208, 78), (214, 0), (280, 0)]


def export(name, width, height, *, icon=False, background="none"):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    if icon:
        transform = Affine2D().scale(2.7).translate(134, 512)
        linewidth = 19
    else:
        transform = Affine2D().scale(1.25).translate(24, 128)
        linewidth = 6.5
    for points, color in [(AMBER_POINTS, AMBER), (BLUE_POINTS, BLUE)]:
        path = Curve(points, [Curve.MOVETO] + [Curve.CURVE4] * 12)
        ax.add_patch(PathPatch(path, transform=transform + ax.transData,
                               fill=False, edgecolor=color, linewidth=linewidth,
                               capstyle="round", joinstyle="round"))
    if not icon:
        ax.text(412, 128, "jaxglitches", va="center", ha="left", color=INK,
                fontsize=58, fontweight="bold", fontfamily="DejaVu Sans")
    for extension in ("svg", "png"):
        fig.savefig(DESTINATION / f"{name}.{extension}", facecolor=background,
                    transparent=(background == "none"),
                    metadata={"Date": None} if extension == "svg" else {})
    plt.close(fig)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": "jaxglitches-logo"}):
        export("jaxglitches-logo", 1040, 256)
        export("jaxglitches-mark", 1024, 1024, icon=True)
        export("jaxglitches-icon", 1024, 1024, icon=True, background="white")


if __name__ == "__main__":
    main()
