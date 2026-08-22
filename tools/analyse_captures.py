"""Reads the numbers off a panel capture that no log can tell you.

Three questions, and each needs pixels rather than state:

* **Do the channels reach the right eyes?** Not "do the two halves differ" —
  they differ whichever way round they are. The proof is the *sign* of the
  parallax. A near object appears to the right in the left eye and to the left
  in the right, so the shift is negative; swapped, it is positive and the same
  size. Everything about the picture looks identical and nobody can wear it for
  ten minutes.
* **Did the anaglyph resolve?** Its output must be grey. Any colour left means
  channels were mixed rather than separated, and the eyes are still fighting.
* **Is the whole panel being used?** A picture in the middle of a black frame
  is a lost field of view, and on a screen you wear that is most of what you
  came for.

Called by `glasses_measure.sh`; standalone it takes a directory of captures.
"""
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - a host without them still runs the rest
    print("   (numpy and Pillow are needed for capture analysis)")
    raise SystemExit(0)

# The reference scene's pillar, and where it stands. From
# tools/make_ods_reference.py — change one and change the other.
PILLAR_RGB = (255, 90, 90)
PILLAR_DISTANCE_M = 1.0
HALF_IPD_M = 0.0315
PANEL_FOV_V_DEG = 25.76
PANEL_ASPECT = 16 / 9


def horizontal_fov():
    half = math.tan(math.radians(PANEL_FOV_V_DEG / 2)) * PANEL_ASPECT
    return 2 * math.degrees(math.atan(half))


def expected_shift(half_width):
    """Where the pillar should land, in pixels, and it is not `angle / fov *
    width`.

    The sphere is projected perspectively, so a small angle near the middle of
    the view maps through the tangent: `x = tan(lon) / tan(fov/2)`. The linear
    approximation is out by six per cent here, which is exactly the sort of gap
    that gets written off as measurement noise.
    """
    lon = 2 * math.asin(HALF_IPD_M / PILLAR_DISTANCE_M)
    return lon / math.tan(math.radians(horizontal_fov() / 2)) * (half_width / 2)


def eyes(image):
    a = np.asarray(image.convert("RGB")).astype(int)
    half = a.shape[1] // 2
    return a[:, :half], a[:, half:], half


def centroid(img, colour, tolerance=110):
    distance = np.abs(img - np.array(colour)).sum(axis=2)
    xs = np.where(distance < tolerance)[1]
    return (xs.mean(), len(xs)) if len(xs) else (None, 0)


def line(name, measured, expected, verdict=""):
    print("%-3s %-26s %-22s %-22s %s" % ("7", name, measured, expected, verdict))


def parallax(path):
    left, right, half = eyes(Image.open(path))
    a, na = centroid(left, PILLAR_RGB)
    b, nb = centroid(right, PILLAR_RGB)
    if a is None or b is None:
        line("eye assignment", "reference not in view", "the red pillar")
        return
    shift = b - a
    want = expected_shift(half)
    off = abs(abs(shift) - want) / want * 100
    verdict = "" if shift < 0 else "EYES SWAPPED"
    line("parallax", "%.0f px" % shift, "%.0f px" % -want, verdict)
    line("", "%.1f%% off" % off, "< 3%", "" if off < 3 else "check the field of view")


def greyness(path):
    left, right, _ = eyes(Image.open(path))
    spread = float(np.abs(left[..., 0] - left[..., 1]).mean()
                   + np.abs(left[..., 1] - left[..., 2]).mean())
    between = float(np.abs(left - right).mean())
    line("anaglyph is grey", "%.2f" % spread, "< 1", "" if spread < 1 else "colour left over")
    line("anaglyph eyes differ", "%.1f" % between, "> 3", "" if between > 3 else "both eyes the same")


def coverage(path):
    a = np.asarray(Image.open(path).convert("L")).astype(int)
    h, w = a.shape
    # A border that is uniformly dark all the way round is a picture inset into
    # the panel rather than filling it.
    edge = 8
    border = np.concatenate([
        a[:edge].ravel(), a[-edge:].ravel(), a[:, :edge].ravel(), a[:, -edge:].ravel()
    ])
    lit = float((border > 12).mean()) * 100
    line("panel filled to the edge", "%.0f%% lit" % lit, "> 20%",
         "" if lit > 20 else "the picture is inset")
    line("", "%dx%d" % (w, h), "3840x1080 in stereo")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for name, handler in (("ods360", parallax), ("anaglyph", greyness), ("real360", coverage)):
        path = root / f"{name}.png"
        if path.exists():
            handler(path)
        else:
            line(name, "no capture", "run step 6 first")


if __name__ == "__main__":
    main()
