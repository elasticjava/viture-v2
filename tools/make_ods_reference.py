"""A stereoscopic 360 test scene whose depth can be checked with arithmetic.

Real 360 footage tells you that the two eyes differ. It does not tell you
whether they differ *correctly*, and the difference between correct stereo and
plausible stereo is the difference between a scene you can look around in and
one that gives you a headache after ten minutes. So this scene is built rather
than filmed, with everything at a distance that was chosen, so the disparity
each object should show is a number that can be worked out beforehand and
compared against what arrives.

Omnidirectional stereo is the projection every stereoscopic 360 camera and
renderer produces, and it is a compromise with a known shape. A single pair of
eyes cannot be right in every direction at once — turn your head and the eyes
must swing with it — so each *column* of the image is rendered from a different
viewpoint: the eyes ride a circle of half the interpupillary distance, always
tangent to it, facing the direction that column represents. Look straight
ahead and it is exactly right. Look up, and the disparity falls away to nothing
at the pole, which is the artefact everybody has seen and nobody can remove.

What the scene contains, and why:

* **Pillars at 1, 2, 4 and 8 metres**, at four headings. Their disparity is
  `2 * asin(r / d)` in longitude, where `r` is half the interpupillary
  distance — a metre away that is 1.8 degrees and eight metres away it is
  0.23, so the near one should be obviously offset between the eyes and the
  far one nearly still. That ratio is the whole test.
* **A sky at infinity**, which must show *no* offset at all. Any disparity in
  the stars means the eyes have been given a parallax that does not exist, and
  the scene will feel like a diorama rather than a place.
* **A ground plane in a grid**, because a floor is the surface a person judges
  scale against, and a wrong vertical field of view shows up there first.
"""
import math
import sys

import numpy as np

WIDTH = 4096
HEIGHT = 2048
# Half the interpupillary distance. 63 mm is the adult average, so 31.5 mm.
EYE_RADIUS = 0.0315
EYE_HEIGHT = 1.6

# Pillars: (distance in metres, heading in degrees, colour).
PILLARS = [
    (1.0, 0.0, (255, 90, 90)),
    (2.0, 90.0, (90, 255, 120)),
    (4.0, 180.0, (255, 220, 80)),
    (8.0, 270.0, (110, 160, 255)),
]
PILLAR_RADIUS = 0.12


def directions():
    """Unit ray direction for every pixel of an equirectangular frame.

    Longitude runs from -pi at the left edge to +pi at the right, zero straight
    ahead; latitude from +pi/2 at the top row to -pi/2 at the bottom. Straight
    ahead is -Z, up is +Y, right is +X — the same convention the renderer
    builds its sphere to, which is the only reason the two agree.
    """
    u = (np.arange(WIDTH) + 0.5) / WIDTH
    v = (np.arange(HEIGHT) + 0.5) / HEIGHT
    lon = (u - 0.5) * 2.0 * math.pi
    lat = (0.5 - v) * math.pi
    lon = lon[None, :]
    lat = lat[:, None]
    cos_lat = np.cos(lat)
    return (
        cos_lat * np.sin(lon),
        np.broadcast_to(np.sin(lat), (HEIGHT, WIDTH)),
        -cos_lat * np.cos(lon),
        lon,
    )


def eye_origins(lon, sign):
    """Where each column's eye sits, for omnidirectional stereo.

    The eyes ride a circle of radius `EYE_RADIUS`, tangent to it, facing the
    direction the column represents. `sign` is -1 for the left eye and +1 for
    the right: the offset is perpendicular to the view direction, which for a
    longitude `lon` is the direction `lon + 90 degrees`.
    """
    offset = sign * EYE_RADIUS
    ox = offset * np.cos(lon)
    oz = offset * np.sin(lon)
    return (
        np.broadcast_to(ox, (HEIGHT, WIDTH)).astype(np.float32),
        np.zeros((HEIGHT, WIDTH), np.float32),
        np.broadcast_to(oz, (HEIGHT, WIDTH)).astype(np.float32),
    )


def render(sign):
    dx, dy, dz, lon = directions()
    dx = dx.astype(np.float32)
    dy = dy.astype(np.float32)
    dz = dz.astype(np.float32)
    ox, oy, oz = eye_origins(lon, sign)

    image = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    depth = np.full((HEIGHT, WIDTH), np.inf, np.float32)

    # The sky, at infinity: a field of stars that must not move between the
    # eyes. Generated from a fixed seed so both eyes get the same sky — a sky
    # that differed would be a parallax where there is no object.
    rng = np.random.default_rng(7)
    image[:] = np.array([0.02, 0.02, 0.05], np.float32)
    stars = rng.random((HEIGHT, WIDTH)) > 0.9995
    image[stars] = np.array([0.8, 0.85, 1.0], np.float32)

    # Ground plane at y = -EYE_HEIGHT, in a one-metre grid.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-EYE_HEIGHT - oy) / dy
    hits_ground = (dy < -1e-6) & (t > 0)
    gx = ox + dx * t
    gz = oz + dz * t
    # Fade with distance, or the horizon becomes an aliased mess.
    distance = np.sqrt(np.maximum(gx * gx + gz * gz, 1e-6))
    fade = np.clip(1.0 - distance / 25.0, 0.0, 1.0)
    grid = (np.minimum(np.abs(gx % 1.0 - 0.5), np.abs(gz % 1.0 - 0.5)) > 0.47)
    ground = np.where(grid[..., None], np.array([0.35, 0.38, 0.45]), np.array([0.10, 0.11, 0.14]))
    ground = (ground * fade[..., None]).astype(np.float32)
    apply = hits_ground & (t < depth)
    image[apply] = ground[apply]
    depth[apply] = t[apply]

    # Pillars: infinite vertical cylinders, so the intersection is a circle
    # problem in the horizontal plane alone.
    for dist, heading, colour in PILLARS:
        rad = math.radians(heading)
        cx, cz = dist * math.sin(rad), -dist * math.cos(rad)
        # |o + t*d - c|^2 = r^2, in x and z only.
        ex, ez = ox - cx, oz - cz
        a = dx * dx + dz * dz
        b = 2.0 * (ex * dx + ez * dz)
        c = ex * ex + ez * ez - PILLAR_RADIUS * PILLAR_RADIUS
        disc = b * b - 4.0 * a * c
        hit = disc > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (-b - np.sqrt(np.maximum(disc, 0))) / (2.0 * a)
        # Three metres of pillar, standing on the ground.
        y = oy + dy * t
        hit &= (t > 0) & (y > -EYE_HEIGHT) & (y < -EYE_HEIGHT + 3.0) & (t < depth)
        shade = np.clip(1.0 - (y + EYE_HEIGHT) / 6.0, 0.3, 1.0)
        painted = (np.array(colour, np.float32) / 255.0)[None, None, :] * shade[..., None]
        image[hit] = painted[hit]
        depth[hit] = t[hit]

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def main(out_path):
    from PIL import Image

    left = render(-1.0)
    right = render(+1.0)
    # Over-under, left eye on top: what the driver's `uv_window` expects, and
    # what most stereoscopic 360 material uses because halving the vertical
    # resolution of a 2:1 frame costs less than halving the horizontal.
    stacked = np.concatenate([left, right], axis=0)
    Image.fromarray(stacked).save(out_path)
    print(f"wrote {out_path}: {stacked.shape[1]}x{stacked.shape[0]}")

    # What the disparity should be, for checking against what arrives.
    print("\nexpected horizontal offset between the eyes, at the equator:")
    for dist, heading, _ in PILLARS:
        degrees = 2.0 * math.degrees(math.asin(min(EYE_RADIUS / dist, 1.0)))
        pixels = degrees / 360.0 * WIDTH
        print(f"  {dist:>4.1f} m at {heading:>5.1f}°: {degrees:5.2f}° = {pixels:5.1f} px")
    print("  infinity (the sky):  0.00° =   0.0 px")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ods_stereo_360.png")
