"""Writes Google's spherical-video metadata into an MP4, as a camera would.

A file from a 360 camera does not need to be told what it is: it says so, in
two boxes inside the video track's sample description. `st3d` says how the two
eyes are packed, and `sv3d` says how the picture maps onto the world. Everything
else a player does — reading the name, measuring the aspect ratio, asking the
person watching — is guesswork standing in for these.

Which makes them the one part of the format that had never been tested here.
Every test file so far was told what it was through a debug flag, so the path
that reads a real file's own answer had never run on the device at all. This
produces files that exercise it.

    python3 inject_spatial_metadata.py in.mp4 out.mp4 --stereo top-bottom
    python3 inject_spatial_metadata.py in.mp4 out.mp4 --stereo left-right \\
        --crop-left 0.25 --crop-right 0.25      # a VR180 hemisphere

## The awkward part

Boxes go inside the sample entry, which is nested six levels deep, and every
enclosing box carries its own length — so growing the innermost one means
rewriting five sizes above it. Worse, if `moov` sits before `mdat` (which it
does in anything meant to be streamed) then growing `moov` moves every byte of
media that follows, and the chunk offset table still points at where they used
to be. That table has to be corrected by exactly the number of bytes added, or
the file plays as noise.

Both are handled here. Neither is difficult; both are easy to forget, and the
symptom of forgetting is a file that looks fine to `ffprobe` and decodes to
nothing.
"""
import argparse
import struct
import sys

# Sample entries whose payload begins with the 78-byte visual sample entry
# header, after which child boxes may be appended.
VIDEO_SAMPLE_ENTRIES = {b"avc1", b"avc3", b"hev1", b"hvc1", b"av01", b"vp09", b"mp4v"}

# Boxes that contain other boxes and therefore have to be walked into.
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta"}

STEREO_MODES = {"mono": 0, "top-bottom": 1, "left-right": 2}


def box(kind: bytes, payload: bytes) -> bytes:
    """A box: big-endian length including the header, then the four-character
    type, then the payload."""
    return struct.pack(">I", len(payload) + 8) + kind + payload


def full_box(kind: bytes, payload: bytes, version: int = 0, flags: int = 0) -> bytes:
    """A box whose payload begins with a version byte and three flag bytes."""
    return box(kind, struct.pack(">B3s", version, flags.to_bytes(3, "big")) + payload)


def st3d(mode: str) -> bytes:
    """How the two eyes are packed into one frame."""
    return full_box(b"st3d", struct.pack(">B", STEREO_MODES[mode]))


def sv3d(bounds) -> bytes:
    """How the picture maps onto the world: equirectangular, over the given
    bounds.

    The bounds are 0.32 fixed point — the whole word is fraction — and say what
    proportion of the sphere is *missing* from each edge. All zero is a full
    360; a quarter cropped from each side is the VR180 hemisphere.
    """
    top, bottom, left, right = bounds
    to_fixed = lambda v: min(int(v * 4294967296.0), 0xFFFFFFFF)  # noqa: E731
    equi = full_box(
        b"equi",
        struct.pack(">IIII", *(to_fixed(v) for v in (top, bottom, left, right))),
    )
    # Pose: yaw, pitch and roll of the camera, 16.16 fixed point. Zero means
    # the picture is oriented the way it was shot, which is what anything
    # generated is.
    prhd = full_box(b"prhd", struct.pack(">iii", 0, 0, 0))
    proj = box(b"proj", prhd + equi)
    # A note about where the metadata came from, which the specification asks
    # for and nothing reads.
    svhd = full_box(b"svhd", b"uxspace\x00")
    return box(b"sv3d", svhd + proj)


def walk(data: bytes, start: int, end: int, path=()):
    """Yields every box in a range as (type, offset, size, header_size)."""
    offset = start
    while offset + 8 <= end:
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise ValueError(f"box {kind!r} at {offset} claims {size} bytes")
        yield kind, offset, size, header, path
        if kind in CONTAINERS:
            yield from walk(data, offset + header, offset + size, path + (kind,))
        elif kind == b"stsd":
            # A full box with an entry count before its children.
            yield from walk(data, offset + header + 8, offset + size, path + (kind,))
        offset += size


def inject(data: bytes, extra: bytes) -> bytes:
    """Appends `extra` to the first video sample entry, fixing every size and
    offset that depends on it.

    Everything is patched *before* the bytes are inserted, and the order is not
    a style choice. Patching afterwards means walking a buffer whose contents
    have shifted but whose box lengths have not yet been corrected — the walk
    then reads a length from one box and lands in the middle of another, and
    the first attempt at this produced a file with no `moov` at all.
    """
    entries = [
        (offset, size)
        for kind, offset, size, _, path in walk(data, 0, len(data))
        if kind in VIDEO_SAMPLE_ENTRIES and b"stsd" in path
    ]
    if not entries:
        raise SystemExit("no video sample entry found — is this an MP4?")
    entry_offset, entry_size = entries[0]
    insert_at = entry_offset + entry_size
    grew = len(extra)

    out = bytearray(data)

    # Every box that encloses the insertion point grows by the same amount, and
    # so does the sample entry itself.
    for kind, offset, size, header, _ in walk(data, 0, len(data)):
        encloses = offset < insert_at <= offset + size
        if encloses or offset == entry_offset:
            if header == 8:
                struct.pack_into(">I", out, offset, size + grew)
            else:
                struct.pack_into(">Q", out, offset + 8, size + grew)

    # Media that sits after the insertion point moves with it, and the chunk
    # offset table still points at where it used to be. Forgetting this leaves a
    # file that `ffprobe` reads happily and that decodes to noise.
    for kind, offset, size, header, _ in walk(data, 0, len(data)):
        if kind not in (b"stco", b"co64"):
            continue
        count = struct.unpack_from(">I", data, offset + header + 4)[0]
        base = offset + header + 8
        width = 4 if kind == b"stco" else 8
        fmt = ">I" if kind == b"stco" else ">Q"
        for i in range(count):
            at = base + i * width
            value = struct.unpack_from(fmt, data, at)[0]
            if value >= insert_at:
                struct.pack_into(fmt, out, at, value + grew)

    out[insert_at:insert_at] = extra
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--stereo", choices=sorted(STEREO_MODES), default="mono")
    parser.add_argument("--crop-top", type=float, default=0.0)
    parser.add_argument("--crop-bottom", type=float, default=0.0)
    parser.add_argument("--crop-left", type=float, default=0.0)
    parser.add_argument("--crop-right", type=float, default=0.0)
    args = parser.parse_args()

    data = open(args.source, "rb").read()
    bounds = (args.crop_top, args.crop_bottom, args.crop_left, args.crop_right)
    extra = st3d(args.stereo) + sv3d(bounds)
    open(args.destination, "wb").write(inject(data, extra))

    covered_lon = (1.0 - args.crop_left - args.crop_right) * 360.0
    covered_lat = (1.0 - args.crop_top - args.crop_bottom) * 180.0
    print(f"wrote {args.destination}: {len(extra)} bytes of metadata")
    print(f"  stereo: {args.stereo}")
    print(f"  covers: {covered_lon:.0f}° across, {covered_lat:.0f}° down")


if __name__ == "__main__":
    sys.exit(main())
