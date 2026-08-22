"""Puts a real run beside the mock baseline and says where they part company.

The point is not that they should match. Some of them cannot: a virtual panel
has its own refresh rate, its own presentation deadline and no optics at all, so
the prediction lookahead and the frame timings are *expected* to differ and the
difference is information about the stand-in rather than about the glasses.

What matters is the other set. If a format is read differently, if the parallax
comes out with a different sign, if the eyes swap — those are the simulation
telling a story the hardware does not, and every hour spent developing against
it afterwards is spent on the wrong problem.

So each row is labelled with which kind it is, rather than reduced to a tick.
"""
import sys

# Measurements that legitimately differ, and why. Anything not listed here is
# expected to agree, and a disagreement is worth looking at.
EXPECTED_TO_DIFFER = {
    "prediction lookahead": "the stand-in reports its own presentation deadline",
    "frame gap, idle": "a virtual panel composites differently",
    "frame gap, 4K sphere": "a virtual panel composites differently",
    "frame gap, head turning": "and has no head attached",
    "glasses product id": "there are no glasses in a mock run",
    "battery / charging": "different moment, different battery",
    "battery saver": "different moment",
    "2D: refresh rates": "the stand-in is whatever it was created as",
    "2D: widest mode": "likewise",
    "side-by-side: widest": "likewise",
    "side-by-side: rates": "likewise",
    "DisplayPort": "a virtual display has no cable",
    "pose rate": "the simulation delivers its own rate",
    "panel and field of view": "no optics behind a virtual panel",
}


def read(path):
    values = {}
    for raw in open(path, errors="ignore"):
        if "\t" not in raw:
            continue
        key, value = raw.rstrip("\n").split("\t", 1)
        values.setdefault(key.strip(), value.strip())
    return values


def main():
    real, mock = read(sys.argv[1]), read(sys.argv[2])
    shared = [k for k in real if k in mock]
    if not shared:
        print("   nothing in common — was the baseline taken with a different set of steps?")
        return

    surprises = 0
    for key in shared:
        same = real[key] == mock[key]
        if same:
            verdict = "same"
        elif key in EXPECTED_TO_DIFFER:
            verdict = "differs — " + EXPECTED_TO_DIFFER[key]
        else:
            verdict = "DIFFERS, and should not"
            surprises += 1
        print("   %-26s %-20s %-20s %s" % (key, real[key][:20], mock[key][:20], verdict))

    print()
    if surprises:
        print("   %d measurement(s) differ that were expected to agree. Those are the" % surprises)
        print("   ones worth reading: the simulation is telling a story the hardware")
        print("   does not, and work done against it afterwards is work on the wrong")
        print("   problem.")
    else:
        print("   Everything that was expected to agree does. The simulation is a fair")
        print("   surrogate for these measurements, and the rest differ for reasons")
        print("   that are properties of a virtual panel rather than faults.")


if __name__ == "__main__":
    main()
