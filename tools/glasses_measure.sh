#!/usr/bin/env sh
# Measures the whole stack against real glasses, in one run, and prints a table.
#
# Everything here is a reading of the hardware. There is no mock anywhere in it,
# and the run refuses to start if the glasses are not on the bus — a measurement
# taken against a stand-in panel looks exactly like a measurement and is not
# one, and that mistake is only ever found much later.
#
#   ./glasses_measure.sh              # everything, against the glasses
#   ./glasses_measure.sh 3 6          # only steps 3 and 6
#   ./glasses_measure.sh --mock       # the same battery against the stand-in
#
# The mock run exists to be compared against, not to stand in for the real one.
# Run it once and it becomes the baseline; every real run then prints a column
# showing where the two disagree. Where they agree, the simulation is a fair
# surrogate for that measurement and can be trusted between sessions. Where they
# do not, the simulation is lying about something, and the gap is the finding.
#
# Raw logs, captures and per-step detail go under ./measurements/<timestamp>/.
# What reaches the terminal is one line per step: what was measured, what was
# expected, and whether it agrees. That is deliberate — a run that prints three
# thousand lines of logcat is a run nobody reads.
set -eu

MODE=real
case "${1:-}" in --mock) MODE=mock; shift ;; esac

adb="${ADB:-adb}"
here=$(cd "$(dirname "$0")" && pwd)
stamp=$(date +%Y%m%d-%H%M%S)
out="${OUT:-$here/measurements/$stamp}"
mkdir -p "$out"

APP=com.uxspace
ACTIVITY=$APP/.MainActivity
MOVIES=/sdcard/Movies
PULL=/sdcard/Android/data/$APP/files

# --- reporting ---------------------------------------------------------------
# One line per measurement. `report <step> <what> <measured> <expected>
# <verdict>` where verdict is ok / off / n-a.
report() {
    printf '%-3s %-26s %-22s %-22s %s\n' "$1" "$2" "$3" "$4" "$5" | tee -a "$out/summary.txt"
    # Machine-readable beside the table, so a later run can compare rather than
    # asking somebody to read two tables side by side.
    printf '%s\t%s\n' "$2" "$3" >> "$out/results.tsv"
}

note() { printf '   %s\n' "$1" | tee -a "$out/summary.txt"; }

# STEPS is set before this is first called; an empty list means every step.
# `$#` here would be the *function's* argument count, which is always one — a
# mistake that silently ran nothing at all.
STEPS="$*"
wanted() {
    [ -z "$STEPS" ] && return 0
    for s in $STEPS; do
        [ "$s" = "$1" ] && return 0
    done
    return 1
}

# The battery plays films, and a measurement run that fills a room with sound
# is a measurement run somebody stops halfway through. Muted for the duration
# and put back afterwards, including if the run is interrupted.
mute_media() {
    saved_volume=$("$adb" shell cmd media_session volume --stream 3 --get 2>/dev/null |
        grep -o '[0-9]*' | tail -1 || echo "")
    "$adb" shell cmd media_session volume --stream 3 --set 0 >/dev/null 2>&1 || true
}
restore_media() {
    [ -n "${saved_volume:-}" ] || return 0
    "$adb" shell cmd media_session volume --stream 3 --set "$saved_volume" >/dev/null 2>&1 || true
}
trap 'restore_media; "$adb" shell am force-stop com.uxspace >/dev/null 2>&1 || true' EXIT INT TERM

logcat_clear() { "$adb" logcat -c 2>/dev/null || true; }
logcat_save() { "$adb" logcat -d > "$out/$1.log" 2>/dev/null || true; }
stop_app() { "$adb" shell am force-stop $APP >/dev/null 2>&1 || true; }

# --- preflight: real hardware only -------------------------------------------
printf 'Measuring into %s\n\n' "$out"

if ! "$adb" get-state >/dev/null 2>&1; then
    echo "No device. Connect the phone (adb) first." >&2
    exit 1
fi

# VITURE is vendor 0x35CA. Any product id is fine — an unknown one is worth
# measuring precisely *because* it is unknown, and the field-of-view table needs
# filling in.
usb=$("$adb" shell dumpsys usb 2>/dev/null | tr ',' '\n' | grep -A2 'manufacturer_name=VITURE' || true)
pid=$("$adb" shell dumpsys usb 2>/dev/null | grep -B4 'manufacturer_name=VITURE' | grep -o 'product_id=[0-9]*' | head -1 | cut -d= -f2 || true)
[ -n "${pid:-}" ] || pid=0

# The panel is the check that decides, and the USB reading is only a hint. A
# product id survives in `dumpsys usb` after the glasses have been unplugged —
# it was read from a cache during this very script's first run, with nothing
# attached — so trusting it alone would let a run start against no hardware and
# produce numbers that look like measurements.
#
# A panel in SurfaceFlinger's list cannot be cached in that way: it is either
# scanning or it is not.
physid=$("$adb" shell dumpsys SurfaceFlinger --display-id 2>/dev/null |
    grep -i viture | grep -o 'Display [0-9]*' | head -1 | cut -d' ' -f2 || true)
if [ "$MODE" = real ] && [ -z "${physid:-}" ]; then
    cat >&2 <<'EOF'
The glasses' panel is not there.

Either they are not plugged in, or they are plugged in and not being worn —
the panel powers itself down without a head, and from here the two look the
same.

This will not quietly fall back to the simulated panel: a run against a
stand-in produces numbers that look exactly like measurements and are not.

Put them on and run again, or ask for the baseline explicitly:

    ./glasses_measure.sh --mock
EOF
    exit 2
fi

# In mock mode every launch carries the stand-in flags, and captures come from
# the ImageReader behind it — screencap will not photograph a virtual display.
EXTRAS=""
if [ "$MODE" = mock ]; then
    # A turning head as well, so the baseline has a pose rate and a frame
    # timing under motion to be compared against. Both legitimately differ from
    # the real ones; a baseline with neither compares against nothing.
    EXTRAS="--ei com.uxspace.extra.SIMULATE_GLASSES 2 --ez com.uxspace.extra.STEREO true"
    EXTRAS="$EXTRAS --ei com.uxspace.extra.SIMULATED_MOTION 1"
fi

mute_media

printf '%-3s %-26s %-22s %-22s %s\n' "#" "measurement" "measured" "expected" "" | tee "$out/summary.txt"
printf -- '---------------------------------------------------------------------------------------\n' | tee -a "$out/summary.txt"

# --- 1. identity and power ---------------------------------------------------
if wanted 1; then
    "$adb" shell dumpsys battery > "$out/battery.txt" 2>/dev/null || true
    level=$(grep -o 'level: [0-9]*' "$out/battery.txt" | head -1 | cut -d' ' -f2 || echo '?')
    ac=$(grep -o 'AC powered: [a-z]*' "$out/battery.txt" | head -1 | cut -d' ' -f3 || echo '?')
    saver=$("$adb" shell settings get global low_power 2>/dev/null | tr -d '\r')
    report 1 "glasses product id" "0x$(printf '%04x' "$pid")" "0x1301 = Pro 2" ""
    report "" "battery / charging" "${level}% / $ac" "note it, not a pass" ""
    report "" "battery saver" "$saver" "0 preferred" ""
    note "everything below is read against this power state"
fi

# --- 2. panel modes, in both of its own modes --------------------------------
# The 3840-wide mode does not exist until the panel is switched to side-by-side
# and re-advertises. Anything that picks a mode once at startup picks from the
# wrong list, which is why this asks twice.
if wanted 2; then
    modes_now() {
        "$adb" shell dumpsys display 2>/dev/null |
            tr ',' '\n' | grep -A0 -E 'width=|height=|fps=|supportedRefreshRates' |
            tr '\n' ' '
    }
    # Only the glasses' own section. Grepping the whole dump takes the first
    # display, which is the phone — and the phone runs at 120 Hz, so the run
    # reported that the glasses did too. They do not, and a measurement that
    # answers an open question with the wrong display's number is worse than no
    # measurement at all.
    viture_section() {
        python3 - "$1" <<'PYEOF'
import sys
text = open(sys.argv[1], errors="ignore").read()
start = text.find('DisplayDeviceInfo{"VITURE"')
print(text[start:start + 3000] if start >= 0 else "")
PYEOF
    }
    "$adb" shell dumpsys display > "$out/display-2d-full.txt" 2>/dev/null || true
    viture_section "$out/display-2d-full.txt" > "$out/display-2d.txt"
    rates=$(grep -o 'supportedRefreshRates \[[^]]*\]' "$out/display-2d.txt" | head -1 || echo '?')
    widest=$(grep -o 'width=[0-9]*' "$out/display-2d.txt" | cut -d= -f2 | sort -n | tail -1 || echo '?')
    report 2 "2D: refresh rates" "${rates#supportedRefreshRates }" "[60.0, 30.0, 20.0]" ""
    report "" "2D: widest mode" "$widest" "1920" ""

    "$adb" shell am start -n $ACTIVITY --ei com.uxspace.extra.DISPLAY_MODE 50 >/dev/null 2>&1 || true
    sleep 8
    "$adb" shell dumpsys display > "$out/display-sbs-full.txt" 2>/dev/null || true
    viture_section "$out/display-sbs-full.txt" > "$out/display-sbs.txt"
    widest_sbs=$(grep -o 'width=[0-9]*' "$out/display-sbs.txt" | cut -d= -f2 | sort -n | tail -1 || echo '?')
    rates_sbs=$(grep -o 'supportedRefreshRates \[[^]]*\]' "$out/display-sbs.txt" | head -1 || echo '?')
    report "" "side-by-side: widest" "$widest_sbs" "3840" ""
    report "" "side-by-side: rates" "${rates_sbs#supportedRefreshRates }" "[60.0, ...]" ""

    # Back to 2D before anything else runs, and this is not tidiness.
    #
    # The panel re-advertises on every mode change, and Android picks from the
    # list it then offers — which includes 640x480. Cycling modes without
    # letting it settle walked it into exactly that, and from there the display
    # left DisplayManager's list entirely: the glasses were still on the bus,
    # still reporting 640x480 to dumpsys, and invisible to every app. Nothing
    # short of unplugging them brought it back.
    #
    # So: one switch, one reading, one switch home, with time to settle either
    # side.
    "$adb" shell am start -n $ACTIVITY --ei com.uxspace.extra.DISPLAY_MODE 49 >/dev/null 2>&1 || true
    sleep 8
    "$adb" shell dumpsys display > "$out/display-back.txt" 2>/dev/null || true
    back=$(viture_section "$out/display-back.txt" | grep -o '[0-9]* x [0-9]*' | head -1 || true)
    report "" "panel back in 2D" "${back:-?}" "1920 x 1080" ""

     note "above 60 here answers the 120 Hz question, and it has to come from the"
    note "glasses own section rather than the phone, which does run at 120"

    lanes=$("$adb" shell dumpsys usb 2>/dev/null | grep -o 'numLanes=[0-9]*' | head -1 || echo '?')
    train=$("$adb" shell dumpsys usb 2>/dev/null | grep -o 'linkTrainingStatus=[a-z]*' | head -1 || echo '?')
    report "" "DisplayPort" "$lanes $train" "numLanes=4 success" ""
fi

# --- 3. pose rate and stream health ------------------------------------------
# Counted rather than claimed: the device is configured for 120 and delivers
# 118.9, and one percent becomes a millisecond once a lookahead multiplies it.
if wanted 3; then
    stop_app; sleep 1; logcat_clear
    # shellcheck disable=SC2086
    "$adb" shell am start -n $ACTIVITY $EXTRAS >/dev/null 2>&1 || true
    sleep 12
    logcat_save pose
    python3 - "$out/pose.log" <<'PY' | tee -a "$out/summary.txt"
import re, sys
lines = [l for l in open(sys.argv[1], errors="ignore") if "diag: head=" in l]
def stamp(l):
    m = re.match(r"(\d+)-(\d+) (\d+):(\d+):([\d.]+)", l)
    h, mi, s = int(m.group(3)), int(m.group(4)), float(m.group(5))
    return h * 3600 + mi * 60 + s
def count(l):
    return int(re.search(r"head=(\d+)", l).group(1))
if len(lines) < 2:
    print("%-3s %-26s %-22s %-22s" % ("3", "pose rate", "no poses", "118.9 Hz"))
    raise SystemExit
a, b = lines[0], lines[-1]
dt = stamp(b) - stamp(a)
rate = (count(b) - count(a)) / dt if dt > 0 else 0
errs = re.search(r"errno=(-?\d+)", b)
print("%-3s %-26s %-22s %-22s" % ("3", "pose rate", "%.1f Hz" % rate, "118.9 Hz"))
print("%-3s %-26s %-22s %-22s" % ("", "reader errno", errs.group(1) if errs else "?", "0"))
PY
fi

# --- 4. what the display says about its own timing ---------------------------
# The prediction lookahead is derived from these; a wrong deadline is a wrong
# lead, and a wrong lead is the world lagging behind the head.
if wanted 4; then
    lead=$(grep -o 'predicting [0-9.,]* ms ahead' "$out/pose.log" 2>/dev/null | tail -1 || echo '?')
    fov=$(grep -o 'panel: [0-9]*x[0-9]*.*vertical' "$out/pose.log" 2>/dev/null | tail -1 || echo '?')
    report 4 "prediction lookahead" "${lead:-none}" "9-11 ms on the panel" ""
    report "" "panel and field of view" "${fov:-none}" "1920x1080, 25.8°" ""
fi

# --- 5. frame timing, three ways ---------------------------------------------
# Idle, with a 4K sphere decoding, and with the head turning. The panel is
# 60 Hz, so the number that matters is the median gap against 16.7 ms.
if wanted 5; then
    frames_for() {
        stop_app; sleep 1; logcat_clear
        # shellcheck disable=SC2086
        "$adb" shell am start $1 $EXTRAS >/dev/null 2>&1 || true
        sleep 35
        "$adb" logcat -d 2>/dev/null | grep 'UxSpace/Frames: frames' | tail -3 > "$out/frames-$2.txt" || true
        awk '{for(i=1;i<=NF;i++) if($i=="gap"){split($(i+1),g,"/"); s+=g[2]; n++}} END{if(n) printf "%.1f ms", s/n; else printf "?"}' "$out/frames-$2.txt"
    }
    idle=$(frames_for "-n $ACTIVITY" idle)
    report 5 "frame gap, idle" "$idle" "< 16.7 ms" ""
    if "$adb" shell test -f $MOVIES/motorist_original.mp4 2>/dev/null; then
        vid=$(frames_for "-a android.intent.action.VIEW -d file://$MOVIES/motorist_original.mp4 -t video/mp4 -n $ACTIVITY" video)
        report "" "frame gap, 4K sphere" "$vid" "< 16.7 ms" ""
    else
        report "" "frame gap, 4K sphere" "file missing" "push motorist_original.mp4" ""
    fi
    note "turn your head while the next one runs"
    sleep 3
    turn=$(frames_for "-a android.intent.action.VIEW -d file://$MOVIES/motorist_original.mp4 -t video/mp4 -n $ACTIVITY" turning)
    report "" "frame gap, head turning" "$turn" "< 16.7 ms" ""
fi

# Waits for something to appear in the log, up to a limit.
#
# Fixed sleeps were how this waited at first, and they are a guess that has to
# be wrong in one of two directions: too short and the measurement reads a log
# from before the thing happened, too long and every step pays for the worst
# case. On real hardware the wait varies by more than a factor of two, because
# a mode change renegotiates the video link and nothing is drawn until it
# comes back.
#
# Returns as soon as the pattern shows up, or after $2 seconds.
wait_for_log() {
    pattern=$1; limit=$2; waited=0
    while [ "$waited" -lt "$limit" ]; do
        if "$adb" logcat -d 2>/dev/null | grep -q "$pattern"; then return 0; fi
        sleep 2; waited=$((waited + 2))
    done
    return 1
}

# --- 6. formats, each read from the file alone -------------------------------
# No layout or projection is passed. What is being measured is whether the file
# is understood from its own metadata, which is what a camera's file carries.
if wanted 6; then
    # In 2D, deliberately. What this step measures is whether a file is
    # understood from its own metadata, and that happens before anything is
    # drawn — the panel's mode has no bearing on it. Asking for stereo here
    # cost four video-link renegotiations in a row, and the panel does not
    # survive that: it comes back offering 640x480 and nothing else. Step 7
    # switches once, for the two measurements that genuinely need two eyes.
    check_format() {
        file=$1; want_stereo=$2; want_bounds=$3; tag=$4
        if ! "$adb" shell test -f "$MOVIES/$file" 2>/dev/null; then
            report 6 "$tag" "file missing" "$file" ""
            return
        fi
        stop_app; sleep 1; logcat_clear
        "$adb" shell am start -a android.intent.action.VIEW -d "file://$MOVIES/$file" \
            -t video/mp4 -n $ACTIVITY $EXTRAS >/dev/null 2>&1 || true
        # Until the format is actually reported, rather than for a fixed guess.
        wait_for_log 'stereoMode=' 40 || true
        sleep 2
        "$adb" logcat -d > "$out/format-$tag.log" 2>/dev/null || true
        got_stereo=$(grep -o 'stereoMode=-\?[0-9]*' "$out/format-$tag.log" | tail -1 | cut -d= -f2 || echo '?')
        got_bounds=$(grep -o 'bounds [0-9., ]*' "$out/format-$tag.log" | tail -1 | sed 's/bounds //' || echo '?')
        if [ "$MODE" = real ]; then
            # A physical panel photographs; a virtual one does not.
            "$adb" shell screencap -d "$physid" -p /sdcard/_shot.png >/dev/null 2>&1 || true
            "$adb" pull /sdcard/_shot.png "$out/$tag.png" >/dev/null 2>&1 || true
        else
            "$adb" shell am start -n $ACTIVITY --es com.uxspace.extra.CAPTURE_TO "$tag.png" \
                >/dev/null 2>&1 || true
            sleep 6
            "$adb" pull "$PULL/$tag.png" "$out/$tag.png" >/dev/null 2>&1 || true
        fi
        verdict=""
        [ "$got_stereo" = "$want_stereo" ] || verdict="stereo mode differs"
        report 6 "$tag" "st3d=$got_stereo $got_bounds" "st3d=$want_stereo $want_bounds" "$verdict"
    }
    check_format ods_360_3d_tagged.mp4 1 "0.0, 0.0, 0.0, 0.0"   ods360
    check_format vr180_3d_tagged.mp4   2 "0.0, 0.0, 0.25, 0.25" vr180
    check_format motorist_original.mp4 1 "0.0, 0.0, 0.0, 0.0"   real360
    check_format ibiza_anaglyph.mp4   -1 "-"                    anaglyph
fi

# --- 7. the numbers only a picture can give ----------------------------------
# Disparity and greyness, measured off the panel itself. This is the one that
# proves the channels reach the right eyes: swapped, near objects diverge
# instead of converging, and the sign flips.
if wanted 7; then
    # The two measurements that need two eyes get the panel put into
    # side-by-side once, together, and it is put back afterwards. Once —
    # every switch renegotiates the video link, and the panel answers a rapid
    # series of them by dropping to 640x480 and staying there.
    if [ "$MODE" = real ]; then
        recapture_in_stereo() {
            file=$1; tag=$2
            got=""
            "$adb" shell test -f "$MOVIES/$file" 2>/dev/null || return 0
            stop_app; sleep 1
            "$adb" shell am start -a android.intent.action.VIEW -d "file://$MOVIES/$file" \
                -t video/mp4 -n $ACTIVITY $EXTRAS --ez com.uxspace.extra.STEREO true \
                >/dev/null 2>&1 || true
            # Two things have to happen, and the second is the slow one: the
            # file has to be understood, and the panel has to come back from
            # renegotiating the video link into side-by-side.
            wait_for_log 'stereoMode=' 40 || true
            waited=0
            while [ "$waited" -lt 40 ]; do
                got=$("$adb" shell dumpsys display 2>/dev/null | grep '"VITURE"' | head -1 \
                      | tr ',' '\n' | grep -oE '[0-9]{3,4} x [0-9]{3,4}' | head -1)
                case "$got" in 3840*) break ;; esac
                sleep 2; waited=$((waited + 2))
            done
            case "$got" in
                3840*) ;;
                *)
                    # Refuse rather than photograph. A capture taken while the
                    # panel is anywhere else is not a stereo pair, and measuring
                    # it produces a confident number about nothing — which is
                    # worse than no number, because it gets believed.
                    report 7 "$tag" "panel at ${got:-nothing}" "3840 x 1080" \
                        "the panel would not go side-by-side"
                    return 0
                    ;;
            esac
            sleep 3
            "$adb" shell screencap -d "$physid" -p /sdcard/_shot.png >/dev/null 2>&1 || true
            "$adb" pull /sdcard/_shot.png "$out/$tag.png" >/dev/null 2>&1 || true
        }
        recapture_in_stereo ods_360_3d_tagged.mp4 ods360
        recapture_in_stereo ibiza_anaglyph.mp4    anaglyph
        # Back to 2D, whatever happened above. Leaving the glasses split is a
        # bad state to hand back: every later run starts by disagreeing with it.
        stop_app; sleep 1
        "$adb" shell am start -n $ACTIVITY --ez com.uxspace.extra.STEREO false \
            >/dev/null 2>&1 || true
        sleep 8
    fi
    if [ -f "$here/analyse_captures.py" ]; then
        python3 "$here/analyse_captures.py" "$out" | tee -a "$out/summary.txt"
    else
        report 7 "capture analysis" "analyse_captures.py missing" "beside this script" ""
    fi
fi

# --- 8. reading a library ----------------------------------------------------
if wanted 8; then
    if "$adb" shell test -x /data/local/tmp/library_bench 2>/dev/null; then
        "$adb" shell /data/local/tmp/library_bench > "$out/library.txt" 2>&1 || true
        per=$(awk '$1=="1024"{print $4" "$5}' "$out/library.txt" | head -1)
        plan=$(grep -o 'threshold.*' "$out/library.txt" | head -1)
        report 8 "library, per entry" "${per:-?}" "~56 ns" ""
        report "" "parallel split" "${plan:-?}" "measured, not assumed" ""
    else
        report 8 "library benchmark" "not on device" "push library_bench" ""
    fi
fi

stop_app
printf -- '---------------------------------------------------------------------------------------\n' | tee -a "$out/summary.txt"

# --- the mock, for comparison ------------------------------------------------
baseline="$here/measurements/baseline-mock.tsv"
if [ ! -f "$out/results.tsv" ]; then
    # Steps that report only in prose — step 7 is one — leave nothing to compare.
    # Running one of them on its own is a normal thing to want, and it used to
    # end in a Python traceback about a missing file.
    printf '\nNothing tabular in this run to compare against the mock.\n' | tee -a "$out/summary.txt"
elif [ "$MODE" = mock ]; then
    cp "$out/results.tsv" "$baseline"
    printf 'Baseline written to %s — real runs compare against it.\n' "$baseline"
elif [ -f "$baseline" ]; then
    printf '\nagainst the mock:\n' | tee -a "$out/summary.txt"
    python3 "$here/compare_to_mock.py" "$out/results.tsv" "$baseline" | tee -a "$out/summary.txt"
else
    printf '\nNo mock baseline yet. Run ./glasses_measure.sh --mock once to make one.\n'
fi

printf 'Detail in %s\n' "$out"
