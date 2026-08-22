#!/usr/bin/env sh
# Re-takes the measurements in fixtures/measured_device.json.
#
# The glasses must be attached and the phone reachable over adb. Everything it
# prints is a reading of the machine as it is at that moment, which is why the
# power state is printed first: the same phone on a charger and on a flat
# battery is not the same computer, and a number without that context is not a
# measurement.
#
# Nothing here writes to the device or changes a setting.
set -eu

adb="${ADB:-adb}"

section() {
    printf '\n## %s\n' "$1"
}

printf '# Measured %s\n' "$(date -Iseconds)"

section 'Power — read everything else against this'
"$adb" shell dumpsys battery 2>/dev/null |
    grep -E 'level|status|AC powered|USB powered|temperature' || true
printf '  battery saver low_power = %s\n' \
    "$("$adb" shell settings get global low_power 2>/dev/null | tr -d '\r')"
printf '  auto-on level = %s\n' \
    "$("$adb" shell settings get global low_power_trigger_level 2>/dev/null | tr -d '\r')"

section 'CPU — how many cores, in how many tiers, at what ceiling'
"$adb" shell 'for c in /sys/devices/system/cpu/cpu[0-9]*; do
    n=${c##*/}
    on=$(cat "$c/online" 2>/dev/null || echo 1)
    gov=$(cat "$c/cpufreq/scaling_governor" 2>/dev/null)
    cur=$(cat "$c/cpufreq/scaling_cur_freq" 2>/dev/null)
    max=$(cat "$c/cpufreq/scaling_max_freq" 2>/dev/null)
    hw=$(cat "$c/cpufreq/cpuinfo_max_freq" 2>/dev/null)
    echo "  $n online=$on gov=$gov cur=${cur:-?} max=${max:-?} hw_max=${hw:-?}"
done' 2>/dev/null || true

section 'Glasses — USB identity'
"$adb" shell dumpsys usb 2>/dev/null |
    grep -E 'product_name=VITURE|manufacturer_name=VITURE|vendor_id|product_id' |
    head -6 || true

section 'Glasses — the display, and above all which modes it is offering'
# The side-by-side modes appear only once the panel has been switched into 0x32
# and has re-advertised, so run this again after a mode change to see them.
"$adb" shell dumpsys display 2>/dev/null |
    tr ',' '\n' |
    grep -E 'VITURE|modeId|width=|height=|fps=|supportedRefreshRates|state |dpi' |
    head -40 || true

section 'Glasses — pose rate, counted over five seconds'
# The bridge logs a running count of poses received. Two readings and the clock
# between them give the rate, which is more honest than any figure the device
# claims for itself.
"$adb" logcat -c 2>/dev/null || true
sleep 5
"$adb" logcat -d 2>/dev/null | grep 'diag: head=' | tail -2 || true
printf '  (poses between the two lines, divided by the seconds between them)\n'

section 'Library reading — needs the cross-compiled bench on the device'
if "$adb" shell test -x /data/local/tmp/library_bench 2>/dev/null; then
    "$adb" shell /data/local/tmp/library_bench
else
    printf '  not present. Build and push it:\n'
    printf '    cargo build --release --target aarch64-linux-android --example library_bench\n'
    printf '    adb push target/aarch64-linux-android/release/examples/library_bench /data/local/tmp/\n'
fi
