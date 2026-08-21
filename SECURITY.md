# Security

## Reporting

Open a [private security advisory](https://github.com/elasticjava/viture-v2/security/advisories/new).
Please do not open a public issue for anything exploitable.

## What this code touches

This is a USB driver, so the trust boundary is worth stating plainly.

**It talks to a device over a raw HID or usbfs endpoint.** Every frame that comes
back is parsed with the length and checksum checked before the payload is read, and
payload access is bounds-checked — a malformed or hostile frame yields an error, not
a read past the buffer. The parsers are covered by tests built from real captures.

**It uses `unsafe`.** Raw syscalls via inline assembly, `mmap` for the io_uring
rings, ioctls for usbfs, and a C ABI for the Android bridge. That is the price of
having no `libc` and no crate dependencies. Every `unsafe` function carries a
`# Safety` section, and CI reports the size of that surface on every run so growth is
visible in the diff.

**It has no dependencies.** Not "few" — none. `cargo-deny` runs daily and fails on
advisories, unknown registries and wildcard versions, which keeps that true and makes
any future dependency a deliberate, reviewable decision.

**It does not open the device by itself on Android.** The descriptor is handed in by
`termux-usb` or by `UsbDeviceConnection`, so the permission prompt stays with the
platform where it belongs.

## Handling of device data

The glasses return their serial number in plaintext. The CLI deliberately prints only
a fragment of it. If you attach logs to an issue, check them for that string first.

## Supported versions

The tip of `master`. This is pre-1.0; fixes go forward, not into old tags.
