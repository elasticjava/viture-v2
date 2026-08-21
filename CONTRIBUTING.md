# Contributing

## The one rule that matters

Protocol claims need evidence from hardware, not reasoning. If you add a message id
or a payload field, attach the capture it came from. The method that produced
`PROTOCOL.md` works well and is repeatable: run the official SDK, capture the bus with
`usbmon`, and match the frames against the SDK's own
`ProtocolBuilderV2: Built command - MsgID: …` log lines. That gives labelled data
instead of guesses.

Everything in `PROTOCOL.md` was verified twice — once from the capture, once by an
independent implementation driving the device without the vendor library. New findings
should clear the same bar.

## Before opening a pull request

```bash
cargo fmt --all
cargo clippy --all-targets --all-features   # CI denies warnings
cargo test --release
```

Cross-compilation is part of CI, so check the targets you affect:

```bash
cargo build --release --target aarch64-unknown-linux-musl
cargo build --release --target aarch64-linux-android   # needs the NDK linker
```

## Shape of the code

The protocol core is transport-agnostic on purpose. `Transport` is the seam: io_uring,
usbfs and hidraw all implement it, and everything above works unchanged. New platforms
should be a new transport, not a fork of the parser.

The hot path does not allocate and does not sleep. If a change adds an allocation or a
`sleep` between the kernel and the consumer, it needs a reason in the commit message —
measurements in the README show what that costs.

`unsafe` needs a `# Safety` section stating the caller's obligation. CI counts the
surface on every run.

## Commit messages

Conventional prefixes (`feat:`, `fix:`, `ci:`, `docs:`, `refactor:`). Say what changed
and why; the diff already shows how.
