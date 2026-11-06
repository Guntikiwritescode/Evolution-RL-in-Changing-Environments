#!/usr/bin/env bash
# Fetch the MinuteLabs evolution simulator at the pinned commit and build the
# native reference harness (reference/s1ref). Needs git and a Rust toolchain.
# Source: https://github.com/minutelabsio/evolution-simulator (GPL-3.0), by Jasper Palfree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT=55af956a085189e1f307037555fc4d2b6ee2e95a
SRC="$HERE/minutelabs-evolution-simulator"
if [ ! -d "$SRC/.git" ]; then
  git clone --quiet https://github.com/minutelabsio/evolution-simulator.git "$SRC"
fi
git -C "$SRC" checkout --quiet "$COMMIT"
cargo build --release --manifest-path "$HERE/s1ref/Cargo.toml"
echo "built: $HERE/s1ref/target/release/s1ref"
