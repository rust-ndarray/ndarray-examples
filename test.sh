#!/bin/sh

set -x
set -e

FEATURES=$1
CHANNEL=$2

if [[ "$CHANNEL" != "beta" ]]; then
    rustup component add rustfmt
    cargo fmt --all -- --check
    rustup component add clippy
fi

# Loop over the directories in the project, skipping the target directory
for f in *; do
    if [[ -d ${f} ]] && [[ ${f} != "target" ]]; then
        # Will not run if no directories are available
        echo "\n\nTesting '${f}' example.\n\n"
        cd ${f}
        cargo run --features "${FEATURES}"
        if [[ "$CHANNEL" != "beta" ]]; then
            cargo clippy -- -D warnings
        fi
        cd ..
    fi
done

