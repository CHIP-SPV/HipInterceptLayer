#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <chipStar binary> <hipamd binary> <binary args...>"
    exit 1
fi

CHIPSTAR_BINARY="$1"
HIPAMD_BINARY="$2"
shift 2  # Remove first two arguments, leaving only the binary args

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run with ChipStar
echo "Running with ChipStar..."
export BINARY="$CHIPSTAR_BINARY"
LD_PRELOAD="$SCRIPT_DIR/build/libHIPIntercept.so /space/pvelesko/install/HIP/chipStar/test/lib/libCHIP.so" $BINARY "$@"

echo -e "\nRunning with HIP-AMD..."
export BINARY="$HIPAMD_BINARY"
LD_PRELOAD="$SCRIPT_DIR/build/libHIPIntercept.so /opt/rocm-6.2.4/lib/libamdhip64.so.6.2.60204" $BINARY "$@" 