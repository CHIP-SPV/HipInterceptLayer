# HIP Intercept Layer

A library for intercepting and analyzing HIP API calls and kernel executions on AMD GPUs.

## Overview

The HIP Intercept Layer provides tools to:
- Intercept HIP API calls (memory operations, kernel launches)
- Track GPU memory allocations and changes
- Record kernel executions and their memory effects
- Compare execution traces between different runs

## Installation

1. Prerequisites:
   - CMake 3.10+
   - chipStar Implementation of HIP
   - C++17 compiler

2. Build:
   ```bash
   mkdir build && cd build
   cmake ..
   make -
   ```

3. Set environment variables:
   ```bash
   # Set location for trace files
   export HIP_TRACE_LOCATION=/path/to/trace/dir

   default: $HOME/HipInterceptLayerTraces
      ```

## Example Usage

```bash     
LD_PRELOAD="/space/pvelesko/install/HIP/HipInterceptLayer/lib/libHIPIntercept.so /opt/rocm-6.2.4/lib/libamdhip64.so.6.2.60204"  /space/pvelesko/openmm/build/TestHipATMForce "single"

cd $HIP_TRACE_LOCATION
/space/pvelesko/install/HIP/HipInterceptLayer/bin/HipInterceptCompare ./TestHipATMForce-0.trace ./TestHipATMForce-1.trace

Traces differ at:

Kernel #0 (clearTwoBuffers)
  Config differences: Grid dimensions differ, Block dimensions differ
  Arg 2: 76800 pre-exec diffs (first: idx 0: 0 vs -0.372549)

Kernel #1 (copyState)
  Arg 2: 27 pre-exec diffs (first: idx 3: 0 vs -0.372549), 27 post-exec diffs (first: idx 3: 0 vs -0.372549)

Kernel #2 (clearTwoBuffers)
  Config differences: Grid dimensions differ, Block dimensions differ
  Arg 2: 76800 pre-exec diffs (first: idx 0: 0 vs -0.372549)

Kernel #3 (computeParameters)
  Arg 2: 1 pre-exec diffs (first: idx 0: 0 vs -0.372549), 1 post-exec diffs (first: idx 0: 0 vs -0.372549)

....

Kernel #498 (integrateLangevinMiddlePart1)
  Arg 2: 81 pre-exec diffs (first: idx 0: -3.21502e+06 vs -0.281743), 81 post-exec diffs (first: idx 0: -3.215e+06 vs -0.281266)

Kernel #499 (integrateLangevinMiddlePart2)
  Arg 3: 81 pre-exec diffs (first: idx 0: -12885.9 vs -0.00114601), 81 post-exec diffs (first: idx 0: -12834.3 vs -0.00112499)
  Arg 2: 81 pre-exec diffs (first: idx 0: -12885.9 vs -0.00114601), 81 post-exec diffs (first: idx 0: -12834.3 vs -0.00112499)
  Arg 1: 81 pre-exec diffs (first: idx 0: -3.215e+06 vs -0.281266), 81 post-exec diffs (first: idx 0: -3.20217e+06 vs -0.281229)

Memory operation errors: Different number of events in traces

```

