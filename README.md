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
   make
   ```

3. Set environment variables:
   ```bash
   # Set location for trace files
   export HIP_TRACE_LOCATION=/path/to/trace/dir
   # default: $HOME/HipInterceptLayerTraces
   ```

## Command Line Options

The HipInterceptCompare tool supports the following commands:

1. Compare two traces:
   ```bash
   HipInterceptCompare <trace1> <trace2>
   ```
   Shows differences between two execution traces, including:
   - Kernel configuration differences (grid/block dimensions)
   - Argument value differences (pre and post execution)
   - Memory operation differences

2. Generate kernel reproducer:
   ```bash
   HipInterceptCompare <trace> --gen-repro <operation_number>
   ```
   Creates a standalone reproducer for a specific kernel execution, including:
   - Complete kernel source if available
   - Input data setup
   - Kernel launch configuration
   - Pre/post execution value comparison

3. Print detailed values:
   ```bash
   HipInterceptCompare <trace> --print-vals
   ```
   Shows detailed information for each operation in the trace:
   - Kernel configurations
   - Pre-execution argument values/checksums
   - Post-execution argument values/checksums

## Example Usage

```bash     
# Run your application with the intercept layer
LD_PRELOAD="/path/to/libHIPIntercept.so /path/to/libamdhip64.so" ./your_application

# Compare two trace files
HipInterceptCompare ./trace1.trace ./trace2.trace

# Generate a reproducer for operation #5
HipInterceptCompare ./trace1.trace --gen-repro 5
```

The reproducer output will show:
- PRE-EXECUTION VALUES: Initial state of all arguments
- POST-EXECUTION VALUES: Final state after kernel execution
- SUMMARY OF CHANGES: List of arguments that were modified by the kernel
