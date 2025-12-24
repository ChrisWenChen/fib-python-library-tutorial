## fib-python-library-tutorial

A small Python library and CLI for generating Fibonacci sequences, with multiple algorithms and I/O helpers (CSV/NPY/HDF5). It is designed for learning, benchmarking, and practical sequence export.

## Install

Local editable install:

```bash
pip install -e .
```

Or standard install:

```bash
pip install .
```

## Python API

```python
from fib_python_library_tutorial.fib import fib

# Default: iterative, start at F(0)
print(fib(6))
# [0, 1, 1, 2, 3, 5]

# Start from F(1)
print(fib(6, start=1))
# [1, 1, 2, 3, 5, 8]

# Choose an algorithm
print(fib(10, method="matrix_exponentiation")[:5])
# [0, 1, 1, 2, 3]
```

NumPy output:

```python
import numpy as np
from fib_python_library_tutorial.fib import fib

arr = fib(6, output="numpy", dtype=np.int64)
print(arr)
```

## CLI

Generate a sequence and save to CSV:

```bash
fib gen 6 --method iterative --start 0 --output seq.csv --format csv --overwrite
```

Auto-detect output format by suffix:

```bash
fib gen 10 --output seq.npy --overwrite
```

Load a saved sequence:

```bash
fib load seq.csv
```

## Algorithm Comparison

- iterative (DP): O(n) time, O(1) extra space; best overall for full sequences.
- recursive (memoized): O(n) time, O(n) cache; good for teaching, slower overhead.
- matrix_exponentiation: O(log start + n) for sequences with reuse; exact integer math.
- binet_formula: O(n) using floats; fast but approximate, limited to small indices.
