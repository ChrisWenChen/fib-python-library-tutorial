# benchmarks/test_benchmark_fib.py
from __future__ import annotations

import pytest

from fib_python_library_tutorial.fib import fib


# -----------------------------
# Group 1: same-n comparison
# -----------------------------
SAME_N = 2_000

@pytest.mark.benchmark(group="fib-seq-same-n")
def test_bench_iterative_same_n(benchmark) -> None:
    benchmark(lambda: fib(SAME_N, method="iterative", start=0, output="list"))

@pytest.mark.benchmark(group="fib-seq-same-n")
def test_bench_recursive_same_n(benchmark) -> None:
    benchmark(lambda: fib(SAME_N, method="recursive", start=0, output="list"))

@pytest.mark.benchmark(group="fib-seq-same-n")
def test_bench_matrix_same_n(benchmark) -> None:
    benchmark(lambda: fib(SAME_N, method="matrix_exponentiation", start=0, output="list"))

@pytest.mark.benchmark(group="fib-seq-same-n")
def test_bench_iterative_same_n_numpy_object(benchmark) -> None:
    np = pytest.importorskip("numpy")
    benchmark(lambda: fib(SAME_N, method="iterative", start=0, output="numpy", dtype=np.object_))


# -----------------------------
# Group 2: large-n (only fast method)
# -----------------------------
@pytest.mark.benchmark(group="fib-seq-large")
def test_bench_iterative_10k(benchmark) -> None:
    benchmark(lambda: fib(10_000, method="iterative", start=0, output="list"))


# -----------------------------
# Group 3: binet (limited range)
# -----------------------------
@pytest.mark.benchmark(group="fib-binet")
def test_bench_binet_70(benchmark) -> None:
    benchmark(lambda: fib(70, method="binet_formula", start=0, output="list"))
