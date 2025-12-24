# tests/test_fib.py
from __future__ import annotations

import math
import pytest

from fib_python_library_tutorial.fib import fib



@pytest.mark.parametrize(
    "n,start,expected",
    [
        (0, 0, []),
        (1, 0, [0]),
        (2, 0, [0, 1]),
        (6, 0, [0, 1, 1, 2, 3, 5]),
        (6, 1, [1, 1, 2, 3, 5, 8]),
        (6, 2, [1, 2, 3, 5, 8, 13]),
        (1, 10, [55]),  # F(10)=55
        (5, 10, [55, 89, 144, 233, 377]),
    ],
)
@pytest.mark.parametrize("method", ["iterative", "recursive", "matrix_exponentiation"])
def test_fib_sequence_exact_methods(n: int, start: int, expected: list[int], method: str) -> None:
    """Exact methods should match expected sequences."""
    assert fib(n, method=method, start=start, output="list") == expected


@pytest.mark.parametrize(
    "n,start",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (10, 0),
        (10, 5),
        (20, 0),
        (20, 7),
    ],
)
def test_methods_agree_on_small_ranges(n: int, start: int) -> None:
    """
    For modest n/start, iterative/recursive/matrix should agree exactly.
    (recursive uses memoization in your implementation)
    """
    a = fib(n, method="iterative", start=start)
    b = fib(n, method="recursive", start=start)
    c = fib(n, method="matrix_exponentiation", start=start)
    assert a == b == c


def test_binet_matches_small_range() -> None:
    """Binet is approximate but should match for small indices."""
    seq = fib(20, method="iterative", start=0)
    approx = fib(20, method="binet_formula", start=0)
    assert approx == seq


def test_binet_raises_for_large_k() -> None:
    """Your _fib_binet_single restricts k>70, so ranges crossing it should fail."""
    # start=71 already invalid
    with pytest.raises(ValueError):
        fib(1, method="binet_formula", start=71)

    # start=60 length=20 crosses 70 -> should fail at some i
    with pytest.raises(ValueError):
        fib(20, method="binet_formula", start=60)


def test_invalid_n_type() -> None:
    with pytest.raises(TypeError):
        fib("10")  # type: ignore[arg-type]


def test_invalid_n_negative() -> None:
    with pytest.raises(ValueError):
        fib(-1)


def test_invalid_start_type() -> None:
    with pytest.raises(TypeError):
        fib(5, start="0")  # type: ignore[arg-type]


def test_invalid_start_negative() -> None:
    with pytest.raises(ValueError):
        fib(5, start=-2)


def test_invalid_method() -> None:
    with pytest.raises(ValueError):
        fib(5, method="nope")  # type: ignore[arg-type]


def test_invalid_output() -> None:
    with pytest.raises(ValueError):
        fib(5, output="nope")  # type: ignore[arg-type]


def test_numpy_output_when_available() -> None:
    np = pytest.importorskip("numpy")
    arr = fib(6, output="numpy")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.dtype("O")  # 默认 object，避免溢出
    assert arr.tolist() == [0, 1, 1, 2, 3, 5]


def test_numpy_dtype_int64_overflow_note() -> None:
    """
    This test does not enforce overflow prevention; it only verifies that the dtype parameter works.
    Note: F(93) exceeds int64, so overflow is expected behavior (user assumes the risk).
    """
    np = pytest.importorskip("numpy")
    arr = fib(10, output="numpy", dtype=np.int64)
    assert arr.dtype == np.int64
    assert arr.tolist() == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]