from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal, Sequence, overload

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

Method = Literal["recursive", "iterative", "matrix_exponentiation", "binet_formula"]
Output = Literal["list", "numpy"]


def fib(
    n: int,
    method: Method = "iterative",
    *,
    start: int = 0,
    output: Output = "list",
    dtype: "object | None" = None,
) -> "list[int] | np.ndarray":
    """
    Generate a Fibonacci sequence of length ``n``.

    By default this returns ``[F(0), F(1), ..., F(n-1)]``.

    Parameters
    ----------
    n
        Length of the sequence to generate. Must be a non-negative integer.
    method
        Algorithm used to generate the sequence.

        - ``"recursive"``: recursion with memoization (teaching; still slower than DP).
        - ``"iterative"``: dynamic programming / iterative (recommended).
        - ``"matrix_exponentiation"``: compute each term using fast 2x2 matrix exponentiation.
        - ``"binet_formula"``: use Binet's closed-form (floating point; may be inaccurate).
    start
        Starting index of the sequence.

        - ``start=0`` yields: ``[F(0), F(1), ..., F(n-1)]``.
        - ``start=1`` yields: ``[F(1), F(2), ..., F(n)]``.
    output
        Output container type.

        - ``"list"``: Python list of ints.
        - ``"numpy"``: NumPy array (requires NumPy installed).
    dtype
        Only used when ``output="numpy"``.
        If None, defaults to ``object`` to avoid overflow for large Fibonacci numbers.

        For small ranges you may pass an integer dtype (e.g. ``np.int64``),
        but be aware Fibonacci grows fast and will overflow fixed-width integers.

    Returns
    -------
    list[int] or numpy.ndarray
        Fibonacci sequence of requested length and start index.

    Raises
    ------
    TypeError
        If ``n`` is not int; or if ``output="numpy"`` but NumPy is not installed.
    ValueError
        If ``n`` is negative, ``start`` is negative, method/output unsupported,
        or Binet method is requested for ranges that are too large for safe rounding.

    Notes
    -----
    - DP (Dynamic Programming) here simply means: store subproblem results, construct iteratively from small to large, and avoid massive redundant computations in recursion.
    - For generating a complete sequence, DP/iteration is the most natural and efficient approach: O(n) time, O(1) extra space.
    - Matrix exponentiation is suitable for computing a single F(n) in O(log n) time, but generating a full sequence this way takes O(n log n).
    - Binet's formula is a floating-point approximation: for large n, rounding errors may produce incorrect integers; a conservative limit is applied here.

    Examples
    --------
    >>> fib(6)
    [0, 1, 1, 2, 3, 5]
    >>> fib(6, start=1)
    [1, 1, 2, 3, 5, 8]
    >>> fib(10, method="matrix_exponentiation")[:5]
    [0, 1, 1, 2, 3]
    >>> import numpy as np
    >>> fib(6, output="numpy", dtype=np.int64)
    array([0, 1, 1, 2, 3, 5])
    """
    _validate_n(n)
    _validate_start(start)
    seq: list[int]

    if n == 0:
        seq = []
    else:
        if method == "iterative":
            seq = _fib_seq_iterative(n, start=start)
        elif method == "recursive":
            seq = _fib_seq_recursive(n, start=start)
        elif method == "matrix_exponentiation":
            seq = _fib_seq_matrix(n, start=start)
        elif method == "binet_formula":
            seq = _fib_seq_binet(n, start=start)
        else:
            raise ValueError(f"Unsupported method: {method!r}")

    if output == "list":
        return seq

    if output == "numpy":
        if np is None:
            raise TypeError("NumPy is not installed, but output='numpy' was requested.")
        if dtype is None:
            dtype = np.object_  # avoid overflow by default
        return np.array(seq, dtype=dtype)

    raise ValueError(f"Unsupported output: {output!r}")

def _validate_n(n: int) -> None:
    """Validate sequence length."""
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if n < 0:
        raise ValueError("n must be non-negative")


def _validate_start(start: int) -> None:
    """Validate starting index."""
    if not isinstance(start, int):
        raise TypeError(f"start must be int, got {type(start).__name__}")
    if start < 0:
        raise ValueError("start must be non-negative")

# -----------------------------------------------------------------------------
# 1) Iterative / DP
# -----------------------------------------------------------------------------
def _fib_seq_iterative(n: int, *, start: int) -> list[int]:
    """
    Generate Fibonacci sequence using iterative DP.

    Parameters
    ----------
    n
        Number of terms to generate.
    start
        Starting index.

    Returns
    -------
    list[int]
        Sequence of length n.
    """
    # We want F(start) ... F(start+n-1).
    # If start is 0, we can build directly.
    # For general start, we can iterate to start+n-1 and slice.
    end = start + n - 1
    if end == 0:
        return [0]
    if end == 1:
        base = [0, 1]
        return base[start : start + n]

    a, b = 0, 1  # F(0), F(1)
    out = [0, 1]
    for _ in range(2, end + 1):
        a, b = b, a + b
        out.append(b)

    return out[start : start + n]

# -----------------------------------------------------------------------------
# 2) Recursive (with memoization for practicality; still teaching-oriented)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _fib_recursive_single(k: int) -> int:
    """
    Fibonacci single term via recursion (memoized).

    Parameters
    ----------
    k
        Index.

    Returns
    -------
    int
        F(k).
    """
    if k < 2:
        return k
    return _fib_recursive_single(k - 1) + _fib_recursive_single(k - 2)


def _fib_seq_recursive(n: int, *, start: int) -> list[int]:
    """
    Generate Fibonacci sequence using recursion (memoized).

    Parameters
    ----------
    n
        Number of terms to generate.
    start
        Starting index.

    Returns
    -------
    list[int]
        Sequence of length n.
    """
    return [_fib_recursive_single(i) for i in range(start, start + n)]

# -----------------------------------------------------------------------------
# 3) Matrix exponentiation (exact integer math)
# -----------------------------------------------------------------------------
def _mat_mul_2x2(
    X: tuple[int, int, int, int],
    Y: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Multiply two 2x2 matrices.

    Parameters
    ----------
    X, Y
        Matrices in row-major order: (a00, a01, a10, a11).

    Returns
    -------
    tuple[int, int, int, int]
        X @ Y in the same row-major order.
    """
    x00, x01, x10, x11 = X
    y00, y01, y10, y11 = Y
    return (
        x00 * y00 + x01 * y10,
        x00 * y01 + x01 * y11,
        x10 * y00 + x11 * y10,
        x10 * y01 + x11 * y11,
    )


def _mat_pow_2x2(A: tuple[int, int, int, int], e: int) -> tuple[int, int, int, int]:
    """
    Fast exponentiation for a 2x2 matrix.

    Parameters
    ----------
    A
        Base matrix.
    e
        Exponent >= 0.

    Returns
    -------
    tuple[int, int, int, int]
        A**e.
    """
    result = (1, 0, 0, 1)  # identity
    base = A
    n = e
    while n > 0:
        if n & 1:
            result = _mat_mul_2x2(result, base)
        base = _mat_mul_2x2(base, base)
        n >>= 1
    return result


def _fib_matrix_single(k: int) -> int:
    """
    Compute F(k) exactly using matrix exponentiation.

    Parameters
    ----------
    k
        Index.

    Returns
    -------
    int
        F(k).

    Notes
    -----
    Uses:

        [F(k+1)]   [1 1]^k [1]
        [F(k)  ] = [1 0]   [0]

    So F(k) is the (1,0) element of A^k.
    """
    if k < 2:
        return k
    A = (1, 1, 1, 0)
    Ak = _mat_pow_2x2(A, k)
    return Ak[2]


def _fib_seq_matrix(n: int, *, start: int) -> list[int]:
    """
    Generate Fibonacci sequence using matrix exponentiation with reuse.

    This computes A**start once, then advances by multiplying A each step.

    Parameters
    ----------
    n
        Number of terms to generate.
    start
        Starting index.

    Returns
    -------
    list[int]
        Sequence [F(start), F(start+1), ..., F(start+n-1)].

    Notes
    -----
    Complexity: O(log start + n). Exact integer arithmetic.
    """
    if n <= 0:
        return []
    if start < 0:
        raise ValueError("start must be non-negative")

    A = (1, 1, 1, 0)

    # M = A**start
    M = _mat_pow_2x2(A, start)

    out: list[int] = []
    for _ in range(n):
        # M's first column is [F(k+1), F(k)]^T, so F(k) == M[2]
        out.append(M[2])
        # advance: A^(k+1) = A^k @ A
        M = _mat_mul_2x2(M, A)

    return out

# -----------------------------------------------------------------------------
# 4) Binet formula (approximate)
# -----------------------------------------------------------------------------
def _fib_binet_single(k: int) -> int:
    """
    Approximate F(k) using Binet's formula and round to nearest integer.

    Parameters
    ----------
    k
        Index.

    Returns
    -------
    int
        Rounded approximation of F(k).

    Raises
    ------
    ValueError
        If k is too large for safe rounding in double precision.

    Notes
    -----
    Binet:

        F(k) = (phi^k - psi^k) / sqrt(5)

    For large k, floating-point rounding becomes unreliable.
    """
    # Very conservative bound for "likely safe" integer rounding on IEEE-754 double.
    # Many sources show correctness up to around 70~75 depending on implementation.
    if k > 70:
        raise ValueError(
            "binet_formula is not reliable for k > 70 with float precision. "
            "Use method='iterative' or 'matrix_exponentiation' instead."
        )
    sqrt5 = math.sqrt(5.0)
    phi = (1.0 + sqrt5) / 2.0
    psi = (1.0 - sqrt5) / 2.0
    val = (phi**k - psi**k) / sqrt5
    return int(round(val))


def _fib_seq_binet(n: int, *, start: int) -> list[int]:
    """
    Generate Fibonacci sequence using Binet's formula (approximate).

    Parameters
    ----------
    n
        Number of terms to generate.
    start
        Starting index.

    Returns
    -------
    list[int]
        Sequence of length n.
    """
    return [_fib_binet_single(i) for i in range(start, start + n)]