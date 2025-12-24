from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .fib import fib, Method
from .io import load_sequence, save_sequence


def _non_negative_int(value: str) -> int:
    """Argparse type: non-negative integer."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if n < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return n


def _cmd_gen(args: argparse.Namespace) -> int:
    seq = fib(
        args.n,
        method=args.method,
        start=args.start,
        output="list",
    )

    if args.output is not None:
        save_sequence(
            path=args.output,
            seq=seq,
            format=args.format,
            dataset=args.dataset,
            overwrite=args.overwrite,
            metadata={
                "method": args.method,
                "start": args.start,
                "n": args.n,
            },
        )
        return 0

    print(seq)
    return 0


def _cmd_load(args: argparse.Namespace) -> int:
    seq = load_sequence(
        path=args.path,
        format=args.format,
        dataset=args.dataset,
    )
    print(seq)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fib",
        description="Fibonacci sequence CLI (CSV / NPY / HDF5).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # fib gen
    # ------------------------------------------------------------------
    p_gen = sub.add_parser("gen", help="Generate a Fibonacci sequence.")
    p_gen.add_argument("n", type=_non_negative_int, help="Length of the sequence.")
    p_gen.add_argument(
        "--start",
        type=_non_negative_int,
        default=0,
        help="Starting index.",
    )
    p_gen.add_argument(
        "--method",
        type=str,
        default="iterative",
        choices=["recursive", "iterative", "matrix_exponentiation", "binet_formula"],
        help="Fibonacci algorithm.",
    )
    p_gen.add_argument(
        "--output",
        "--save",
        dest="output",
        type=str,
        default=None,
        help="Save sequence to file (.csv/.npy/.h5/.hdf5).",
    )
    p_gen.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["csv", "npy", "hdf5"],
        help="Force output file format.",
    )
    p_gen.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files.",
    )
    p_gen.add_argument(
        "--dataset",
        type=str,
        default="fib",
        help="HDF5 dataset name.",
    )
    p_gen.set_defaults(func=_cmd_gen)

    # ------------------------------------------------------------------
    # fib load
    # ------------------------------------------------------------------
    p_load = sub.add_parser("load", help="Load a sequence from file.")
    p_load.add_argument("path", type=str, help="Input file path.")
    p_load.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["csv", "npy", "hdf5"],
        help="Force input file format.",
    )
    p_load.add_argument(
        "--dataset",
        type=str,
        default="fib",
        help="HDF5 dataset name.",
    )
    p_load.set_defaults(func=_cmd_load)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
