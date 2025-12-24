from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

Format = Literal["csv", "npy", "hdf5"]

def save_sequence(
    path: str | Path,
    seq: Sequence[int],
    *,
    format: Format | None = None,
    dataset: str = "fib",
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save an integer sequence to disk.

    Parameters
    ----------
    path
        Output file path.
    seq
        Sequence of integers to save.
    format
        File format: "csv", "npy", "hdf5". If None, infer from suffix.
    dataset
        HDF5 dataset name (only used when format="hdf5").
    metadata
        Optional metadata saved into:
        - CSV: header comment lines "# key: value"
        - NPY: ignored (NPY stores only array by default)
        - HDF5: file attributes
    overwrite
        Whether to allow overwriting an existing file.

    Returns
    -------
    pathlib.Path
        Resolved path of the saved file.

    Raises
    ------
    ValueError
        If format cannot be inferred or is unsupported.
    TypeError
        If numpy/h5py is required but not installed.
    """
    p = Path(path).expanduser().resolve()
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")
    fmt = _infer_format(p, format)

    if fmt == "csv":
        _save_csv(p, seq, metadata=metadata)
        return p

    if fmt == "npy":
        _require_numpy()
        _save_npy(p, seq)
        return p

    if fmt == "hdf5":
        _save_hdf5(p, seq, dataset=dataset, metadata=metadata)
        return p

    raise ValueError(f"Unsupported format: {fmt!r}")

def load_sequence(
    path: str | Path,
    *,
    format: Format | None = None,
    dataset: str = "fib",
) -> list[int]:
    """
    Load an integer sequence from disk.

    Parameters
    ----------
    path
        Input file path.
    format
        File format: "csv", "npy", "hdf5". If None, infer from suffix.
    dataset
        HDF5 dataset name (only used when format="hdf5").

    Returns
    -------
    list[int]
        Loaded integer sequence.

    Raises
    ------
    ValueError
        If format cannot be inferred or is unsupported.
    TypeError
        If numpy/h5py is required but not installed.
    """
    p = Path(path).expanduser().resolve()
    fmt = _infer_format(p, format)

    if fmt == "csv":
        return _load_csv(p)

    if fmt == "npy":
        _require_numpy()
        return _load_npy(p)

    if fmt == "hdf5":
        return _load_hdf5(p, dataset=dataset)

    raise ValueError(f"Unsupported format: {fmt!r}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _infer_format(path: Path, format: Format | None) -> Format:
    """Infer format from suffix if not provided."""
    if format is not None:
        return format

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".npy":
        return "npy"
    if suffix in {".h5", ".hdf5"}:
        return "hdf5"

    raise ValueError(
        f"Cannot infer format from suffix {suffix!r}. "
        "Please pass format='csv'|'npy'|'hdf5'."
    )


def _require_numpy() -> None:
    if np is None:
        raise TypeError("NumPy is required for .npy I/O, but it is not installed.")


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
def _save_csv(path: Path, seq: Sequence[int], *, metadata: dict[str, Any] | None) -> None:
    """
    Save sequence as CSV (one integer per row).

    Notes
    -----
    We write optional metadata as comment lines starting with "# ".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        writer = csv.writer(f)
        for x in seq:
            writer.writerow([x])


def _load_csv(path: Path) -> list[int]:
    """Load CSV (ignores comment lines starting with '#')."""
    out: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # A CSV row could be like "123" or "123,..." but we only use first column.
            first = line.split(",", 1)[0].strip()
            out.append(int(first))
    return out


# -----------------------------------------------------------------------------
# NPY
# -----------------------------------------------------------------------------
def _save_npy(path: Path, seq: Sequence[int]) -> None:
    """
    Save sequence to .npy.

    Notes
    -----
    We default to dtype=object to avoid overflow for large Fibonacci integers.
    """
    assert np is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(seq, dtype=np.object_)
    np.save(path, arr, allow_pickle=True)


def _load_npy(path: Path) -> list[int]:
    """Load .npy -> list[int]."""
    assert np is not None
    arr = np.load(path, allow_pickle=True)
    # Convert to Python ints explicitly
    return [int(x) for x in arr.tolist()]


# -----------------------------------------------------------------------------
# HDF5
# -----------------------------------------------------------------------------
def _save_hdf5(
    path: Path,
    seq: Sequence[int],
    *,
    dataset: str,
    metadata: dict[str, Any] | None,
) -> None:
    """
    Save sequence to HDF5.

    Notes
    -----
    To avoid int64 overflow, we store numbers as UTF-8 strings in a 1D dataset.
    """
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise TypeError("h5py is required for HDF5 I/O, but it is not installed.") from e

    path.parent.mkdir(parents=True, exist_ok=True)

    # store as strings to support arbitrarily large ints
    data = [str(x) for x in seq]

    with h5py.File(path, "w") as f:
        dt = h5py.string_dtype(encoding="utf-8")
        dset = f.create_dataset(dataset, data=data, dtype=dt)

        # Store metadata as file attributes (not dataset attrs, either is fine)
        if metadata:
            for k, v in metadata.items():
                # h5py attrs prefer scalar strings/numbers; cast complex objects to str
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = str(v)

        # helpful minimal attrs
        dset.attrs["kind"] = "integer_sequence"
        dset.attrs["encoding"] = "utf-8"


def _load_hdf5(path: Path, *, dataset: str) -> list[int]:
    """Load HDF5 sequence (stored as strings) -> list[int]."""
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise TypeError("h5py is required for HDF5 I/O, but it is not installed.") from e

    with h5py.File(path, "r") as f:
        if dataset not in f:
            raise ValueError(f"Dataset {dataset!r} not found in file: {path}")
        dset = f[dataset]
        raw = dset[()]  # could be numpy-like array of bytes/str

    out: list[int] = []
    for x in raw:
        if isinstance(x, bytes):
            out.append(int(x.decode("utf-8")))
        else:
            out.append(int(x))
    return out
