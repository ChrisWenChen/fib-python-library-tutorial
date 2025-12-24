from __future__ import annotations

import pytest

from fib_python_library_tutorial.io import load_sequence, save_sequence


def test_csv_roundtrip(tmp_path) -> None:
    p = tmp_path / "seq.csv"
    seq = [0, 1, 1, 2, 3, 5, 8]
    save_sequence(p, seq, metadata={"start": 0, "n": len(seq)})
    got = load_sequence(p)
    assert got == seq


def test_npy_roundtrip(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    p = tmp_path / "seq.npy"
    seq = [0, 1, 1, 2, 3, 5, 8, 13, 21]
    save_sequence(p, seq)
    got = load_sequence(p)
    assert got == seq


def test_hdf5_roundtrip(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "seq.h5"
    # 测试大整数：避免 int64 溢出
    seq = [0, 1, 1, 2, 3, 5, 8, 13, 21, int("9" * 50)]
    save_sequence(p, seq, dataset="fib", metadata={"note": "big int test"})
    got = load_sequence(p, dataset="fib")
    assert got == seq


def test_infer_format_error(tmp_path) -> None:
    p = tmp_path / "seq.unknown"
    with pytest.raises(ValueError):
        save_sequence(p, [1, 2, 3])


def test_missing_dataset(tmp_path) -> None:
    pytest.importorskip("h5py")
    p = tmp_path / "seq.h5"
    save_sequence(p, [1, 2, 3], format="hdf5", dataset="fib")
    with pytest.raises(ValueError):
        load_sequence(p, dataset="no_such_dataset")
