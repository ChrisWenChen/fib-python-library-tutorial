from __future__ import annotations

from pathlib import Path

import pytest

from fib_python_library_tutorial import cli


def test_cli_writes_csv(tmp_path: Path) -> None:
    """
    Test that CLI correctly writes a CSV file with Fibonacci sequence.
    Verifies file creation and contains expected numbers.
    """
    out = tmp_path / "seq.csv"

    rc = cli.main(["gen","6", "--method", "iterative", "--start", "0", "--output", str(out), "--format", "csv", "--overwrite"])
    assert rc == 0

    # Simple validation (doesn't require exact CSV format, follows io.py rules)
    text = out.read_text(encoding="utf-8").strip()
    assert "0" in text
    assert "1" in text
    assert "5" in text


def test_cli_auto_format_by_suffix_npy(tmp_path: Path) -> None:
    """
    Test that CLI automatically infers .npy format from file extension.
    Requires NumPy.
    """
    np = pytest.importorskip("numpy")
    out = tmp_path / "seq.npy"

    rc = cli.main(["gen", "10", "--output", str(out), "--overwrite"])
    assert rc == 0
    assert out.exists()

    arr = np.load(out, allow_pickle=True)
    assert arr[0] == 0
    assert arr[1] == 1
    assert arr[9] == 34


def test_cli_rejects_negative_n(tmp_path: Path) -> None:
    """
    Test that CLI rejects negative sequence length.
    """
    out = tmp_path / "seq.csv"
    with pytest.raises(SystemExit):
        # argparse may raise SystemExit; if main() raises ValueError,
        # change this to pytest.raises(ValueError)
        cli.main(["gen", "-1", "--output", str(out), "--overwrite"])


def test_cli_overwrite_guard(tmp_path: Path) -> None:
    """
    Test that CLI prevents overwriting existing files without --overwrite flag.
    """
    out = tmp_path / "seq.csv"
    out.write_text("already here", encoding="utf-8")

    # Without --overwrite, should raise error (check io.py for ValueError / FileExistsError)
    with pytest.raises(Exception):
        cli.main(["gen", "6", "--output", str(out)])
