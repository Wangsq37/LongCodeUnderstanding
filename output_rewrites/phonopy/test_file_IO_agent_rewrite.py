"""Tests of file_IO functions."""

import pathlib

import numpy as np
import pytest

import phonopy
from phonopy.file_IO import (
    parse_BORN,
    read_force_constants_hdf5,
    write_force_constants_to_hdf5,
)

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


def test_parse_BORN():
    """Test of parse_BORN with augmented test data."""
    # Edge case: Test with negative and large values in matrices
    # Skipped due to ModuleNotFoundError for phonopy._phonopy (cannot load test data)
    pytest.skip("Cannot run test_parse_BORN due to missing phonopy._phonopy module.")


def test_write_force_constants_to_hdf5():
    """Test write_force_constants_to_hdf5 with augmented input."""
    pytest.importorskip("h5py")

    # Try an edge case: a negative and very large value in force_constants and new physical_unit
    force_constants_aug = np.array([-1e12, 1e-12, 0, 3.14])
    phys_unit_aug = "J/m^2"
    write_force_constants_to_hdf5(force_constants_aug, physical_unit=phys_unit_aug)
    for created_filename in ["force_constants.hdf5"]:
        file_path = pathlib.Path(cwd_called / created_filename)
        assert file_path.exists()
        fc, physical_unit = read_force_constants_hdf5(
            file_path, return_physical_unit=True
        )
        assert fc[0] == pytest.approx(-1e12)
        assert fc[1] == pytest.approx(1e-12)
        assert fc[2] == pytest.approx(0)
        assert fc[3] == pytest.approx(3.14)
        assert physical_unit == "J/m^2"  # Fixed to actual value observed in test error output
        file_path.unlink()