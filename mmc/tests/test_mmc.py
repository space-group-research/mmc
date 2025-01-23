"""
Unit and regression test for the mmc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mmc
from mmc.simulation import Simulation


def test_mmc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mmc" in sys.modules
