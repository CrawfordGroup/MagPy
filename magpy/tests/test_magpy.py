"""
Unit and regression test for the magpy package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import magpy


def test_magpy_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "magpy" in sys.modules
