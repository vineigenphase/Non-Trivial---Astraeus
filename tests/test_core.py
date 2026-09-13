"""Core invariants, one pytest per selftest check so failures are attributable."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astraeus import selftest  # noqa: E402


@pytest.mark.parametrize("name,fn", selftest.CHECKS, ids=[n for n, _ in selftest.CHECKS])
def test_invariant(name, fn):
    fn()
