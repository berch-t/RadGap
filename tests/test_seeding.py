"""Le seeding déterministe rend les tirages reproductibles."""

import numpy as np

from radgap.utils import set_determinism


def test_set_determinism_reproducible():
    set_determinism(123)
    a = np.random.rand(5)
    set_determinism(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_determinism_different_seeds_differ():
    set_determinism(1)
    a = np.random.rand(5)
    set_determinism(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)
