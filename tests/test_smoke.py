"""Tests triviaux M0 : le package s'importe et expose sa version."""

import radgap


def test_version():
    assert isinstance(radgap.__version__, str)
    assert radgap.__version__


def test_utils_importable():
    from radgap.utils import get_logger, set_determinism

    assert callable(get_logger)
    assert callable(set_determinism)
