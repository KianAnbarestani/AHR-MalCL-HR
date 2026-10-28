from importlib.util import find_spec


def test_revision_r1_package_is_importable_from_its_root():
    assert find_spec("revision_r1") is not None

