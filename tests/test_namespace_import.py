from hm.egt.heiz import __version__


def test_namespace_import() -> None:
    assert __version__ == "0.1.0"
