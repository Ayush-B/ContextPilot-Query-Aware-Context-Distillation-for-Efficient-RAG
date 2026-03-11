def test_import_contextpilot() -> None:
    import contextpilot

    assert contextpilot is not None


def test_import_settings() -> None:
    from contextpilot.config.settings import get_settings

    settings = get_settings()

    assert settings is not None