from contextpilot.config.settings import get_settings


def test_settings_load() -> None:
    settings = get_settings()

    assert settings.project_name is not None
    assert settings.openai_model is not None