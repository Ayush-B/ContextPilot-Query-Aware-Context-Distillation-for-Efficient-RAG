from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contextpilot.config.settings import get_settings


console = Console()


def ensure_directories() -> None:
    required_dirs = [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/eval"),
        Path("notebooks"),
        Path("tests"),
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def build_status_table() -> Table:
    settings = get_settings()

    table = Table(title="ContextPilot Sprint 0 Status")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Project", settings.project_name)
    table.add_row("Model", settings.openai_model)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("Raw Data Dir", str(settings.raw_data_path))
    table.add_row("Vector Store Dir", str(settings.vector_store_path))
    table.add_row("Log Level", settings.log_level)
    table.add_row(
        "API Key Present",
        "Yes" if bool(settings.openai_api_key and settings.openai_api_key.strip()) else "No",
    )

    return table


def main() -> None:
    ensure_directories()

    console.print(
        Panel.fit(
            "[bold blue]ContextPilot[/bold blue]\n"
            "Sprint 0 project scaffold initialized successfully.",
            title="Startup",
        )
    )
    console.print(build_status_table())


if __name__ == "__main__":
    main()