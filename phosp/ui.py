from __future__ import annotations
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

_STAGE_LABELS: dict[str, str] = {
    "stage1": "Stage 1 — Chemical Modification",
    "stage2": "Stage 2 — MD Preparation",
    "stage3": "Stage 3 — MD Simulation",
    "stage4": "Stage 4 — Analysis",
}


class PhospUI:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._live: Live | None = None
        self._start: float = 0.0

    def stage_start(self, name: str, description: str = "") -> None:
        label = _STAGE_LABELS.get(name, name)
        self._console.print(f"\n[bold cyan]▶  {label}[/]")
        self._start = time.monotonic()
        spinner = Spinner("dots", text=f"  {description or 'Running...'}")
        self._live = Live(spinner, console=self._console, refresh_per_second=10)
        self._live.start()

    def stage_complete(self, name: str, elapsed_s: float) -> None:
        self._stop_live()
        label = _STAGE_LABELS.get(name, name)
        mins, secs = divmod(int(elapsed_s), 60)
        elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        self._console.print(f"[green]✓  {label} complete[/]  ({elapsed_str})")

    def stage_error(self, name: str, exc: Exception) -> None:
        self._stop_live()
        label = _STAGE_LABELS.get(name, name)
        self._console.print(Panel(
            f"[bold]{type(exc).__name__}:[/] {exc}",
            title=f"[red]✗ {label} failed[/]",
            border_style="red",
        ))

    def plugin_start(self, plugin_name: str) -> None:
        if self._live:
            self._live.update(Spinner("dots", text=f"  Running plugin: {plugin_name}"))

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
