import asyncio
import atexit
import logging
import time
from typing import Optional

from rich.console import Console as RichConsole
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def format_bytes(count: float) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{int(count)} B"


class AssetTransferProgress:
    """
    Terminal-side view of what a connected client is pulling from the asset server.

    Both clients fetch every mesh, texture and video over HTTP one file at a time
    after SCENE_INIT arrives, and that transfer is the whole wait between "Pipeline
    complete" and a scene actually appearing -- hundreds of files and hundreds of
    megabytes, all of it squeezed through the SSH tunnel `frame.sh remote` sets up.
    Until now the only place that was visible was the client's own console (Unity's
    Editor log), so the terminal running the server sat silent for minutes with no
    way to tell a slow download from a hung one.

    Two rendering modes, chosen from the pipeline's log_mode:

      live  — a Rich progress bar on stdout. This is the default ("panel") mode, and
              the one that matters: panel mode installs no console handler at all
              once the pipeline's own Live display is torn down (see
              PipelineConfiguration._configure_logging), so log lines alone would be
              invisible in exactly the mode people run.
      lines — one log line per delivered asset. plain/verbose already put every log
              record on the terminal, and a second Live region would fight with
              them. Also used whenever stdout is not a terminal (piped, nohup),
              where a Live display renders nothing useful.

    Every event is logged either way, so the run's log file always has the full
    per-asset record regardless of which mode was used to draw it.
    """

    # How long the display sticks around with nothing in flight before it gives up
    # and reports what did arrive. Clients skip assets legitimately (a billboard
    # video is only fetched if its object ends up on screen), so "expected" is an
    # upper bound and waiting for it to be reached exactly would hang the bar
    # forever on a perfectly healthy run.
    IDLE_TIMEOUT = 20.0

    def __init__(self, log: logging.Logger, log_mode: str = "panel") -> None:
        self._log = log
        self._live_mode = log_mode not in ("plain", "verbose")
        self._console = RichConsole()

        self._progress: Optional[Progress] = None
        self._task = None
        self._watchdog: Optional[asyncio.Future] = None

        self._expected: list[str] = []
        self._sizes: dict[str, int] = {}      # asset key -> size on disk
        self._received: dict[str, int] = {}   # asset key -> bytes written to clients
        self._complete: set[str] = set()
        self._in_flight: dict[str, float] = {}

        self._bytes = 0
        self._started_at = 0.0
        self._last_activity = 0.0
        self._active = False

        # Progress.start() hides the terminal cursor; a Ctrl+C out of the server
        # unwinds asyncio.run() without ever reaching stop(), which would leave the
        # shell cursorless.
        atexit.register(self.stop)

    # -- lifecycle ---------------------------------------------------------

    def begin(self, expected: list[str]) -> None:
        """Start tracking a fresh scene's transfer. Safe to call repeatedly -- a
        reconnecting client is re-sent the cached snapshot and re-downloads
        everything, which is a new transfer, not a continuation of the last one."""
        self.stop()

        self._expected = list(expected)
        self._sizes = {}
        self._received = {}
        self._complete = set()
        self._in_flight = {}
        self._bytes = 0
        self._started_at = time.monotonic()
        self._last_activity = self._started_at
        self._active = True

        total = len(self._expected)
        self._log.info(f"Client downloading {total} assets…")

        if self._use_live():
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[detail]}"),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
            )
            self._progress.start()
            self._task = self._progress.add_task(
                "Downloading assets",
                total=total or None,
                detail="",
            )

        self._watchdog = self._spawn(self._watch())

    def stop(self) -> None:
        """Tear the display down without reporting a summary (the pipeline is about
        to run again, or the server is shutting down)."""
        self._active = False
        if self._watchdog is not None:
            try:
                self._watchdog.cancel()
            except RuntimeError:
                pass  # loop already closed (atexit)
            self._watchdog = None
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None

    # -- events ------------------------------------------------------------

    def note(self, message: str) -> None:
        """Transient status for work that happens before a byte is sent -- notably
        exporting an asset that the archive holds but has never written out."""
        self._log.info(message)
        self._set_detail(message)

    def asset_requested(self, name: str, size: int) -> None:
        if not self._active:
            # A client can request assets from a previous scene, or before the
            # snapshot went out at all. Track it anyway so the numbers add up.
            self.begin(self._expected)
        self._sizes[name] = size
        self._in_flight[name] = time.monotonic()
        self._last_activity = time.monotonic()
        self._set_detail(f"{format_bytes(self._bytes)} · {name}")

    def asset_delivered(self, name: str, sent: int, duration: float, status: int) -> None:
        started = self._in_flight.pop(name, None)
        self._last_activity = time.monotonic()

        if status >= 400:
            self._log.warning(f"[!] Asset request failed ({status}): {name}")
            self._set_detail(f"{format_bytes(self._bytes)} · {name} failed ({status})")
            return

        self._bytes += sent
        self._received[name] = self._received.get(name, 0) + sent

        # A video asset is streamed by the client's player in byte ranges, so one
        # file arrives as several partial responses; only the last one that carries
        # it past its own size counts as the file being through.
        size = self._sizes.get(name, 0)
        newly_complete = (
            name not in self._complete
            and size > 0
            and self._received[name] >= size
        )
        if newly_complete:
            self._complete.add(name)
            if self._progress is not None and self._task is not None:
                self._progress.update(self._task, completed=len(self._complete))

        rate = f" ({format_bytes(sent / duration)}/s)" if duration > 0.05 else ""
        self._log.info(
            f"[{len(self._complete):>4}/{len(self._expected)}] {name}  "
            f"{format_bytes(sent)} in {duration:.2f}s{rate}"
        )
        self._set_detail(f"{format_bytes(self._bytes)} · {name}")

    # -- internals ---------------------------------------------------------

    def _use_live(self) -> bool:
        return self._live_mode and self._console.is_terminal

    def _set_detail(self, detail: str) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, detail=detail)

    def _spawn(self, coro) -> Optional[asyncio.Future]:
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            # No running loop (unit tests, or a caller outside the server's loop):
            # the display still works, it just never self-terminates.
            coro.close()
            return None

    async def _watch(self) -> None:
        while self._active:
            await asyncio.sleep(1.0)
            if not self._in_flight:
                if self._expected and len(self._complete) >= len(self._expected):
                    self._finish("complete")
                    return
                if time.monotonic() - self._last_activity > self.IDLE_TIMEOUT:
                    self._finish("idle")
                    return

    def _finish(self, reason: str) -> None:
        elapsed = max(time.monotonic() - self._started_at, 1e-6)
        summary = (
            f"Assets: {len(self._complete)}/{len(self._expected)} delivered · "
            f"{format_bytes(self._bytes)} in {elapsed:.1f}s "
            f"({format_bytes(self._bytes / elapsed)}/s)"
        )
        if reason == "idle" and len(self._complete) < len(self._expected):
            # Not an error: the client only fetches what it ends up rendering.
            summary += " · client idle"

        live = self._progress is not None
        self.stop()
        self._log.info(summary)
        if live:
            self._console.print(f"[green]✓[/green] {summary}", highlight=False)
