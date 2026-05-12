"""
pipeline_monitor.py — Nestable CPU/GPU monitor for AI pipelines.

Tracks peak & average utilization + memory for each stage, with timing.
Stages are nestable — an outer "pipeline" stage wraps inner "step" stages.

Dependencies:
    pip install psutil pynvml

Usage:
    from pipeline_monitor import PipelineMonitor

    mon = PipelineMonitor(interval=0.25)

    with mon.stage("full pipeline"):
        with mon.stage("load data"):
            load_data()
        with mon.stage("embed"):
            embed()
        with mon.stage("generate"):
            generate()

    mon.print_summary()
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    cpu_pct: float
    ram_used_gb: float
    gpu_pct: Optional[float]
    gpu_mem_used_gb: Optional[float]


@dataclass
class StageStats:
    name: str
    depth: int                        # nesting level (0 = top)
    elapsed: float = 0.0
    samples: list = field(default_factory=list)
    children: list = field(default_factory=list)   # child StageStats

    # ── computed properties ──────────────────────────────────────────────────

    def _vals(self, attr):
        return [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]

    @property
    def cpu_avg(self):      return _avg(self._vals("cpu_pct"))
    @property
    def cpu_peak(self):     return _peak(self._vals("cpu_pct"))
    @property
    def ram_avg(self):      return _avg(self._vals("ram_used_gb"))
    @property
    def ram_peak(self):     return _peak(self._vals("ram_used_gb"))
    @property
    def gpu_avg(self):      return _avg(self._vals("gpu_pct"))
    @property
    def gpu_peak(self):     return _peak(self._vals("gpu_pct"))
    @property
    def gpu_mem_avg(self):  return _avg(self._vals("gpu_mem_used_gb"))
    @property
    def gpu_mem_peak(self): return _peak(self._vals("gpu_mem_used_gb"))


def _avg(vals):
    return sum(vals) / len(vals) if vals else None

def _peak(vals):
    return max(vals) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────

class PipelineMonitor:
    """
    Background-polling CPU/GPU monitor with nestable named stages.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds (default 0.25 s).
    gpu_index : int
        Which GPU to monitor (default 0).
    """

    def __init__(self, interval: float = 0.25, gpu_index: int = 0):
        self.interval = interval
        self._gpu_handle = None
        if _GPU_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                pass

        self._lock = threading.Lock()
        self._stack: list[StageStats] = []   # active stage stack (innermost last)
        self._roots: list[StageStats] = []   # finished top-level stages
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── public API ───────────────────────────────────────────────────────────

    @contextmanager
    def stage(self, name: str):
        """Context manager for a named stage. Nestable."""
        self._ensure_running()

        with self._lock:
            depth = len(self._stack)
            stat = StageStats(name=name, depth=depth)
            if self._stack:
                self._stack[-1].children.append(stat)
            else:
                self._roots.append(stat)
            self._stack.append(stat)

        t0 = time.perf_counter()
        try:
            yield stat
        finally:
            stat.elapsed = time.perf_counter() - t0
            with self._lock:
                self._stack.pop()
            # Stop polling thread when the outermost stage finishes
            with self._lock:
                if not self._stack:
                    self._stop_event.set()

    def print_summary(self):
        """Print a formatted summary of all recorded stages."""
        has_gpu = self._gpu_handle is not None
        _print_summary(self._roots, has_gpu)

    # ── internals ────────────────────────────────────────────────────────────

    def _ensure_running(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            sample = self._take_sample()
            with self._lock:
                # Samples are added to ALL active stages (outer gets everything,
                # inner gets only what happens while it's active)
                for stat in self._stack:
                    stat.samples.append(sample)
            time.sleep(self.interval)

    def _take_sample(self) -> Sample:
        cpu_pct    = psutil.cpu_percent()
        ram        = psutil.virtual_memory()
        ram_used_gb = ram.used / 1024 ** 3

        gpu_pct = gpu_mem_gb = None
        if self._gpu_handle:
            try:
                util       = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem        = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_pct    = float(util.gpu)
                gpu_mem_gb = mem.used / 1024 ** 3
            except Exception:
                pass

        return Sample(cpu_pct, ram_used_gb, gpu_pct, gpu_mem_gb)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_WHITE  = "\033[97m"


def _fmt_time(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.1f}s"


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:5.1f}%" if v is not None else "   N/A"


def _fmt_gb(v: Optional[float]) -> str:
    return f"{v:5.2f}GB" if v is not None else "    N/A"


def _print_summary(roots: list, has_gpu: bool):
    # Column widths (content only, no ANSI codes in padding)
    STAGE_W  = 32
    CPU_W    = 28   # "  avg:XX.X% / peak:XX.X%"
    RAM_W    = 26   # "  X.XXgb / X.XXgb"
    GPU_W    = 28 if has_gpu else 0
    VRAM_W   = 26 if has_gpu else 0
    TIME_W   = 12
    INNER    = STAGE_W + CPU_W + RAM_W + GPU_W + VRAM_W + TIME_W
    W        = INNER + 2

    def rule(l, r):
        print(_CYAN + l + "─" * INNER + r + _RESET)

    def divider():
        print(_DIM + _CYAN + "│" + "·" * INNER + "│" + _RESET)

    def row(cells: list[tuple[str, int]], bold=False, dim=False):
        """Each cell is (content_with_ansi, display_width)."""
        prefix = _BOLD if bold else (_DIM if dim else "")
        inner = "".join(
            content + " " * max(0, width - _visible_len(content))
            for content, width in cells
        )
        print(_CYAN + "│" + _RESET + prefix + inner + _RESET + _CYAN + "│" + _RESET)

    def _visible_len(s: str) -> int:
        """Length of string excluding ANSI escape codes."""
        import re
        return len(re.sub(r'\x1b\[[0-9;]*m', '', s))

    def _center(text: str, width: int) -> str:
        """Center text accounting for invisible ANSI codes."""
        visible = _visible_len(text)
        total_pad = max(0, width - visible)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        return " " * left_pad + text + " " * right_pad

    def _ljust(text: str, width: int) -> str:
        visible = _visible_len(text)
        return text + " " * max(0, width - visible)

    # ── Top border ──────────────────────────────────────────────────────────
    rule("┌", "┐")

    # Title
    title = f"  {_WHITE}{_BOLD}PIPELINE SUMMARY{_RESET}"
    row([(title, INNER)])

    rule("├", "┤")

    # Header row
    row([
        (_DIM + f" {'STAGE':<{STAGE_W - 1}}", STAGE_W),
        (_DIM + _center("CPU avg / peak", CPU_W),  CPU_W),
        (_DIM + _center("RAM avg / peak", RAM_W),  RAM_W),
        *([
            (_DIM + _center("GPU avg / peak",  GPU_W),  GPU_W),
            (_DIM + _center("VRAM avg / peak", VRAM_W), VRAM_W),
        ] if has_gpu else []),
        (_DIM + _center("time", TIME_W), TIME_W),
    ])

    rule("├", "┤")

    # ── Stages ──────────────────────────────────────────────────────────────
    def print_stage(stat: StageStats):
        indent = "  " * stat.depth
        prefix = "▸ " if stat.children else "  "
        name   = indent + prefix + stat.name
        color  = (_YELLOW + _BOLD) if stat.depth == 0 else ""

        col_stage = color + name + _RESET
        col_cpu   = f"  {_fmt_pct(stat.cpu_avg)} / {_fmt_pct(stat.cpu_peak)}"
        col_ram   = f"  {_fmt_gb(stat.ram_avg)} / {_fmt_gb(stat.ram_peak)}"
        col_gpu   = f"  {_fmt_pct(stat.gpu_avg)} / {_fmt_pct(stat.gpu_peak)}" if has_gpu else None
        col_vram  = f"  {_fmt_gb(stat.gpu_mem_avg)} / {_fmt_gb(stat.gpu_mem_peak)}" if has_gpu else None
        col_time  = f"  {_fmt_time(stat.elapsed):>{TIME_W - 2}}"

        cells = [
            (col_stage, STAGE_W),
            (col_cpu,   CPU_W),
            (col_ram,   RAM_W),
            *([
                (col_gpu,  GPU_W),
                (col_vram, VRAM_W),
            ] if has_gpu else []),
            (col_time, TIME_W),
        ]
        row(cells)

        for child in stat.children:
            print_stage(child)

    for i, root in enumerate(roots):
        print_stage(root)
        if i < len(roots) - 1:
            divider()

    rule("└", "┘")
    print()