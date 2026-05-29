"""
pipeline_monitor.py — Nestable CPU/multi-GPU monitor for AI pipelines.

Tracks peak & average utilization + memory for each stage, with timing.
Stages are nestable — an outer "pipeline" stage wraps inner "step" stages.

Dependencies:
    pip install psutil pynvml

Usage:
    from pipeline_monitor import PipelineMonitor

    # Monitor all GPUs (default)
    mon = PipelineMonitor(interval=0.25)

    # Monitor specific GPUs by index
    mon = PipelineMonitor(interval=0.25, gpu_indices=[0, 1])

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
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
    _GPU_AVAILABLE = _GPU_COUNT > 0
except Exception:
    _GPU_AVAILABLE = False
    _GPU_COUNT = 0


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUSample:
    pct: float
    mem_used_gb: float


@dataclass
class Sample:
    cpu_pct: float
    ram_used_gb: float
    gpus: list[GPUSample]   # one entry per monitored GPU, in gpu_indices order


@dataclass
class GPUStats:
    index: int
    samples_pct: list = field(default_factory=list)
    samples_mem: list = field(default_factory=list)

    @property
    def avg_pct(self):      return _avg(self.samples_pct)
    @property
    def peak_pct(self):     return _peak(self.samples_pct)
    @property
    def avg_mem(self):      return _avg(self.samples_mem)
    @property
    def peak_mem(self):     return _peak(self.samples_mem)

    def append(self, gs: GPUSample):
        self.samples_pct.append(gs.pct)
        self.samples_mem.append(gs.mem_used_gb)


@dataclass
class StageStats:
    name: str
    depth: int                        # nesting level (0 = top)
    gpu_indices: list[int]            # which GPUs are tracked
    elapsed: float = 0.0
    samples: list = field(default_factory=list)
    children: list = field(default_factory=list)   # child StageStats
    # per-GPU accumulators, keyed by gpu index
    gpu_stats: dict = field(default_factory=dict)

    def __post_init__(self):
        self.gpu_stats = {i: GPUStats(index=i) for i in self.gpu_indices}

    def add_sample(self, sample: Sample):
        self.samples.append(sample)
        for i, gs in zip(self.gpu_indices, sample.gpus):
            self.gpu_stats[i].append(gs)

    # ── CPU / RAM computed properties ────────────────────────────────────────

    def _vals(self, attr):
        return [getattr(s, attr) for s in self.samples]

    @property
    def cpu_avg(self):   return _avg(self._vals("cpu_pct"))
    @property
    def cpu_peak(self):  return _peak(self._vals("cpu_pct"))
    @property
    def ram_avg(self):   return _avg(self._vals("ram_used_gb"))
    @property
    def ram_peak(self):  return _peak(self._vals("ram_used_gb"))


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
    gpu_indices : list[int] | None
        Which GPUs to monitor. Defaults to all available GPUs.
        Pass an empty list to disable GPU monitoring entirely.
    """

    def __init__(self, interval: float = 0.25, gpu_indices: Optional[list[int]] = None):
        self.interval = interval
        self._gpu_handles: list[tuple[int, object]] = []   # (index, handle)

        if _GPU_AVAILABLE:
            if gpu_indices is None:
                gpu_indices = list(range(_GPU_COUNT))
            for i in gpu_indices:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self._gpu_handles.append((i, handle))
                except Exception:
                    pass

        self._gpu_indices = [i for i, _ in self._gpu_handles]

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
            stat = StageStats(name=name, depth=depth, gpu_indices=self._gpu_indices)
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
        _print_summary(self._roots, self._gpu_indices)

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
                for stat in self._stack:
                    stat.add_sample(sample)
            time.sleep(self.interval)

    def _take_sample(self) -> Sample:
        cpu_pct     = psutil.cpu_percent()
        ram         = psutil.virtual_memory()
        ram_used_gb = ram.used / 1024 ** 3

        gpu_samples: list[GPUSample] = []
        for _, handle in self._gpu_handles:
            try:
                util  = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem   = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_samples.append(GPUSample(
                    pct=float(util.gpu),
                    mem_used_gb=mem.used / 1024 ** 3,
                ))
            except Exception:
                gpu_samples.append(GPUSample(pct=0.0, mem_used_gb=0.0))

        return Sample(cpu_pct, ram_used_gb, gpu_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_WHITE  = "\033[97m"
_GREEN  = "\033[32m"


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


def _visible_len(s: str) -> int:
    import re
    return len(re.sub(r'\x1b\[[0-9;]*m', '', s))


def _center(text: str, width: int) -> str:
    visible = _visible_len(text)
    total_pad = max(0, width - visible)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return " " * left_pad + text + " " * right_pad


def _print_summary(roots: list, gpu_indices: list[int]):
    has_gpu   = len(gpu_indices) > 0
    n_gpus    = len(gpu_indices)

    # Column widths
    STAGE_W  = 32
    CPU_W    = 28   # "  avg:XX.X% / peak:XX.X%"
    RAM_W    = 26
    # Per-GPU columns (repeated n_gpus times)
    GPU_W    = 28 if has_gpu else 0
    VRAM_W   = 26 if has_gpu else 0
    TIME_W   = 12

    gpu_block_w = (GPU_W + VRAM_W) * n_gpus
    INNER = STAGE_W + CPU_W + RAM_W + gpu_block_w + TIME_W

    def rule(l, r):
        print(_CYAN + l + "─" * INNER + r + _RESET)

    def divider():
        print(_DIM + _CYAN + "│" + "·" * INNER + "│" + _RESET)

    def row(cells: list[tuple[str, int]], bold=False, dim=False):
        prefix = _BOLD if bold else (_DIM if dim else "")
        inner = "".join(
            content + " " * max(0, width - _visible_len(content))
            for content, width in cells
        )
        print(_CYAN + "│" + _RESET + prefix + inner + _RESET + _CYAN + "│" + _RESET)

    # ── Top border & title ──────────────────────────────────────────────────
    rule("┌", "┐")
    gpu_label = f"{n_gpus} GPU{'s' if n_gpus != 1 else ''}" if has_gpu else "CPU only"
    title = f"  {_WHITE}{_BOLD}PIPELINE SUMMARY{_RESET}{_DIM}  [{gpu_label}]{_RESET}"
    row([(title, INNER)])
    rule("├", "┤")

    # ── GPU index sub-header (only when >1 GPU) ─────────────────────────────
    if has_gpu and n_gpus > 1:
        gpu_hdrs = []
        for i in gpu_indices:
            label = _GREEN + _BOLD + f" GPU {i}" + _RESET
            gpu_hdrs += [
                (_center(label, GPU_W),  GPU_W),
                ("", VRAM_W),
            ]
        row([
            ("", STAGE_W),
            ("", CPU_W),
            ("", RAM_W),
            *gpu_hdrs,
            ("", TIME_W),
        ])

    # ── Column headers ───────────────────────────────────────────────────────
    gpu_col_hdrs = []
    for i in gpu_indices:
        lbl = f"GPU{i}" if n_gpus > 1 else "GPU"
        gpu_col_hdrs += [
            (_DIM + _center(f"{lbl} util avg/peak", GPU_W),  GPU_W),
            (_DIM + _center(f"{lbl} VRAM avg/peak", VRAM_W), VRAM_W),
        ]

    row([
        (_DIM + f" {'STAGE':<{STAGE_W - 1}}", STAGE_W),
        (_DIM + _center("CPU avg / peak", CPU_W),  CPU_W),
        (_DIM + _center("RAM avg / peak", RAM_W),  RAM_W),
        *gpu_col_hdrs,
        (_DIM + _center("time", TIME_W), TIME_W),
    ])
    rule("├", "┤")

    # ── Stage rows ───────────────────────────────────────────────────────────
    def print_stage(stat: StageStats):
        indent = "  " * stat.depth
        prefix = "▸ " if stat.children else "  "
        name   = indent + prefix + stat.name
        color  = (_YELLOW + _BOLD) if stat.depth == 0 else ""

        col_stage = color + name + _RESET
        col_cpu   = f"  {_fmt_pct(stat.cpu_avg)} / {_fmt_pct(stat.cpu_peak)}"
        col_ram   = f"  {_fmt_gb(stat.ram_avg)} / {_fmt_gb(stat.ram_peak)}"
        col_time  = f"  {_fmt_time(stat.elapsed):>{TIME_W - 2}}"

        gpu_cells = []
        for i in gpu_indices:
            gs = stat.gpu_stats.get(i)
            if gs:
                gpu_cells += [
                    (f"  {_fmt_pct(gs.avg_pct)} / {_fmt_pct(gs.peak_pct)}", GPU_W),
                    (f"  {_fmt_gb(gs.avg_mem)} / {_fmt_gb(gs.peak_mem)}",   VRAM_W),
                ]
            else:
                gpu_cells += [("  N/A", GPU_W), ("  N/A", VRAM_W)]

        cells = [
            (col_stage, STAGE_W),
            (col_cpu,   CPU_W),
            (col_ram,   RAM_W),
            *gpu_cells,
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