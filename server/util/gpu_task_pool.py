import logging
import queue
import threading
from typing import Any, Callable

import torch

from util.device_utils import DeviceStrategy, all_devices, clean_device_cache, preferred_device

_log = logging.getLogger("util.gpu_task_pool")


class GpuTaskPool:
    """
    Runs a batch of GPU-bound tasks across every available device in parallel --
    one worker per device (see all_devices()), each pulling from a shared queue
    until it's empty -- with an OOM-tolerant fallback: any task whose work_fn
    raises on its first attempt is deferred rather than lost, and retried once
    all parallel work has drained, serially, on a single preferred device.

    Serial retry is the closest thing to "give it more headroom" without needing
    real per-device memory accounting: by the time it runs, nothing else in this
    pool is using any GPU, so a task that OOM'd fighting other workers for memory
    gets a genuinely uncontended shot on top of whichever device
    preferred_device(device_strategy) picks.

    A task that fails on that fallback attempt is NOT caught -- there is nowhere
    left to retry it, so the exception propagates out of run() as-is.

    work_fn: Callable[[torch.device, Any], Any] -- receives the device to run on
      and the task's data object; its return value (if any) becomes that task's
      result.
    device_strategy: which device preferred_device() should pick for the serial
      fallback phase (default DeviceStrategy.AUTO). Independent of the parallel
      phase, which always uses every device from all_devices().
    """

    def __init__(
        self,
        work_fn: Callable[[torch.device, Any], Any],
        device_strategy: DeviceStrategy = DeviceStrategy.AUTO,
    ) -> None:
        self._work_fn = work_fn
        self._device_strategy = device_strategy
        self._work_queue: "queue.Queue[tuple[int, Any]]" = queue.Queue()
        self._next_index = 0

    def enqueue(self, data: Any) -> None:
        """Queue one task for the next run(). Not safe to call concurrently
        with run() -- enqueue everything first, then run() once."""
        self._work_queue.put((self._next_index, data))
        self._next_index += 1

    def run(self) -> list[Any]:
        """Drains the work pool across every available device in parallel, then
        retries anything that failed serially on the fallback device. Returns
        results in enqueue order. Raises whatever the fallback phase raises."""
        results: list[Any] = [None] * self._next_index
        fallback: list[tuple[int, Any]] = []
        fallback_lock = threading.Lock()

        devices = all_devices()
        threads = [
            threading.Thread(target=self._worker, args=(device, results, fallback, fallback_lock))
            for device in devices
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if fallback:
            fallback_device, _ = preferred_device(self._device_strategy)
            _log.warning(
                "GpuTaskPool: %d task(s) failed their parallel attempt, retrying serially on %s",
                len(fallback), fallback_device,
            )
            for idx, data in fallback:
                results[idx] = self._work_fn(fallback_device, data)
                clean_device_cache(fallback_device)

        return results

    def _worker(
        self,
        device: torch.device,
        results: list[Any],
        fallback: list[tuple[int, Any]],
        fallback_lock: threading.Lock,
    ) -> None:
        while True:
            try:
                idx, data = self._work_queue.get_nowait()
            except queue.Empty:
                return

            try:
                results[idx] = self._work_fn(device, data)
            except Exception as e:
                _log.warning("GpuTaskPool: task failed on %s (%s), deferring to fallback", device, e)
                with fallback_lock:
                    fallback.append((idx, data))
            finally:
                clean_device_cache(device)
