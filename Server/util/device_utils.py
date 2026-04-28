from enum import Enum
import torch

def offload_pipeline(device: torch.device, pipeline):
    if device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    elif device.type == "mps":
        pipeline.enable_sequential_cpu_offload()

def clean_device_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()

class DeviceStrategy(Enum):
    AUTO = "auto"
    MEMORY = "memory"

def preferred_device(strategy: DeviceStrategy = DeviceStrategy.AUTO) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if strategy == DeviceStrategy.MEMORY:
            best = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.mem_get_info(i)[1]
            )
            device = torch.device(f"cuda:{best}")
        else:
            device = torch.device("cuda")

        torch_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float
    else:
        device = torch.device("cpu")
        torch_dtype = torch.bfloat16

    return device, torch_dtype

def device_name(device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device.index or 0)
        return f"{device} ({name})"

    elif device.type == "mps":
        # MPS (Apple Silicon) doesn't expose a hardware name like CUDA
        # So we infer it from the system
        return f"{device} (Apple Silicon GPU - MPS)"

    else:
        return str(device)

def device_id(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{str(props.uuid)}"

    if device.type == "mps":
        return "mps"

    return "cpu"

def device_from_id(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")

    if s == "mps":
        return torch.device("mps")

    if s.startswith("cuda:"):
        target = s.split(":", 1)[1]

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            if str(props.uuid) == target:
                return torch.device(f"cuda:{i}")

        raise RuntimeError(f"CUDA device not found for UUID {target}")

    raise ValueError(f"Unknown device string: {s}")