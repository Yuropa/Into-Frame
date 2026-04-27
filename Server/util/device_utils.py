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

def preferred_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        # Select Cuda device with the most memory available
        best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[1])
        device = torch.device(f"cuda:{best}")
        device = torch.device("cuda")
        torch_dtype = torch.bfloat16
    elif torch.mps.is_available():
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

def device_id(device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        uuid = torch.cuda.get_device_properties(idx).uuid
        return f"cuda:{uuid}"

    elif device.type == "mps":
        return "mps"

    else:
        return "cpu"

def device_from_id(s: str) -> torch.device:
    if s == "cpu":
        return torch.device("cpu")

    if s == "mps":
        return torch.device("mps")

    if s.startswith("cuda:"):
        target_uuid = s.split(":", 1)[1]

        for i in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(i).uuid) == target_uuid:
                return torch.device(f"cuda:{i}")

        raise RuntimeError(f"CUDA device with UUID {target_uuid} not found")

    raise ValueError(f"Unknown device string: {s}")