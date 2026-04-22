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
        torch_dtype = torch.bfloat16
    elif torch.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float
    else:
        device = torch.device("cpu")
        torch_dtype = torch.bfloat16

    return device, torch_dtype