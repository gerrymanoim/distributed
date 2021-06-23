import os

import dask

try:
    import pynvml
except ImportError:
    pynvml = None

nvmlInitialized = False
nvmlAttemptedInit = False
nvmlOwnerPID = None


def init_once():
    global nvmlInitialized, nvmlAttemptedInit, nvmlOwnerPID

    if dask.config.get("distributed.diagnostics.nvml") is False:
        nvmlAttemptedInit = True
        return

    if pynvml is None or (nvmlAttemptedInit is True and nvmlOwnerPID == os.getpid()):
        return

    nvmlAttemptedInit = True
    try:
        pynvml.nvmlInit()
        nvmlInitialized = True
        nvmlOwnerPID = os.getpid()
    except pynvml.NVMLError:
        pass


def device_get_count():
    init_once()
    if not nvmlInitialized:
        return 0
    else:
        return pynvml.nvmlDeviceGetCount()


def _pynvml_handles():
    count = device_get_count()
    if count == 0:
        if not nvmlInitialized and nvmlAttemptedInit:
            raise RuntimeError("Error running pynvml.nvmlInit()")
        else:
            raise RuntimeError("No GPUs available")

    try:
        cuda_visible_devices = [
            int(idx) for idx in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        ]
    except ValueError:
        # CUDA_VISIBLE_DEVICES is not set
        cuda_visible_devices = False
    if not cuda_visible_devices:
        cuda_visible_devices = list(range(count))
    gpu_idx = cuda_visible_devices[0]
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)


def real_time():
    h = _pynvml_handles()
    return {
        "utilization": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
        "memory-used": pynvml.nvmlDeviceGetMemoryInfo(h).used,
    }


def one_time():
    h = _pynvml_handles()
    return {
        "memory-total": pynvml.nvmlDeviceGetMemoryInfo(h).total,
        "name": pynvml.nvmlDeviceGetName(h).decode(),
    }
