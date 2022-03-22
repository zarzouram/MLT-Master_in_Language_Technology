from typing import List, Tuple
import torch
import subprocess


def get_gpu_memory():
    """Get the current gpu usage.
    Reference: https://stackoverflow.com/a/49596019

    Returns
    -------
    usage: list
        Values of memory usage per GPU as integers in MB.
    """
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ])
    # Convert lines into a dictionary
    result = result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]

    return gpu_memory


def get_gpus_avail():
    """Get the GPU ids that have memory usage less than 50%
    """
    memory_usage = get_gpu_memory()

    memory_usage_percnt = [m / 11178 for m in memory_usage]
    cuda_ids = [(i, m) for i, m in enumerate(memory_usage_percnt) if m <= 0.4]

    header = ["cuda id", "Memory usage"]
    no_gpu_mssg = "No available GPU"
    if cuda_ids:
        print(f"{header[0]:^10}{header[1]:^15}")
        for (idx, m) in cuda_ids:
            print(f"{idx:^10}{m:^15.2%}")
    else:
        print(f"{no_gpu_mssg:-^25}")
        print(f"{header[0]:^10}{header[1]:^15}")
        for idx, m in enumerate(memory_usage_percnt):
            print(f"{idx:^10}{m:^15.2%}")

    return sorted(cuda_ids, key=lambda tup:
                  (tup[1], tup[0])) if cuda_ids else cuda_ids


def select_device(gpu_id: int) -> torch.device:
    # get gpus that have >=75% free space
    cuda_idx = get_gpus_avail()  # type: List[Tuple[int, float]]
    cuda_id = None
    if cuda_idx:
        if gpu_id != -1:
            selected_gpu_avail = next(
                (i for i, v in enumerate(cuda_idx) if v[0] == gpu_id), None)
            if selected_gpu_avail is not None:
                cuda_id = gpu_id  # selected gpu has suitable free space
        else:
            cuda_id = cuda_idx[0][0]  # gpu with the most avail free space

    if cuda_id is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda_id}")

    print(f"\ndevice selected: {device}")

    return device
