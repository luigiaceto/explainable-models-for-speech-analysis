from __future__ import annotations
import os
import random
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
import urllib.request
import tarfile

def device_or_default(device: str | None = None) -> torch.device:
    """Return the requested device, or choose CUDA/MPS/CPU in that order."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set the most common random seeds used by this project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def download_with_progress(url: str, dest: Path, desc: str) -> None:
    """Download a file with a tqdm progress bar."""
    if dest.exists():
        print(f"File {dest.name} already exists. Skipping download.")
        return

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)

def extract_tar_gz(tar_path: Path, extract_path: Path) -> None:
    """Extract a tar.gz file."""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)