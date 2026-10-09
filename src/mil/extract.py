"""One pass over a slide's tiles that returns everything we may need later, because the
.tif is deleted afterwards: Phikon CLS token, mean patch token and simple per-tile image
statistics (colour, stain, sharpness).
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from skimage import color, filters
from transformers import ViTModel

from wsi.dataset import TileDataset
from wsi.tiling import read_tile

STAT_NAMES = [
    "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "sat_mean", "val_mean",                          # HSV saturation and brightness
    "hema_mean", "hema_std", "eosin_mean", "eosin_std",  # colour deconvolution (H&E)
    "sharpness",                                     # variance of the Laplacian: low = blurred
]


def tile_stats(rgb: np.ndarray) -> np.ndarray:
    """Hand-crafted descriptors of one uint8 RGB tile, used as a colour baseline for the
    centre probe and for quality control."""
    # Colour statistics on a 2x-subsampled tile: 3x cheaper, same values (r >= 0.995 on
    # tumor_031). Sharpness depends on fine detail, so it uses the full-resolution tile.
    small = rgb[::2, ::2]
    x = small.reshape(-1, 3).astype(np.float32) / 255
    hsv = color.rgb2hsv(small).reshape(-1, 3)
    hed = color.rgb2hed(small).reshape(-1, 3)
    sharp = filters.laplace(color.rgb2gray(rgb)).var()
    return np.array([*x.mean(0), *x.std(0), hsv[:, 1].mean(), hsv[:, 2].mean(),
                     hed[:, 0].mean(), hed[:, 0].std(), hed[:, 1].mean(), hed[:, 1].std(),
                     sharp], dtype=np.float32)


class TileStatsDataset(TileDataset):
    """Project 1's TileDataset, plus the statistics of the same tile (read only once)."""

    def __getitem__(self, idx: int):
        img = read_tile(self.slide, self.tiles.iloc[idx])
        return self.transform(img), idx, torch.from_numpy(tile_stats(np.asarray(img)))


def iter_batches(dataset: TileStatsDataset, batch_size: int, n_threads: int, prefetch: int = 8):
    """Yield collated batches (images, idx, stats), read by a pool of threads.

    Why threads and not DataLoader workers: on macOS each worker is a fresh Python process
    that re-imports torch/transformers/skimage, which cost ~35 s per slide with 4 workers,
    i.e. hours over 399 slides. OpenSlide is thread-safe and its C code (like PIL's resize)
    releases the GIL, so 4 threads already read ~500 tiles/s, 4x faster than the model.
    At most `prefetch` batches are in flight, so memory stays bounded on large slides.
    """
    def load(idxs):
        items = [dataset[i] for i in idxs]
        return (torch.stack([it[0] for it in items]), torch.tensor([it[1] for it in items]),
                torch.stack([it[2] for it in items]))

    dataset.slide  # open the shared handle once, before the threads race to do it
    starts = range(0, len(dataset), batch_size)
    with ThreadPoolExecutor(n_threads) as pool:
        pending = deque()
        for s in starts:
            pending.append(pool.submit(load, range(s, min(s + batch_size, len(dataset)))))
            if len(pending) >= prefetch:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def extract(model: ViTModel, dataset: TileStatsDataset, device: torch.device,
            batch_size: int = 64, n_threads: int = 4):
    """-> cls (n, 768), patch_mean (n, 768), stats DataFrame (n, len(STAT_NAMES)),
    all in dataset order."""
    n, d = len(dataset), model.config.hidden_size
    cls = np.zeros((n, d), dtype=np.float32)
    patch_mean = np.zeros((n, d), dtype=np.float32)
    stats = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
    with torch.inference_mode():
        for images, idx, st in iter_batches(dataset, batch_size, n_threads):
            hidden = model(pixel_values=images.to(device)).last_hidden_state  # (B, 197, 768)
            i = idx.numpy()
            cls[i] = hidden[:, 0].cpu().numpy()                  # same as project 1
            patch_mean[i] = hidden[:, 1:].mean(1).cpu().numpy()  # average of the 196 patch tokens
            stats[i] = st.numpy()
    return cls, patch_mean, pd.DataFrame(stats, columns=STAT_NAMES)
