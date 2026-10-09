"""Step 2: Phikon features for many slides, streamed so the disk never holds more than ~2 slides.

For each slide in the manifest:
  download .tif (+ .xml) -> verify md5 -> tissue mask + 20x tile grid -> tile labels
  -> Phikon embeddings -> save features -> delete the .tif
The next slide downloads in a background thread while the current one is embedded.
Resumable: slides whose features already exist are skipped, so it can be stopped and restarted.

Usage:
  uv run scripts/extract_all.py --slides tumor_031 normal_001     # specific slides (pilot)
  caffeinate -i uv run scripts/extract_all.py --split all         # everything, keep the Mac awake
  python scripts/extract_all.py --out /content/drive/MyDrive/c16 --raw /content/raw --dtype float16  # Colab
Outputs in --out (default outputs/), all aligned row by row with the tile table:
  features/<id>.npy          (n_tiles, 768) Phikon CLS token; written last = "done" marker
  features/<id>.parquet      level-0 coords, tissue_frac, tumor_frac, label, colour/sharpness stats
  patch_mean/<id>.npy        (n_tiles, 768) mean of the 196 patch tokens
  thumbs/<id>.npz            thumbnail + tissue mask (for heatmaps once the .tif is gone)
  meta/<id>.json             all OpenSlide properties (mpp, levels, ...)
  qc/<id>.png                thumbnail with tissue-mask outline, to check by eye
  extract_log.csv            one line per slide: tiles, timings, errors
"""
import argparse
import fcntl
import hashlib
import json
import os

# Phikon must already be in the local HF cache (the Colab notebook downloads it first): never
# contact the Hub here. A TLS handshake to the Hub once hung for an hour with no timeout.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import platform
import subprocess
import threading
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI: this runs unattended in the background
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd

from wsi.annotations import label_tiles, load_tumor_geometry, tumor_fraction
from mil.extract import TileStatsDataset, extract
from wsi.dataset import phikon_transform
from wsi.embed import get_device, load_phikon
from mil.tissue import compute_tissue_mask  # background-relative mask, not project 1's Otsu
from wsi.tiling import make_tile_grid, slide_mpp

BUCKET = "https://camelyon-dataset.s3.amazonaws.com/CAMELYON16"
ROOT = Path(__file__).resolve().parent.parent
# Set in main() from --raw / --out, so the same script runs on the Mac and on Colab
# (slides on the VM's local disk, features on Google Drive, which survives disconnections).
RAW = FEAT = QC = PATCH = THUMBS = META = LOG = None


def set_paths(raw: Path, out: Path):
    global RAW, FEAT, QC, PATCH, THUMBS, META, LOG
    RAW, LOG = raw, out / "extract_log.csv"
    FEAT, QC, PATCH, THUMBS, META = (out / d for d in ("features", "qc", "patch_mean", "thumbs", "meta"))


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, out: Path, attempts: int = 10):
    """curl with resume, retried. A connection can stay open while sending nothing, and
    curl has no timeout for that by default (one download hung for 30+ min): abort below
    10 kB/s for 60 s, then resume from where it stopped (-C -)."""
    for i in range(attempts):
        r = subprocess.run(["curl", "-fsSL", "-C", "-", "--speed-limit", "10000", "--speed-time", "60",
                            "--connect-timeout", "30", "-o", out, url])
        if r.returncode == 0:
            return
        print(f"  curl exit {r.returncode} on {url.rsplit('/', 1)[-1]}, retry {i + 1}/{attempts}", flush=True)
        time.sleep(min(60, 5 * 2 ** i))
    raise IOError(f"download failed after {attempts} attempts: {url}")


def download(row) -> Path:
    """Download (or reuse) the slide and its XML; raise if the md5 does not match."""
    tif = RAW / f"{row.slide_id}.tif"
    if not tif.exists():
        part = tif.with_suffix(".tif.part")
        curl(f"{BUCKET}/images/{row.slide_id}.tif", part)
        if md5sum(part) != row.md5:
            part.unlink()  # corrupted: start from scratch next time
            raise IOError(f"md5 mismatch for {row.slide_id}")
        part.rename(tif)
    if row.has_xml and not (xml := RAW / f"{row.slide_id}.xml").exists():
        curl(f"{BUCKET}/annotations/{row.slide_id}.xml", xml)
    return tif


def atomic_write(path: Path, write):
    """Write via a temp file in the same folder, flush it to disk, then rename.

    The rename is atomic on POSIX (same filesystem): `path` is either absent or complete.
    fsync first, otherwise after a power cut the renamed file could still be missing data;
    on macOS only F_FULLFSYNC also flushes the drive's own cache (it does not exist on Linux).
    """
    tmp = path.with_name(path.name + ".tmp")  # not matched by the *.npy glob in main
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        if hasattr(fcntl, "F_FULLFSYNC"):
            fcntl.fcntl(f.fileno(), fcntl.F_FULLFSYNC)
        else:
            os.fsync(f.fileno())
    os.replace(tmp, path)


def save_qc(slide_id: str, tissue, n_tiles: int, mpp: float):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    axes[0].imshow(tissue.thumbnail)
    axes[1].imshow(tissue.thumbnail)
    axes[1].contour(tissue.mask, levels=[0.5], colors="lime", linewidths=0.6)
    axes[0].set_title(f"{slide_id}  ({mpp:.3f} µm/px)", fontsize=9)
    axes[1].set_title(f"tissue mask, {n_tiles} tiles", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(QC / f"{slide_id}.png", dpi=100)
    plt.close(fig)


def process(row, tif: Path, model, device, args) -> dict:
    slide = openslide.OpenSlide(tif)
    mpp = slide_mpp(slide)
    t0 = time.time()
    tissue = compute_tissue_mask(slide)
    tiles = make_tile_grid(slide, tissue)
    if row.has_xml:
        tiles["tumor_frac"] = tumor_fraction(tiles, load_tumor_geometry(RAW / f"{row.slide_id}.xml"))
    else:
        tiles["tumor_frac"] = 0.0  # negative slide: no tumor anywhere
    tiles["label"] = label_tiles(tiles.tumor_frac.to_numpy())
    save_qc(row.slide_id, tissue, len(tiles), mpp)
    t_tile = time.time() - t0

    t0 = time.time()
    dataset = TileStatsDataset(tif, tiles, phikon_transform())
    cls, patch_mean, stats = extract(model, dataset, device, args.batch_size, args.threads)
    t_embed = time.time() - t0
    for a in (cls, patch_mean):
        assert a.shape == (len(tiles), 768) and not np.isnan(a).any()
    tiles = pd.concat([dataset.tiles, stats], axis=1)

    # Everything is written atomically, and the CLS .npy last: its presence is what marks
    # a slide as done (see `done` in main), so a crash never leaves a slide half-saved.
    sid = row.slide_id
    atomic_write(META / f"{sid}.json", lambda f: f.write(json.dumps(dict(slide.properties), indent=1).encode()))
    atomic_write(THUMBS / f"{sid}.npz", lambda f: np.savez_compressed(
        f, thumbnail=tissue.thumbnail, mask=tissue.mask, downsample=tissue.downsample))
    # float16 halves the size (Google Drive space); cosine to float32 >= 0.9999998.
    atomic_write(PATCH / f"{sid}.npy", lambda f: np.save(f, patch_mean.astype(args.dtype)))
    atomic_write(FEAT / f"{sid}.parquet", tiles.to_parquet)
    atomic_write(FEAT / f"{sid}.npy", lambda f: np.save(f, cls.astype(args.dtype)))
    return {"mpp": mpp, "level0_w": slide.dimensions[0], "level0_h": slide.dimensions[1],
            "n_tiles": len(tiles), "n_tumor_tiles": int((tiles.label == "tumor").sum()),
            "t_tile_s": round(t_tile, 1), "t_embed_s": round(t_embed, 1),
            "device": f"{device.type} ({platform.node()})"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", nargs="*", help="slide ids (default: every slide of --split)")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--keep-tif", action="store_true", help="do not delete slides after extraction")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4, help="tile-reading threads")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs", help="where features are saved")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw", help="temporary slide storage")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    set_paths(args.raw, args.out)

    for d in (RAW, FEAT, QC, PATCH, THUMBS, META):
        d.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    if args.slides:
        todo = manifest.set_index("slide_id").loc[args.slides].reset_index()
    else:
        todo = manifest if args.split == "all" else manifest[manifest.split == args.split]
    done = {p.stem for p in FEAT.glob("*.npy")}
    todo = [r for r in todo.itertuples() if r.slide_id not in done]
    print(f"{len(todo)} slides to process ({len(done)} already done)")
    if not todo:
        return

    device = get_device()
    model = load_phikon(device)

    # Prefetch: while slide i is embedded, a thread downloads slide i+1.
    def fetch(row, box):
        try:
            box["tif"] = download(row)
        except Exception as e:  # reported by the main loop
            box["error"] = e

    box = {}
    thread = threading.Thread(target=fetch, args=(todo[0], box))
    thread.start()
    for i, row in enumerate(todo):
        t0 = time.time()
        thread.join()
        cur = box
        t_wait = time.time() - t0
        if i + 1 < len(todo):
            box = {}
            thread = threading.Thread(target=fetch, args=(todo[i + 1], box))
            thread.start()

        rec = {"slide_id": row.slide_id, "cls": row.cls, "center": row.center,
               "size_gb": row.size_gb, "t_wait_download_s": round(t_wait, 1), "error": ""}
        try:
            if "error" in cur:
                raise cur["error"]
            rec |= process(row, cur["tif"], model, device, args)
            if not args.keep_tif:
                cur["tif"].unlink()
            print(f"[{i + 1}/{len(todo)}] {row.slide_id}: {rec['n_tiles']} tiles "
                  f"({rec['n_tumor_tiles']} tumor), tile {rec['t_tile_s']}s, embed {rec['t_embed_s']}s, "
                  f"waited {rec['t_wait_download_s']}s for download", flush=True)
        except Exception as e:  # log and move on: one bad slide must not stop an overnight run
            rec["error"] = repr(e)
            print(f"[{i + 1}/{len(todo)}] {row.slide_id}: ERROR {e!r}", flush=True)
            traceback.print_exc()
        pd.DataFrame([rec]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


if __name__ == "__main__":
    main()
