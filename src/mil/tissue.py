"""Tissue mask relative to each slide's own background, replacing project 1's Otsu threshold.

Why: Otsu assumes a two-mode saturation histogram (glass vs tissue). Lymph-node slides
often have three: glass (~0.01), pale tissue such as fat, stroma and capsule (~0.03-0.10),
and dense lymphoid/tumor tissue (~0.2-0.4). Otsu then splits dense tissue from everything
else and drops the pale tissue (normal_027: 3 mm^2 of tissue found; tumor_094: 91 vs 203 mm^2).
Here a pixel is tissue if it is clearly more saturated than this slide's glass.

Returns project 1's TissueMask, so wsi.tiling.make_tile_grid works unchanged.
"""
import numpy as np
import openslide
from scipy import ndimage as ndi
from skimage import color, morphology

from wsi.tiling import TissueMask


def mask_from_thumbnail(thumb: np.ndarray, margin: float = 0.03, min_object_px: int = 200) -> np.ndarray:
    hsv = color.rgb2hsv(thumb)
    sat = ndi.median_filter(hsv[..., 1], size=3)          # remove single-pixel noise
    not_black = hsv[..., 2] > 0.2                         # slide edges, marker crosses
    # Glass level of THIS slide: median saturation of the scanned area. Pure white (255)
    # is the unscanned part of the slide and would pull the estimate towards 0.
    scanned = (thumb.min(axis=-1) < 250) & not_black
    background = np.median(sat[scanned]) if scanned.any() else 0.0
    mask = (sat > background + margin) & not_black
    # Opening removes 1-2 px wide lines (glass edges, label borders) before the usual clean-up.
    mask = morphology.opening(mask, morphology.disk(1))
    mask = morphology.closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, max_size=min_object_px)
    mask = morphology.remove_small_objects(mask, max_size=min_object_px)
    return mask


def compute_tissue_mask(slide: openslide.OpenSlide, target_downsample: float = 64,
                        margin: float = 0.03) -> TissueMask:
    """Same interface as wsi.tiling.compute_tissue_mask (same level, same thumbnail)."""
    level = slide.get_best_level_for_downsample(target_downsample)
    thumb = np.asarray(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    return TissueMask(mask=mask_from_thumbnail(thumb, margin), downsample=slide.level_downsamples[level],
                      thumbnail=thumb)
