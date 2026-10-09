# Slide-level metastasis detection on CAMELYON16 with attention-based MIL

**Work in progress.**

Follow-up to [wsi-embeddings](https://github.com/masnarivan/wsi-embeddings), which showed that frozen Phikon embeddings separate tumor from normal tiles on a single slide. This project tackles the actual CAMELYON16 task: deciding whether a whole slide contains metastases, using only slide-level labels.

Questions:

1. Trained with slide labels only, does an attention-MIL model (ABMIL) find the tumor? How does it compare with a tile classifier trained on the pathologists' polygons and aggregated with max pooling?
2. **Domain shift.** CAMELYON16 comes from two centres, Radboud and Utrecht. Can a linear probe tell the centre from the embeddings? How much AUC is lost when training on one centre and testing on the other, compared with size-matched random splits?

## Status

- [x] Manifest of all 399 slides (`data/manifest.csv`): split, label (negative / macro / micro), centre, md5, annotation flags
- [x] Streaming feature extraction (`scripts/extract_all.py`): download → tissue mask → 20x tiles → Phikon CLS + mean patch token + per-tile colour/sharpness stats → delete slide. Resumable, atomic writes.
- [x] Pilot on 10 slides; tissue mask changed from Otsu to a background-relative threshold (`src/mil/tissue.py`), see `outputs/figures/pilot_tissue_masks.png`
- [ ] Full extraction (Colab: `colab/extract_colab.ipynb`)
- [ ] Baseline tile classifier, ABMIL, evaluation, domain shift

## Run

```bash
uv sync
uv run scripts/make_manifest.py
uv run scripts/extract_all.py --slides tumor_031      # or open colab/extract_colab.ipynb in Colab
```

Data: [CAMELYON16](https://camelyon16.grand-challenge.org/) (Ehteshami Bejnordi et al., *JAMA* 2017), CC0, public AWS bucket.
