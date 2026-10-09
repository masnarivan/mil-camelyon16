"""Step 1: build data/manifest.csv, one row per CAMELYON16 slide.

Sources (all public, no login):
- evaluation/reference.csv  slide label (negative / macro / micro) and centre (0 Radboud, 1 Utrecht)
- bucket listing            slide file sizes, which slides have an annotation XML
- checksums.md5             to verify each download later
- README.md of the bucket   slides whose tumor annotations are NOT exhaustive (hard-coded below)

Usage: uv run scripts/make_manifest.py
"""
import io
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import certifi
import pandas as pd

BUCKET = "https://camelyon-dataset.s3.amazonaws.com"
PREFIX = "CAMELYON16"
ROOT = Path(__file__).resolve().parent.parent
# python.org builds of Python on macOS ship without system CA certificates.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# From the bucket README ("Notes about the data"): these tumor slides may contain tumor
# areas that were not annotated. Fine for MIL (slide label only), but they must not be
# used as tile-level ground truth (baseline training, attention evaluation).
NOT_EXHAUSTIVE = {
    "tumor_010", "tumor_015", "tumor_018", "tumor_020", "tumor_025", "tumor_029",
    "tumor_033", "tumor_034", "tumor_044", "tumor_046", "tumor_051", "tumor_054",
    "tumor_055", "tumor_056", "tumor_067", "tumor_079", "tumor_085", "tumor_092",
    "tumor_095", "tumor_110",
}


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(f"{BUCKET}/{path}", context=SSL_CTX) as r:
        return r.read()


def list_bucket(prefix: str) -> dict[str, int]:
    """file name -> size in bytes for every object under prefix (S3 ListObjectsV2, paginated)."""
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out, token = {}, None
    while True:
        query = f"?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        root = ET.fromstring(fetch(query))
        for obj in root.findall("s3:Contents", ns):
            key = obj.find("s3:Key", ns).text
            out[key.rsplit("/", 1)[-1]] = int(obj.find("s3:Size", ns).text)
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)
        if not token:
            return out


def main():
    ref = pd.read_csv(io.BytesIO(fetch(f"{PREFIX}/evaluation/reference.csv")))
    ref["slide_id"] = ref["image"].str.removesuffix(".tif")

    sizes = list_bucket(f"{PREFIX}/images/")
    xmls = list_bucket(f"{PREFIX}/annotations/")
    md5 = {}
    for line in fetch(f"{PREFIX}/checksums.md5").decode().splitlines():
        digest, name = line.split(maxsplit=1)
        md5[name.lstrip("*")] = digest

    m = pd.DataFrame({
        "slide_id": ref["slide_id"],
        # normal_* / tumor_* are the official training set, test_* the official test set
        "split": ref["slide_id"].str.startswith("test_").map({True: "test", False: "train"}),
        "cls": ref["class"],                                   # negative / macro / micro
        "label": (ref["class"] != "negative").astype(int),    # slide label used by MIL
        "center": ref["center"],                               # 0 Radboud, 1 Utrecht
        "size_gb": ref["slide_id"].map(lambda s: sizes[f"{s}.tif"] / 1e9).round(3),
        "md5": ref["slide_id"].map(lambda s: md5[f"images/{s}.tif"]),
        "has_xml": ref["slide_id"].map(lambda s: f"{s}.xml" in xmls),
    })
    # Usable as tile-level ground truth: negatives (no tumor anywhere) and tumor slides
    # with exhaustive annotations. Not the 20 slides listed above.
    m["exhaustive_annot"] = (m["label"] == 0) | (m["has_xml"] & ~m["slide_id"].isin(NOT_EXHAUSTIVE))

    # Sanity checks: every slide has a file, every positive slide has an XML, no negative has one.
    assert len(m) == len(sizes) == 399, (len(m), len(sizes))
    assert (m["has_xml"] == (m["label"] == 1)).all()
    assert m["slide_id"].isin(NOT_EXHAUSTIVE).sum() == len(NOT_EXHAUSTIVE)

    m = m.sort_values(["split", "slide_id"]).reset_index(drop=True)
    out = ROOT / "data" / "manifest.csv"
    m.to_csv(out, index=False)

    print(f"{len(m)} slides, {m['size_gb'].sum():.0f} GB -> {out}\n")
    print(pd.crosstab([m["split"], m["center"]], m["cls"], margins=True), "\n")
    print(f"exhaustive tile-level ground truth: {m['exhaustive_annot'].sum()} slides "
          f"({(m['exhaustive_annot'] & (m['label'] == 1)).sum()} positive)")


if __name__ == "__main__":
    main()
