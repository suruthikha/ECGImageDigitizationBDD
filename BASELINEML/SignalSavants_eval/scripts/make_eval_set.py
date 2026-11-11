import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import P

IMG = (".png", ".jpg", ".jpeg")

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--n-records", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

img_dir = Path(P["WORK"]) / args.tag / "images"
res_dir = Path(P["EVAL"]) / "results" / args.tag
img_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

rng = random.Random(args.seed)
dataset = Path(P["DATASET"])
dev_root = Path(P["DEV_SAMPLE"])
rows = []
for hosp in sorted(d.name for d in dataset.iterdir() if d.is_dir()):
    dev_dir = dev_root / hosp
    dev = {d.name for d in dev_dir.iterdir()} if dev_dir.is_dir() else set()
    recs = sorted(d.name for d in (dataset / hosp).iterdir()
                  if d.is_dir() and d.name not in dev)
    for rec in rng.sample(recs, min(args.n_records, len(recs))):
        for f in sorted((dataset / hosp / rec).iterdir()):
            ext = f.suffix.lower()
            if ext not in IMG:
                continue
            out = img_dir / (f.stem + ".png")
            if not out.exists():
                if ext == ".png":
                    out.symlink_to(f)
                else:
                    Image.open(f).convert("RGB").save(out)
            rows.append((f.stem, hosp, rec, ext, str(f)))

with open(res_dir / "manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "hospital", "record", "ext", "source"])
    w.writerows(rows)

print(Counter((r[1], r[3]) for r in rows))
print("total images:", len(rows), "| records:", len(set(r[2] for r in rows)))
print("images in:", img_dir)