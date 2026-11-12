import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import P

N_REC = 3
root = Path(P["DATASET"])
for hosp in ["D0", "D1", "D2"]:
    recs = sorted(d for d in (root / hosp).iterdir() if d.is_dir())[:N_REC]
    by = defaultdict(list)
    for r in recs:
        for f in sorted(r.iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            suf = f.stem.split("_")[-1]
            with Image.open(f) as im:
                w, h = im.size
                ex = im.getexif()
                cam = f"{ex.get(271, '')} {ex.get(272, '')}".strip()
                dpi = im.info.get("dpi")
                mode = im.mode
            by[suf].append((w, h, f.stat().st_size / 1e6, cam or "-", str(dpi), mode, f.suffix.lower()))
    print("==", hosp)
    for suf in sorted(by):
        rows = by[suf]
        print(" ", suf,
              "| ext", sorted({r[6] for r in rows}),
              "| sizes", sorted({(r[0], r[1]) for r in rows})[:4],
              "| MB %.1f" % (sum(r[2] for r in rows) / len(rows)),
              "| camera", sorted({r[3] for r in rows}),
              "| dpi", sorted({r[4] for r in rows}),
              "| mode", sorted({r[5] for r in rows}))
        