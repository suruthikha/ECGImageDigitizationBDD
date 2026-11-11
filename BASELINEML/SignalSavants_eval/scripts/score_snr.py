import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import P

# Assumed page layout: 3 rows x 4 columns plus a 10 s rhythm strip (lead II).
# Short leads are 2.5 s long and start at their column offset (seconds).
OFF = {"I": 0.0, "II": 0.0, "III": 0.0,
       "AVR": 2.5, "AVL": 2.5, "AVF": 2.5,
       "V1": 5.0, "V2": 5.0, "V3": 5.0,
       "V4": 7.5, "V5": 7.5, "V6": 7.5}
EXPECTED_SEC = 10.0   # records of any other duration are skipped, not scored
MAX_SHIFT = 25        # samples, for the aligned score
MIN_COVER_SEC = 1.0   # less predicted signal than this counts as empty
MIN_SAMPLES = 50


def snr(y, yh):
    num = np.sum(y ** 2)
    den = np.sum((yh - y) ** 2)
    if num == 0 or den == 0:
        return np.nan
    return 10 * np.log10(num / den)


def aligned_snr(ref_full, est, lo0):
    # slide the ground-truth window +-MAX_SHIFT samples, remove DC offset, keep the best
    n = len(est)
    best = np.nan
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        lo = lo0 + s
        if lo < 0 or lo + n > len(ref_full):
            continue
        seg = ref_full[lo:lo + n]
        v = snr(seg, est - np.mean(est - seg))
        if not np.isnan(v) and (np.isnan(best) or v > best):
            best = v
    return best


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    return tuple(np.percentile(meds, [2.5, 97.5]))


def fmt(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--pred-dir", help="override; default WORK/<tag>/out")
ap.add_argument("--images-dir", help="list expected images from a folder of PNGs instead of manifest.csv")
ap.add_argument("--processed", help="text file of image stems; restrict to these (for runs that were killed)")
ap.add_argument("--out-dir", help="override; default EVAL/results/<tag>")
args = ap.parse_args()

res_dir = Path(args.out_dir) if args.out_dir else Path(P["EVAL"]) / "results" / args.tag
res_dir.mkdir(parents=True, exist_ok=True)
pred_dir = Path(args.pred_dir) if args.pred_dir else Path(P["WORK"]) / args.tag / "out"

if args.images_dir:
    images = sorted(f.stem for f in Path(args.images_dir).glob("*.png"))
else:
    images = sorted(pd.read_csv(res_dir / "manifest.csv").image)
if args.processed:
    keep = set(Path(args.processed).read_text().split())
    images = [i for i in images if i in keep]

gt_cache = {}


def load_gt(hosp, rec):
    if rec not in gt_cache:
        gt_cache[rec] = wfdb.rdrecord(str(Path(P["DATASET"]) / hosp / rec / rec))
    return gt_cache[rec]


rows = []
for name in images:
    rec, idx = name.rsplit("_", 1)
    hosp = rec.split("_")[0]
    base = {"image": name, "hospital": hosp, "record": rec, "idx": idx}
    blank = {"snr_window": np.nan, "snr_aligned": np.nan}
    if not (pred_dir / (name + ".hea")).exists():
        rows.append({**base, "lead": "ALL", "status": "failed", **blank})
        continue
    g = load_gt(hosp, rec)
    if abs(g.sig_len / g.fs - EXPECTED_SEC) > 0.1:
        rows.append({**base, "lead": "ALL", "status": "unscorable_duration", **blank})
        continue
    p = wfdb.rdrecord(str(pred_dir / name))
    gn = [s.upper() for s in g.sig_name]
    pn = [s.upper() for s in p.sig_name]
    for lead in gn:
        r = {**base, "lead": lead, **blank}
        if lead not in pn:
            rows.append({**r, "status": "missing"})
            continue
        y = np.nan_to_num(g.p_signal[:, gn.index(lead)])
        yh = p.p_signal[:, pn.index(lead)]
        if g.fs != p.fs:
            y = resample(y, int(round(len(y) * p.fs / g.fs)))
        nz = np.nonzero(yh)[0]
        if len(nz) < MIN_COVER_SEC * p.fs:
            rows.append({**r, "status": "empty"})
            continue
        a, b = nz[0], nz[-1] + 1
        o = int(OFF.get(lead, 0.0) * p.fs)
        n = min(b - a, len(y) - (a + o))
        if n < MIN_SAMPLES:
            rows.append({**r, "status": "empty"})
            continue
        est = yh[a:a + n]
        r["snr_window"] = snr(y[a + o:a + o + n], est)
        r["snr_aligned"] = aligned_snr(y, est, a + o)
        rows.append({**r, "status": "ok"})

d = pd.DataFrame(rows)
d.to_csv(res_dir / "scores.csv", index=False)

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out(f"tag: {args.tag} | images expected: {len(images)} | predictions: {pred_dir}")
out("aligned = approximation (+-%d samples shift, DC removed); NOT the official Challenge metric" % MAX_SHIFT)
out("layout assumed for every hospital: %s (unverified for D1)" % OFF)
out("a negative SNR is worse than predicting zeros")
out()
for h in sorted(d.hospital.unique()):
    dh = d[d.hospital == h]
    n_img = dh.image.nunique()
    n_fail = dh[dh.status == "failed"].image.nunique()
    n_skip = dh[dh.status == "unscorable_duration"].image.nunique()
    lead_rows = dh[dh.status.isin(["ok", "missing", "empty"])]
    ok = dh[dh.status == "ok"]
    if len(lead_rows) == 0:
        out(f"{h}: images {n_img} | failed {n_fail} | skipped (record is not 10 s) {n_skip} | nothing scored")
        continue
    per_rec = ok.groupby("record").snr_aligned.median().values
    lo, hi = boot_ci(per_rec)
    out(f"{h}: images {n_img} | failed {n_fail} | skipped (not 10 s) {n_skip} | records scored {ok.record.nunique()}"
        f" | usable leads {(lead_rows.status == 'ok').mean():.0%}")
    out(f"    lead-level median: window {fmt(ok.snr_window.median())}, aligned {fmt(ok.snr_aligned.median())}")
    out(f"    record-level median aligned {fmt(float(np.median(per_rec)) if len(per_rec) else np.nan)}"
        f" (95% CI {fmt(lo)} to {fmt(hi)}, bootstrap over records)")
out()

cond = d[d.status.isin(["ok", "missing", "empty"])].copy()
cond["usable"] = cond.status == "ok"
t = cond.groupby(["hospital", "idx"]).agg(
    images=("image", "nunique"),
    leads=("lead", "size"),
    usable_frac=("usable", "mean"),
    window_median=("snr_window", "median"),
    aligned_median=("snr_aligned", "median"),
).round(2)
t.to_csv(res_dir / "by_condition.csv")
out(t.to_string())
(res_dir / "summary.txt").write_text("\n".join(lines) + "\n")