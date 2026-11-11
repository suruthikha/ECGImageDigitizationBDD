import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import P

PROJ = Path(P["PROJ"])
PRED = PROJ / "sigsav_out_test"
SCORES = Path(P["EVAL"]) / "results" / "dev_sample" / "scores.csv"
DATASET = Path(P["DATASET"])
OUT = Path(P["EVAL"]) / "results" / "dev_sample_figs"
OUT.mkdir(parents=True, exist_ok=True)

OFF = {"AVR": 2.5, "AVL": 2.5, "AVF": 2.5, "V1": 5.0, "V2": 5.0, "V3": 5.0,
       "V4": 7.5, "V5": 7.5, "V6": 7.5}

# table: per hospital and image index
d = pd.read_csv(SCORES)
d = d[d.status.isin(["ok", "missing", "empty"])].copy()
d["idx"] = d.image.str.rsplit("_", n=1).str[1]
d["usable"] = d.snr_aligned.notna()
t = d.groupby(["hospital", "idx"]).agg(
    leads=("lead", "size"),
    usable_frac=("usable", "mean"),
    aligned_median=("snr_aligned", "median"),
).round(2)
t.to_csv(OUT / "by_condition.csv")
print(t.to_string())


def overlay(name, leads=("I", "II", "AVR", "V1", "V4")):
    rec = name.rsplit("_", 1)[0]
    g = wfdb.rdrecord(str(DATASET / rec[:2] / rec / rec))
    p = wfdb.rdrecord(str(PRED / name))
    gn = [s.upper() for s in g.sig_name]
    pn = [s.upper() for s in p.sig_name]
    fig, ax = plt.subplots(len(leads), 1, figsize=(14, 2.2 * len(leads)), sharex=True)
    for a, lead in zip(ax, leads):
        if lead not in gn or lead not in pn:
            a.set_ylabel(lead + " (missing)")
            continue
        y = g.p_signal[:, gn.index(lead)]
        yh = p.p_signal[:, pn.index(lead)]
        nz = np.nonzero(yh)[0]
        if len(nz):
            yh = yh[nz[0]:nz[-1] + 1]
        a.plot(np.arange(len(y)) / g.fs, y, lw=0.8, label="ground truth")
        a.plot(np.arange(len(yh)) / p.fs + OFF.get(lead, 0), yh, lw=0.8, label="digitized")
        a.set_ylabel(lead)
    ax[0].legend(loc="upper right")
    ax[-1].set_xlabel("seconds")
    fig.suptitle(name)
    fig.savefig(OUT / f"overlay_{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


for n in ["D0_007904_02", "D0_018718_07", "D2_000262_l00"]:
    overlay(n)