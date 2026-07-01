"""
make_clean_universe.py — turn the data-quality report into a usable universe.

Reads data/data_quality_report.csv and writes universe_clean.txt, KEEPING every
symbol except the genuinely bad ones:

  DROP if:
    - download failed (no data at all), OR
    - >5 zero-volume days (real illiquidity, not a one-off holiday tick), OR
    - any |daily return| > 50% (possible unadjusted corporate action / bad tick), OR
    - any long flat run (the report's max_flat_run > 5).

  KEEP everything else — including names with a single zero-volume day (a holiday
  quirk over 1480 days is harmless).

The harness still picks its ~250 most-liquid names from this list each day; this
just removes the names that would inject garbage into PCA.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

DATA = Path("./data")
report_path = DATA / "data_quality_report.csv"
if not report_path.exists():
    sys.exit(f"missing {report_path} — run data_prep.py first")

df = pd.read_csv(report_path)

# normalize columns that may or may not be present
def col(name, default=0):
    return df[name] if name in df.columns else pd.Series([default] * len(df))

issues = col("issues", "").astype(str).fillna("")
rows = col("rows", 0).fillna(0)
flat = col("max_flat_run", 0).fillna(0)

# parse the few numeric facts we need straight from the issues text where needed
def zero_vol_days(s: str) -> int:
    # issues like "13 zero-volume day(s)"
    for part in str(s).split(";"):
        part = part.strip()
        if "zero-volume" in part:
            try:
                return int(part.split()[0])
            except (ValueError, IndexError):
                return 0
    return 0

def has_extreme_move(s: str) -> bool:
    return ">50%" in str(s)

zvd = issues.apply(zero_vol_days)
extreme = issues.apply(has_extreme_move)
failed = issues.str.contains("download failed", case=False, na=False)

drop_mask = (
    failed
    | (zvd > 5)
    | extreme
    | (flat > 5)
    | (rows < 200)            # safety: far too few rows = unusable
)

keep = df.loc[~drop_mask, "symbol"].astype(str).str.strip().tolist()
dropped = df.loc[drop_mask, ["symbol"]].assign(reason="")  # for printing

# build human-readable drop reasons
reasons = []
for _, r in df.loc[drop_mask].iterrows():
    why = []
    iss = str(r.get("issues", ""))
    if "download failed" in iss.lower(): why.append("no data (likely recent IPO)")
    if zero_vol_days(iss) > 5: why.append(f"{zero_vol_days(iss)} zero-vol days")
    if ">50%" in iss: why.append("extreme daily move")
    fr = r.get("max_flat_run", 0)
    if pd.notna(fr) and fr > 5: why.append(f"{int(fr)} flat closes")
    reasons.append(f"{r['symbol']}: {', '.join(why) or 'failed checks'}")

out = Path("universe_clean.txt")
out.write_text("\n".join(sorted(keep)))

print(f"KEEP {len(keep)} symbols -> {out}")
print(f"DROP {int(drop_mask.sum())} symbols:")
for line in sorted(reasons):
    print("   ", line)
