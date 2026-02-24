import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
import numpy as np
import pandas as pd
import os
import shutil
import time
from pathlib import Path
import math

import Comove  # make sure this imports correctly in the same environment

# Notes:
# - File Handling was done largely with help from ChatGPT
# -


# -------------------------
# SETTINGS YOU WILL EDIT
# -------------------------
CSV_PATH = "Targets.csv"

START_INDEX = 1       # inclusive
END_INDEX   = 11      # exclusive

vlim = 5               # km/s
srad = 15.0            # pc

SHOWPLOTS = False
VERBOSE = False

# Master output folders (created if missing)
ALL_RUNS_DIR = Path("ALL_RUNS")
ALL_CSVS_DIR = Path("ALL_CSVS")


"""
--Header to ids--
hostname = 1
gaia id  = 2
ra       = 6
dec      = 7
st_rv    = 21
st_e_rv  = 22
"""
COL_HOSTNAME = 1
COL_GAIA_ID  = 2
COL_RA       = 6
COL_DEC      = 7
COL_RV       = 21


# -------------------------
# Helpers
# -------------------------
def safe_name(s: str) -> str:
    """Windows-safe name for folders/files."""
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip().replace(" ", "_")


def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def collect_csvs(run_dir: Path, dest_base: Path):
    """
    Copy all CSVs from run_dir into dest_base/<run_dir_name>/.
    Keeps per-target subfolders to avoid filename collisions.
    """
    run_dir = run_dir.resolve()
    target_csv_dir = dest_base / run_dir.name
    target_csv_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(run_dir.glob("*.csv"))
    for f in csvs:
        shutil.copy2(f, target_csv_dir / f.name)

    # If you ALSO want .txt outputs, uncomment:
    # txts = list(run_dir.glob("*.txt"))
    # for f in txts:
    #     shutil.copy2(f, target_csv_dir / f.name)


def move_run_dir(run_dir: Path, all_runs_dir: Path):
    """
    Move the whole run directory into ALL_RUNS/.
    If it already exists there, delete and replace.
    """
    run_dir = run_dir.resolve()
    all_runs_dir.mkdir(parents=True, exist_ok=True)

    dest = all_runs_dir / run_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(run_dir), str(dest))
    return dest


def gaia_distance_pc_from_source_id(source_id: str):
    """
    Fast, single-row Gaia query: fetch parallax for a specific Gaia DR3 source_id.
    Returns distance in pc (float) or None if unavailable.
    """
    try:
        sid = int(str(source_id).strip())
    except Exception:
        return None

    query = f"""
    SELECT parallax, parallax_error
    FROM gaiadr3.gaia_source
    WHERE source_id = {sid}
    """

    try:
        job = Gaia.launch_job_async(query)
        r = job.get_results()
        if len(r) == 0:
            return None

        plx = r["parallax"][0]  # mas
        if plx is None or np.isnan(plx) or plx <= 0:
            return None

        return 1000.0 / float(plx)  # pc

    except Exception:
        return None


# -------------------------
# Load targets
# -------------------------
arr = np.genfromtxt(CSV_PATH, delimiter=",", dtype=str, encoding="utf-8")

Simbad.server = "simbad.cds.unistra.fr"

ALL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
ALL_CSVS_DIR.mkdir(parents=True, exist_ok=True)

# Optional: log file so you can see what succeeded/failed
log_path = Path("run_log.csv")
if not log_path.exists():
    log_path.write_text("index,hostname,gaia_id,rv,ok,runtime_s,output_dir,error\n", encoding="utf-8")


# -------------------------
# Run a range of targets
# -------------------------
for targetIndex in range(START_INDEX, min(END_INDEX, len(arr))):

    host_raw = arr[targetIndex][COL_HOSTNAME]
    gaia_id  = arr[targetIndex][COL_GAIA_ID].strip()
    ra       = arr[targetIndex][COL_RA]
    dec      = arr[targetIndex][COL_DEC]
    rv_raw   = arr[targetIndex][COL_RV]

    host_label = safe_name(host_raw)
    targname = f"Gaia DR3 {gaia_id}"
    rd = [ra, dec]

    radvel = to_float(rv_raw)

    print(f"\n[{targetIndex}] {host_label} | {targname} | RV={rv_raw}")

    if radvel is None:
        msg = "Missing/invalid RV"
        print(f"  SKIP: {msg}")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{targetIndex},{host_label},{gaia_id},{rv_raw},0,0,,{msg}\n")
        continue

    # --- Avoid Comove "all-sky search" mode by capping srad for very nearby targets ---
    d_pc = gaia_distance_pc_from_source_id(gaia_id)
    THETA_MAX_DEG = 10.0  # try 5.0 if still heavy

    srad_eff = srad
    if d_pc is not None:
        srad_eff = min(srad_eff, 0.9*d_pc)  # avoid all-sky
        srad_eff = min(srad_eff, d_pc * math.sin(math.radians(THETA_MAX_DEG)))
    
        if d_pc is None:
            print("  WARNING: couldn't fetch Gaia parallax for source_id; srad not capped (may trigger all-sky search)")
        elif srad_eff >= d_pc:
            srad_eff = 0.9 * d_pc
            print(f"  NOTE: distance ~{d_pc:.2f} pc; capping srad {srad:.2f} -> {srad_eff:.2f} pc to avoid all-sky search")

    # --- Run FriendFinder (timed) ---
    t0 = time.perf_counter()
    ok = 1
    outdir = ""
    err = ""

    try:
        outdir = Comove.findfriends(
            targname,
            radvel,
            velocity_limit=vlim,
            search_radius=srad_eff,   # <-- USE capped radius
            radec=rd,
            output_directory=None,
            verbose=VERBOSE,
            showplots=SHOWPLOTS
        )
    except Exception as e:
        ok = 0
        err = repr(e)
        print(f"  ERROR: {err}")

    runtime = time.perf_counter() - t0
    print(f"  Runtime: {runtime:.2f} s")

    # --- Organize outputs ---
    if ok and outdir and os.path.exists(outdir):
        run_dir = Path(outdir)

        # 1) Copy CSVs into ALL_CSVS/<run_folder_name>/
        collect_csvs(run_dir, ALL_CSVS_DIR)

        # 2) Move full run folder into ALL_RUNS/
        moved_to = move_run_dir(run_dir, ALL_RUNS_DIR)
        print(f"  Saved run folder -> {moved_to}")

        outdir = str(moved_to)

    # --- Log result ---
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{targetIndex},{host_label},{gaia_id},{rv_raw},{ok},{runtime:.2f},{outdir},{err}\n")