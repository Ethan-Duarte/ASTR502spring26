import pandas as pd
import numpy as np
import os
import shutil
import time
import math
from pathlib import Path
from decimal import Decimal, InvalidOperation
from astroquery.gaia import Gaia
import multiprocessing as mp
from datetime import datetime

# -------------------------
# SETTINGS YOU WILL EDIT
# -------------------------
CSV_PATH = r"Targets.csv"

START_INDEX = 67              # Inclusive
END_INDEX   = 100              # Exclusive

vlim = 5.0                    # km / s
srad = 40.0                   # pc

SHOWPLOTS = False             # show plots in output
VERBOSE = False               # lot of print statements :(

ATTEMPT_TIMEOUT_S = 1200       # timeout timer for last attempt

ENABLE_DISTANCE_CAP = True    # caps srad if target is too close. Prevents all sky survey
ENABLE_ANGLE_CAP = True       # caps the maximum portion of the sky to be scanned
THETA_MAX_DEG = 10.0          # maximum degrees of sky to scan

ALL_RUNS_DIR = Path("ALL_RUNS")
ALL_CSVS_DIR = Path("ALL_CSVS")

LOG_PATH = Path("run_log.csv")
SLEEP_BETWEEN_TARGETS_S = 0.5 # How long to wait between targets in seconds

DEDUPE_BY_GAIA_ID = True
MAX_ATTEMPTS = 1              # How many attempts before skipping to next target
BASE_BACKOFF_S = 10           # How many seconds to wait between 1st and 2nd attempt. Doubles each attempt.


# -------------------------
# Helpers
# -------------------------
def safe_name(s: str) -> str:                          # Just makes sure filenames are windows safe
    bad = '<>:"/\\|?*'
    s = "" if s is None else str(s)
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip().replace(" ", "_")


def to_float(x):                                       # Converts value to float. Only used when inputing RV to findfriends. Has exceptions
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def parse_gaia_source_id(raw):                         # Checks that Gaia ID is not empty, and is not shortened with scienctific notation.
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    try:
        d = Decimal(s)
        if d != d.to_integral_value():
            return None
        return int(d)
    except (InvalidOperation, ValueError):
        return None


# VVV Estimates the distance to the target from it's parallax. Works with distance cap. VVV        
def gaia_distance_pc_from_source_id(source_id_int: int):
    if source_id_int is None:
        return None
    query = f"""
    SELECT parallax
    FROM gaiadr3.gaia_source
    WHERE source_id = {int(source_id_int)}
    """
    try:
        job = Gaia.launch_job_async(query, dump_to_file=False)
        r = job.get_results()
        if len(r) == 0:
            return None
        plx = r["parallax"][0]
        if plx is None or np.isnan(plx) or plx <= 0:
            return None
        return 1000.0 / float(plx)
    except Exception:
        return None

### VVV Copies CSVs from comove output into ALL_CSVS VVV
def collect_csvs(run_dir: Path, dest_base: Path, host_label: str):
    """
    Comove writes ONE csv per target. Copy it to ALL_CSVS/ as host_label.csv
    (no subfolders).
    """
    run_dir = run_dir.resolve()
    dest_base.mkdir(parents=True, exist_ok=True)

    csvs = list(run_dir.glob("*.csv"))
    if not csvs:
        print("  WARNING: No CSV found in run folder to copy.")
        return

    src = csvs[0]  # per you: only one
    dst = dest_base / f"{host_label}.csv"
    shutil.copy2(src, dst)

# VVV Moves the output of comove into ALL_RUNS VVV
def move_run_dir(run_dir: Path, all_runs_dir: Path, host_label: str):
    """
    Move run_dir into ALL_RUNS/ and rename to <hostname>_friends.
    If destination exists, replace it.
    """
    run_dir = run_dir.resolve()
    all_runs_dir.mkdir(parents=True, exist_ok=True)

    dest = all_runs_dir / f"{host_label}_friends"
    if dest.exists():
        shutil.rmtree(dest)

    shutil.move(str(run_dir), str(dest))
    return dest

# VVV Renames comove output folder to "hostname_friends" VVV
def expected_outdir_from_targname(targname: str):
    # Matches Comove: './' + targname.replace(" ", "") + '_friends/'
    return Path("./" + str(targname).replace(" ", "") + "_friends/")

# VVV Iitializes findfriends. Has exceptions if there's an error while initializing imports between attempts
def _attempt_findfriends(result_queue, targname, radvel, vlim, srad_eff, rd, verbose, showplots):
    try:
        import Comove
    except Exception as e:
        try:
            result_queue.put(("err", f"ImportError/ComoveImportFail: {repr(e)}"))
        except Exception:
            pass
        return

    try:
        outdir = Comove.findfriends(
            str(targname),
            float(radvel),
            velocity_limit=float(vlim),
            search_radius=float(srad_eff),
            radec=rd,
            output_directory=None,
            verbose=verbose,
            showplots=showplots
        )
        result_queue.put(("ok", outdir))
    except Exception as e:
        try:
            result_queue.put(("err", repr(e)))
        except Exception:
            pass

# VVV Actually runs findfriends. Ton of exceptions for missing values, distance cap, retry attempts. I'll add a more verbose explaination of this function at another time. VVV
def run():
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            "hostname": "string",
            "gaia_dr3_id": "string",
            "ra": "string",
            "dec": "string",
            "st_rv": "string",
            "st_e_rv": "string",
        },
        keep_default_na=False
    )

    required_cols = ["hostname", "gaia_dr3_id", "ra", "dec", "st_rv"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    ALL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ALL_CSVS_DIR.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "index,hostname,gaia_id,rv,query_timestamp,attempt_completed_on,srad_used_pc,ok,runtime_s,output_dir,error\n",
            encoding="utf-8"
        )

    print("Logging to:", LOG_PATH.resolve())

    completed = set()
    try:
        old = pd.read_csv(LOG_PATH)
        if "ok" in old.columns and "gaia_id" in old.columns:
            completed = set(old.loc[old["ok"] == 1, "gaia_id"].astype(str))
    except Exception:
        completed = set()

    seen_gaia = set()

    end = min(END_INDEX, len(df))
    for idx in range(START_INDEX, end):
        row = df.iloc[idx]

        query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attempt_completed_on = 0

        host_label = safe_name(row["hostname"])

        gaia_id_int = parse_gaia_source_id(row["gaia_dr3_id"])
        if gaia_id_int is None:
            msg = f"Bad/invalid Gaia DR3 source_id: {repr(row['gaia_dr3_id'])}"
            print(f"\n[{idx}] {host_label} | SKIP: {msg}")
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    f'{idx},{host_label},,{row["st_rv"]},"{query_timestamp}",{attempt_completed_on},,0,0,,"{msg}"\n'
                )
            continue

        gaia_id = str(gaia_id_int)

        # IMPORTANT CHANGE:
        # Use hostname as targname so Comove writes <hostname>_friends and files named <hostname>*.png/.txt/.csv
        targname = host_label

        if gaia_id in completed:
            print(f"\n[{idx}] {host_label} | {targname} | SKIP: already completed (resume)")
            continue

        if DEDUPE_BY_GAIA_ID:
            if gaia_id_int in seen_gaia:
                print(f"\n[{idx}] {host_label} | {targname} | SKIP: duplicate Gaia source_id in range")
                continue
            seen_gaia.add(gaia_id_int)

        ra = to_float(row["ra"])
        dec = to_float(row["dec"])
        rd = [ra, dec]

        radvel = to_float(row["st_rv"])
        print(f"\n[{idx}] {host_label} | Gaia DR3 {gaia_id} | RV={row['st_rv']}")

        if radvel is None:
            msg = "Missing/invalid RV"
            print(f"  SKIP: {msg}")
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    f'{idx},{host_label},{gaia_id},{row["st_rv"]},"{query_timestamp}",{attempt_completed_on},,0,0,,"{msg}"\n'
                )
            continue

        if ra is None or dec is None:
            print("  WARNING: RA/Dec missing; Comove may fall back to SIMBAD")

        srad_eff = float(srad)
        d_pc = gaia_distance_pc_from_source_id(gaia_id_int)

        if ENABLE_DISTANCE_CAP and d_pc is not None and srad_eff >= d_pc:
            srad_eff = 0.9 * d_pc
            print(f"  NOTE: distance ~{d_pc:.2f} pc; cap-by-distance: srad -> {srad_eff:.2f} pc")

        if ENABLE_ANGLE_CAP and d_pc is not None:
            srad_max_by_angle = d_pc * math.sin(math.radians(float(THETA_MAX_DEG)))
            if srad_eff > srad_max_by_angle:
                print(f"  NOTE: cone would be huge; cap-by-angle {THETA_MAX_DEG}°: {srad_eff:.2f} -> {srad_max_by_angle:.2f} pc")
                srad_eff = srad_max_by_angle

        ok = 0
        outdir = ""
        err = ""
        runtime = 0.0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempt_completed_on = attempt

            local_outdir = expected_outdir_from_targname(targname)
            if local_outdir.exists():
                shutil.rmtree(local_outdir)

            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} (timeout={ATTEMPT_TIMEOUT_S}s)")

            q = mp.Queue()
            p = mp.Process(
                target=_attempt_findfriends,
                args=(q, targname, radvel, vlim, srad_eff, rd, VERBOSE, SHOWPLOTS),
                daemon=False
            )

            t0 = time.perf_counter()
            p.start()
            p.join(ATTEMPT_TIMEOUT_S)

            if p.is_alive():
                p.terminate()
                p.join()
                runtime = time.perf_counter() - t0
                err = f"TimeoutError('Attempt timed out after {ATTEMPT_TIMEOUT_S}s')"
                print(f"  TIMEOUT (attempt {attempt}/{MAX_ATTEMPTS}) after {runtime:.2f}s")

                if attempt < MAX_ATTEMPTS:
                    wait = BASE_BACKOFF_S * (2 ** (attempt - 1))
                    print(f"  Retrying after {wait}s...")
                    time.sleep(wait)
                    continue

                ok = 0
                break

            runtime = time.perf_counter() - t0

            try:
                status, payload = q.get_nowait()
            except Exception:
                status, payload = ("err", "RuntimeError('Child exited before reporting (likely import-time crash)')")

            if status == "ok":
                outdir = payload
                err = ""
                ok = 1
                print(f"  Runtime: {runtime:.2f} s (attempt {attempt})")
                break

            err = payload
            msg = str(payload)
            print(f"  ERROR (attempt {attempt}/{MAX_ATTEMPTS}): {err}")
            print(f"  Runtime: {runtime:.2f} s")

            retryable = ("500" in msg) or ("404" in msg) or ("RemoteDisconnected" in msg) or ("TimeoutError" in msg)
            if retryable and attempt < MAX_ATTEMPTS:
                wait = BASE_BACKOFF_S * (2 ** (attempt - 1))
                print(f"  Retrying after {wait}s...")
                time.sleep(wait)
                continue

            ok = 0
            break

        # --- Organize outputs ---
        if ok and outdir and os.path.exists(outdir):
            run_dir = Path(outdir)

            # copy <hostname>.csv into ALL_CSVS (no subfolders)
            collect_csvs(run_dir, ALL_CSVS_DIR, host_label)

            # move the whole run folder into ALL_RUNS as <hostname>_friends
            moved_to = move_run_dir(run_dir, ALL_RUNS_DIR, host_label)
            print(f"  Saved run folder -> {moved_to}")
            outdir = str(moved_to)

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f'{idx},{host_label},{gaia_id},{row["st_rv"]},"{query_timestamp}",{attempt_completed_on},{srad_eff:.6f},{ok},{runtime:.2f},{outdir},"{err}"\n'
            )

        time.sleep(SLEEP_BETWEEN_TARGETS_S)


if __name__ == "__main__":
    run()