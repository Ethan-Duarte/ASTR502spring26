"""
run_stars.py

Two modes:
  mode=0  -> range mode (START_INDEX..END_INDEX from Targets.csv)
  mode=1  -> list mode (explicit list of hostnames; if not in Targets.csv, try SIMBAD for Gaia DR3 ID + RV)

Outputs (from the directory this file is run from):
  ALL_RUNS/<hostname>_friends/   (full FriendFinder output folder)
  ALL_CSVS/<hostname>.csv        (just the single CSV per target)

Designed to be importable + callable from a notebook.
"""

from __future__ import annotations

import os
import shutil
import time
import math
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    s = "" if s is None else str(s)
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip().replace(" ", "_")


def to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def parse_gaia_source_id(raw):
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


def expected_outdir_from_targname(targname: str) -> Path:
    return Path("./" + str(targname).replace(" ", "") + "_friends/")


def collect_csv(run_dir: Path, dest_base: Path, host_label: str):
    run_dir = run_dir.resolve()
    dest_base.mkdir(parents=True, exist_ok=True)

    csvs = list(run_dir.glob("*.csv"))
    if not csvs:
        print("  WARNING: No CSV found in run folder to copy.")
        return False

    src = csvs[0]
    dst = dest_base / f"{host_label}.csv"
    shutil.copy2(src, dst)
    return True


def move_run_dir(run_dir: Path, all_runs_dir: Path, host_label: str) -> Path:
    run_dir = run_dir.resolve()
    all_runs_dir.mkdir(parents=True, exist_ok=True)

    dest = all_runs_dir / f"{host_label}_friends"
    if dest.exists():
        shutil.rmtree(dest)

    shutil.move(str(run_dir), str(dest))
    return dest


def simbad_lookup_gaia_dr3_and_rv(hostname: str):
    out = {
        "hostname": hostname,
        "gaia_id_int": None,
        "rv": None,
        "ra_deg": None,
        "dec_deg": None,
        "note": "",
    }

    try:
        from astroquery.simbad import Simbad
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception as e:
        out["note"] = f"SimbadImportFail: {repr(e)}"
        return out

    def _get_col(table, candidates):
        if table is None:
            return None
        cols = set(table.colnames)
        for c in candidates:
            if c in cols:
                return c
        return None

    try:
        Simbad.reset_votable_fields()
        custom = Simbad()
        try:
            custom.add_votable_fields("rvz_radvel")
        except Exception:
            pass

        t = custom.query_object(hostname)

        if t is None or len(t) == 0:
            out["note"] = "SimbadQueryFail: no rows returned"
        else:
            ra_col = _get_col(t, ["RA", "ra"])
            dec_col = _get_col(t, ["DEC", "dec"])

            if ra_col and dec_col:
                ra_str = t[ra_col][0]
                dec_str = t[dec_col][0]
                c = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")
                out["ra_deg"] = float(c.ra.deg)
                out["dec_deg"] = float(c.dec.deg)
            else:
                out["note"] = (out["note"] + " | " if out["note"] else "") + f"SimbadQueryWarn: RA/DEC cols missing, got {t.colnames}"

            rv_col = _get_col(t, ["RVZ_RADVEL", "rvz_radvel"])
            if rv_col:
                try:
                    rv_val = t[rv_col].filled(np.nan)[0] if hasattr(t[rv_col], "filled") else t[rv_col][0]
                    if rv_val is not None and not (isinstance(rv_val, float) and np.isnan(rv_val)):
                        out["rv"] = float(rv_val)
                except Exception:
                    pass

    except Exception as e:
        out["note"] = f"SimbadQueryFail: {repr(e)}"

    try:
        ids = Simbad.query_objectids(hostname)
        if ids is None or len(ids) == 0:
            out["note"] = (out["note"] + " | " if out["note"] else "") + "SimbadIDsFail: no IDs returned"
        else:
            id_col = _get_col(ids, ["ID", "id"])
            if id_col is None and len(ids.colnames) == 1:
                id_col = ids.colnames[0]

            if id_col is None:
                out["note"] = (out["note"] + " | " if out["note"] else "") + f"SimbadIDsFail: couldn't find ID column, got {ids.colnames}"
            else:
                id_list = [str(x).strip() for x in ids[id_col]]

                gaia_raw = None
                for s in id_list:
                    if s.startswith("Gaia DR3 "):
                        gaia_raw = s.replace("Gaia DR3 ", "").strip()
                        break
                if gaia_raw is None:
                    for s in id_list:
                        if s.startswith("Gaia DR2 "):
                            gaia_raw = s.replace("Gaia DR2 ", "").strip()
                            break

                if gaia_raw is not None:
                    out["gaia_id_int"] = parse_gaia_source_id(gaia_raw)
                else:
                    out["note"] = (out["note"] + " | " if out["note"] else "") + "No Gaia DR3/DR2 ID found in SIMBAD IDs"

    except Exception as e:
        out["note"] = (out["note"] + " | " if out["note"] else "") + f"SimbadIDsFail: {repr(e)}"

    if out["gaia_id_int"] is None:
        out["note"] = (out["note"] + " | " if out["note"] else "") + "No Gaia DR3 ID usable for FriendFinder"
    if out["rv"] is None:
        out["note"] = (out["note"] + " | " if out["note"] else "") + "No RV found (FriendFinder needs RV)"

    return out


def import_comove_module():
    import importlib

    candidates = ["comove", "Comove"]
    errors = []

    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    raise ImportError("Could not import comove module. Tried: " + " | ".join(errors))


def _attempt_findfriends(result_queue, targname, radvel, vlim, srad_eff, rd, verbose, showplots):
    try:
        Comove = import_comove_module()
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
            showplots=showplots,
        )

        if outdir is None:
            result_queue.put(("err", "RuntimeError('findfriends returned None; likely exited early before producing output')"))
            return

        outdir_str = str(outdir)
        if not os.path.exists(outdir_str):
            result_queue.put(("err", f"RuntimeError('findfriends returned {outdir_str!r}, but that path does not exist')"))
            return

        result_queue.put(("ok", outdir_str))

    except Exception as e:
        try:
            result_queue.put(("err", repr(e)))
        except Exception:
            pass


def run(
    *,
    mode: int = 0,
    hostnames: list[str] | None = None,
    csv_path: str = r"Targets.csv",
    start_index: int | None = 0,
    end_index: int | None = 10,
    vlim: float = 5.0,
    srad: float = 40.0,
    showplots: bool = False,
    verbose: bool = False,
    attempt_timeout_s: int = 300,
    max_attempts: int = 1,
    base_backoff_s: int = 10,
    enable_distance_cap: bool = False,
    enable_angle_cap: bool = False,
    theta_max_deg: float = 10.0,
    all_runs_dir: Path | str = Path("ALL_RUNS"),
    all_csvs_dir: Path | str = Path("ALL_CSVS"),
    log_path: Path | str = Path("run_log.csv"),
    sleep_between_targets_s: float = 0.5,
    dedupe_by_gaia_id: bool = True,
    force_rerun: bool = False,
    hostname_case_insensitive: bool = True,
):
    if mode not in (0, 1):
        raise ValueError("mode must be 0 (range) or 1 (list)")

    all_runs_dir = Path(all_runs_dir)
    all_csvs_dir = Path(all_csvs_dir)
    log_path = Path(log_path)

    df = pd.read_csv(
        csv_path,
        dtype={
            "hostname": "string",
            "gaia_dr3_id": "string",
            "ra": "string",
            "dec": "string",
            "st_rv": "string",
            "st_e_rv": "string",
        },
        keep_default_na=False,
    )

    required_cols = ["hostname", "gaia_dr3_id", "ra", "dec", "st_rv"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    all_runs_dir.mkdir(parents=True, exist_ok=True)
    all_csvs_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        log_path.write_text(
            "index,hostname,gaia_id,rv,query_timestamp,attempt_completed_on,srad_used_pc,ok,runtime_s,output_dir,error\n",
            encoding="utf-8",
        )
    print("Logging to:", log_path.resolve())

    completed = set()
    if not force_rerun:
        try:
            old = pd.read_csv(log_path)
            if "ok" in old.columns and "gaia_id" in old.columns:
                completed = set(old.loc[old["ok"] == 1, "gaia_id"].astype(str))
        except Exception:
            completed = set()

    targets = []

    if mode == 0:
        if start_index is None or end_index is None:
            raise ValueError("mode=0 requires start_index and end_index")

        end = min(int(end_index), len(df))
        for idx in range(int(start_index), end):
            targets.append(("df_row", idx, df.iloc[idx]))
    else:
        if not hostnames or not isinstance(hostnames, (list, tuple)):
            raise ValueError("mode=1 requires hostnames=[...] (list of strings)")

        name_to_row = {}
        for i in range(len(df)):
            name = str(df.iloc[i]["hostname"])
            key = name.lower() if hostname_case_insensitive else name
            if key not in name_to_row:
                name_to_row[key] = (i, df.iloc[i])

        for name in hostnames:
            raw = str(name)
            key = raw.lower() if hostname_case_insensitive else raw
            if key in name_to_row:
                idx, row = name_to_row[key]
                targets.append(("df_row", idx, row))
            else:
                sim = simbad_lookup_gaia_dr3_and_rv(raw)
                targets.append(("simbad", -1, sim))

    seen_gaia = set()

    for kind, idx, payload in targets:
        query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attempt_completed_on = 0

        if kind == "df_row":
            row = payload
            host_label = safe_name(row["hostname"])

            gaia_id_int = parse_gaia_source_id(row["gaia_dr3_id"])
            if gaia_id_int is None:
                msg = f"Bad/invalid Gaia DR3 source_id: {repr(row['gaia_dr3_id'])}"
                print(f"\n[{idx}] {host_label} | SKIP: {msg}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f'{idx},{host_label},,{row["st_rv"]},"{query_timestamp}",0,,0,0,,"{msg}"\n')
                continue

            gaia_id = str(gaia_id_int)
            ra = to_float(row["ra"])
            dec = to_float(row["dec"])
            rd = [ra, dec]
            radvel = to_float(row["st_rv"])

            print(f"\n[{idx}] {host_label} | Gaia DR3 {gaia_id} | RV={row['st_rv']}")
            if radvel is None:
                msg = "Missing/invalid RV"
                print(f"  SKIP: {msg}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f'{idx},{host_label},{gaia_id},{row["st_rv"]},"{query_timestamp}",0,,0,0,,"{msg}"\n')
                continue
        else:
            sim = payload
            host_label = safe_name(sim["hostname"])
            gaia_id_int = sim["gaia_id_int"]
            gaia_id = str(gaia_id_int) if gaia_id_int is not None else ""
            ra = sim["ra_deg"]
            dec = sim["dec_deg"]
            rd = [ra, dec]
            radvel = sim["rv"]

            print(f"\n[SIMBAD] {host_label} | Gaia DR3 {gaia_id if gaia_id else '(unknown)'} | RV={radvel if radvel is not None else '(missing)'}")
            print(f"  SIMBAD note: {sim.get('note', '')}")

            if gaia_id_int is None:
                msg = f"SIMBAD lookup failed: {sim.get('note','')}"
                print(f"  SKIP: {msg}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f'-1,{host_label},,{radvel},"{query_timestamp}",0,,0,0,,"{msg}"\n')
                continue
            if radvel is None:
                msg = f"SIMBAD did not provide RV (FriendFinder needs RV). Note: {sim.get('note','')}"
                print(f"  SKIP: {msg}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f'-1,{host_label},{gaia_id},{radvel},"{query_timestamp}",0,,0,0,,"{msg}"\n')
                continue

        targname = host_label

        if (not force_rerun) and gaia_id and (gaia_id in completed):
            print("  SKIP: already completed (resume)")
            continue

        if dedupe_by_gaia_id:
            if gaia_id_int in seen_gaia:
                print("  SKIP: duplicate Gaia source_id in this run")
                continue
            seen_gaia.add(gaia_id_int)

        if ra is None or dec is None:
            print("  WARNING: RA/Dec missing; Comove may fall back to SIMBAD")

        srad_eff = float(srad)
        d_pc = gaia_distance_pc_from_source_id(gaia_id_int)

        if enable_distance_cap and d_pc is not None and srad_eff >= d_pc:
            srad_eff = 0.9 * d_pc
            print(f"  NOTE: distance ~{d_pc:.2f} pc; cap-by-distance: srad -> {srad_eff:.2f} pc")

        if enable_angle_cap and d_pc is not None:
            srad_max_by_angle = d_pc * math.sin(math.radians(float(theta_max_deg)))
            if srad_eff > srad_max_by_angle:
                print(f"  NOTE: cone would be huge; cap-by-angle {theta_max_deg}°: {srad_eff:.2f} -> {srad_max_by_angle:.2f} pc")
                srad_eff = srad_max_by_angle

        ok = 0
        outdir = ""
        err = ""
        runtime = 0.0

        for attempt in range(1, int(max_attempts) + 1):
            attempt_completed_on = attempt

            local_outdir = expected_outdir_from_targname(targname)
            if local_outdir.exists():
                shutil.rmtree(local_outdir)

            print(f"  Attempt {attempt}/{max_attempts} (timeout={attempt_timeout_s}s)")

            q = mp.Queue()
            p = mp.Process(
                target=_attempt_findfriends,
                args=(q, targname, radvel, vlim, srad_eff, rd, verbose, showplots),
                daemon=False,
            )

            t0 = time.perf_counter()
            p.start()
            p.join(int(attempt_timeout_s))

            if p.is_alive():
                p.terminate()
                p.join()
                runtime = time.perf_counter() - t0
                err = f"TimeoutError('Attempt timed out after {attempt_timeout_s}s')"
                print(f"  TIMEOUT after {runtime:.2f}s")
            
                # clean up partial local output folder after timeout
                local_outdir = expected_outdir_from_targname(targname)
                if local_outdir.exists():
                    try:
                        shutil.rmtree(local_outdir)
                        print(f"  Deleted partial timeout folder: {local_outdir}")
                    except Exception as e:
                        print(f"  WARNING: Could not delete timeout folder {local_outdir}: {e}")
            
                if attempt < int(max_attempts):
                    wait = int(base_backoff_s) * (2 ** (attempt - 1))
                    print(f"  Retrying after {wait}s...")
                    time.sleep(wait)
                    continue
            
                ok = 0
                break

            runtime = time.perf_counter() - t0

            try:
                status, payload2 = q.get_nowait()
            except Exception:
                status, payload2 = ("err", "RuntimeError('Child exited before reporting (likely import-time crash)')")

            if status == "ok":
                outdir = str(payload2)
                if not outdir or not os.path.exists(outdir):
                    err = f"RuntimeError('Worker reported success but output directory is invalid: {outdir!r}')"
                    ok = 0
                    print(f"  ERROR: {err}")
                else:
                    err = ""
                    ok = 1
                    print(f"  Runtime: {runtime:.2f} s (attempt {attempt})")
                    break
            else:
                err = payload2
                print(f"  ERROR: {err}")
                print(f"  Runtime: {runtime:.2f} s")

            if attempt < int(max_attempts):
                wait = int(base_backoff_s) * (2 ** (attempt - 1))
                print(f"  Retrying after {wait}s...")
                time.sleep(wait)
                continue

            ok = 0
            break

        if ok and outdir and os.path.exists(outdir):
            run_dir = Path(outdir)
            collect_csv(run_dir, all_csvs_dir, host_label)
            moved_to = move_run_dir(run_dir, all_runs_dir, host_label)
            print(f"  Saved run folder -> {moved_to}")
            outdir = str(moved_to)

        log_index = idx if kind == "df_row" else -1
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f'{log_index},{host_label},{gaia_id},{radvel},"{query_timestamp}",{attempt_completed_on},{srad_eff:.6f},{ok},{runtime:.2f},{outdir},"{err}"\n'
            )

        time.sleep(float(sleep_between_targets_s))


def run_stars(**kwargs):
    return run(**kwargs)
