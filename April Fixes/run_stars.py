from __future__ import annotations  # allow modern type hints

import multiprocessing as mp  # run each target in a subprocess so timeouts can kill it cleanly
import os  # path existence checks
import shutil  # copy / delete files and folders
import time  # timing and sleeps
from datetime import datetime  # timestamps for logs
from decimal import Decimal, InvalidOperation  # robust Gaia ID parsing
from pathlib import Path  # nicer path handling

import pandas as pd  # read Targets.csv


def safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'  # characters unsafe in Windows filenames
    s = "" if s is None else str(s)  # normalize None to empty string
    for ch in bad:
        s = s.replace(ch, "_")  # replace unsafe characters
    return s.strip().replace(" ", "_")  # trim and replace spaces


def to_float(x):
    try:
        if x is None:
            return None  # preserve missing values
        if isinstance(x, str) and x.strip() == "":
            return None  # blank strings count as missing
        return float(x)  # normal conversion
    except Exception:
        return None  # anything weird becomes missing


def parse_gaia_source_id(raw):
    if raw is None:
        return None  # missing id

    s = str(raw).strip()  # normalize to stripped string
    if s == "" or s.lower() == "nan":
        return None  # blank / nan-like text

    if s.isdigit():
        try:
            return int(s)  # easy clean integer case
        except Exception:
            return None

    try:
        d = Decimal(s)  # handle scientific notation / decimal formatting
        if d != d.to_integral_value():
            return None  # reject non-integers
        return int(d)  # safe integer conversion
    except (InvalidOperation, ValueError):
        return None  # malformed input


def expected_outdir_from_targname(base_dir: Path, targname: str) -> Path:
    return (base_dir / f"{str(targname).replace(' ', '')}_friends").resolve()  # temp folder comove creates


def collect_csv_from_run_dir(run_dir: Path, friends_dir: Path, host_label: str):
    run_dir = run_dir.resolve()  # normalize path
    friends_dir.mkdir(parents=True, exist_ok=True)  # ensure FRIENDS folder exists

    src = run_dir / f"{host_label}.csv"  # exact csv comove should have written
    if not src.exists():
        print(f"  WARNING: Expected CSV not found: {src}")  # do not guess from globbing
        return False

    dst = friends_dir / f"{host_label}.csv"  # final csv location
    shutil.copy2(src, dst)  # copy summary csv
    return True


def _attempt_findfriends(result_queue, targname, gaia_id_int, radvel, vlim, srad_eff, rd, verbose, output_directory):
    try:
        import comove  # fixed module name

        outdir, stats = comove.findfriends(
            str(targname),  # target name / label
            float(radvel),  # target RV
            velocity_limit=float(vlim),  # tangential mismatch threshold
            search_radius=float(srad_eff),  # physical search radius
            radec=rd,  # optional coordinates from Targets.csv
            gaia_source_id=int(gaia_id_int) if gaia_id_int is not None else None,  # exact Gaia target id
            output_directory=output_directory,  # explicit temp folder location
            verbose=verbose,  # optional prints
            showplots=False,  # ignored by current comove but harmless
        )

        if outdir is None:
            result_queue.put(("err", "RuntimeError('findfriends returned None')", None))  # invalid worker output
            return

        outdir_str = str(outdir)  # normalize to string
        if not os.path.exists(outdir_str):
            result_queue.put(("err", f"RuntimeError('findfriends returned {outdir_str!r}, but path does not exist')", None))  # folder should exist
            return

        result_queue.put(("ok", outdir_str, stats))  # return folder + stats on success

    except Exception as e:
        result_queue.put(("err", repr(e), None))  # propagate worker-side error


def run(
    *,
    mode: int = 0,
    hostnames: list[str] | None = None,
    csv_path: str = "Targets.csv",
    start_index: int | None = 0,
    end_index: int | None = 10,
    vlim: float = 5.0,
    srad: float = 40.0,
    verbose: bool = False,
    attempt_timeout_s: int = 300,
    max_attempts: int = 1,
    base_backoff_s: int = 10,
    friends_dir: Path | str = Path("FRIENDS"),
    log_path: Path | str = Path("run_log.csv"),
    sleep_between_targets_s: float = 0.5,
    dedupe_by_gaia_id: bool = True,
    force_rerun: bool = False,
):
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")  # only two modes supported

    if mode == 1:
        raise NotImplementedError("This version keeps the CSV-driven mode only.")  # keep it focused

    base_dir = Path(csv_path).resolve().parent  # anchor all outputs to the folder containing Targets.csv
    friends_dir = (base_dir / friends_dir).resolve() if not Path(friends_dir).is_absolute() else Path(friends_dir).resolve()
    log_path = (base_dir / log_path).resolve() if not Path(log_path).is_absolute() else Path(log_path).resolve()

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
    )  # read csv in text-first mode so ids and blanks survive cleanly

    required_cols = ["hostname", "gaia_dr3_id", "ra", "dec", "st_rv"]  # needed for mode 0
    missing = [c for c in required_cols if c not in df.columns]  # find schema mismatches
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")  # fail early if file shape is wrong

    friends_dir.mkdir(parents=True, exist_ok=True)  # create central FRIENDS output folder

    if not log_path.exists():
        log_path.write_text(
            "index,hostname,gaia_id,rv,query_timestamp,attempt_completed_on,srad_input_pc,srad_used_pc,ok,runtime_s,csv_path,"
            "distance_pc,search_radius_deg,min_parallax_mas,parallax_error_cut_mas,n_query_rows,n_base,n_not_self,n_convergcut,"
            "n_cmd_clean,n_good_rv,n_rv_comoving,n_rv_outlier,n_final,error\n",
            encoding="utf-8",
        )  # initialize log header once

    completed = set()  # previously successful Gaia ids
    if not force_rerun:
        try:
            old = pd.read_csv(log_path)  # inspect prior log
            if "ok" in old.columns and "gaia_id" in old.columns:
                completed = set(old.loc[old["ok"] == 1, "gaia_id"].astype(str))  # skip already successful targets
        except Exception:
            completed = set()  # ignore broken / empty logs

    end = min(int(end_index), len(df))  # end index is exclusive
    seen_gaia = set()  # dedupe within this run

    for idx in range(int(start_index), end):
        row = df.iloc[idx]  # current target row
        host_label = safe_name(row["hostname"])  # safe filename label
        gaia_id_int = parse_gaia_source_id(row["gaia_dr3_id"])  # parse Gaia id
        radvel = to_float(row["st_rv"])  # parse stellar RV
        ra = to_float(row["ra"])  # parse RA if present
        dec = to_float(row["dec"])  # parse Dec if present
        rd = [ra, dec]  # package coordinates for comove

        query_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # human-readable timestamp
        attempt_completed_on = 0  # filled during loop
        stats = None  # filled on success
        final_csv_path = ""  # final csv destination for log

        if gaia_id_int is None:
            msg = f"Bad/invalid Gaia DR3 source_id: {repr(row['gaia_dr3_id'])}"  # invalid source id
            print(f"\n[{idx}] {host_label} | SKIP: {msg}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f'{idx},{host_label},,{row["st_rv"]},"{query_timestamp}",0,{srad:.6f},,0,0,,'
                        f',,,,,,,,,,,,,"{msg}"\n')  # log skip
            continue  # next target

        gaia_id = str(gaia_id_int)  # string form for logging / completed-set lookup
        print(f"\n[{idx}] {host_label} | Gaia DR3 {gaia_id} | RV={row['st_rv']}")  # progress line

        if radvel is None:
            msg = "Missing/invalid RV"  # cannot run without target RV
            print(f"  SKIP: {msg}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f'{idx},{host_label},{gaia_id},{row["st_rv"]},"{query_timestamp}",0,{srad:.6f},,0,0,,'
                        f',,,,,,,,,,,,,"{msg}"\n')  # log skip
            continue  # next target

        if (not force_rerun) and (gaia_id in completed):
            print("  SKIP: already completed")  # resume logic
            continue  # do not rerun

        if dedupe_by_gaia_id:
            if gaia_id_int in seen_gaia:
                print("  SKIP: duplicate Gaia source_id in this run")  # avoid duplicate work in same batch
                continue
            seen_gaia.add(gaia_id_int)  # remember seen target

        if ra is None or dec is None:
            print("  WARNING: RA/Dec missing; comove will fall back if needed")  # Gaia id path should still work

        srad_eff = float(srad)  # requested radius; actual cap happens inside comove

        ok = 0  # assume failure unless worker succeeds
        outdir = ""  # temp run folder path
        err = ""  # error message if any
        runtime = 0.0  # elapsed time

        for attempt in range(1, int(max_attempts) + 1):
            attempt_completed_on = attempt  # record attempt number

            local_outdir = expected_outdir_from_targname(base_dir, host_label)  # exact temp folder path for this run
            if local_outdir.exists():
                shutil.rmtree(local_outdir)  # clean stale leftovers before retry

            print(f"  Attempt {attempt}/{max_attempts} (timeout={attempt_timeout_s}s)")  # progress

            q = mp.Queue()  # child -> parent communication queue
            p = mp.Process(
                target=_attempt_findfriends,
                args=(q, host_label, gaia_id_int, radvel, vlim, srad_eff, rd, verbose, str(local_outdir)),
                daemon=False,
            )  # launch worker with explicit output folder

            t0 = time.perf_counter()  # start timing
            p.start()  # start worker
            p.join(int(attempt_timeout_s))  # wait up to timeout

            if p.is_alive():
                p.terminate()  # kill hung worker
                p.join()  # wait for termination
                runtime = time.perf_counter() - t0  # elapsed time
                err = f"TimeoutError('Attempt timed out after {attempt_timeout_s}s')"  # timeout message
                print(f"  TIMEOUT after {runtime:.2f}s")

                if local_outdir.exists():
                    try:
                        shutil.rmtree(local_outdir)  # delete partial output folder
                    except Exception as e:
                        print(f"  WARNING: Could not delete partial timeout folder: {e}")

                if attempt < int(max_attempts):
                    wait = int(base_backoff_s) * (2 ** (attempt - 1))  # exponential retry backoff
                    print(f"  Retrying after {wait}s...")
                    time.sleep(wait)
                    continue  # retry

                ok = 0  # final failure
                break  # exit attempt loop

            runtime = time.perf_counter() - t0  # elapsed time for finished worker

            try:
                status, payload2, stats = q.get_nowait()  # retrieve worker result
            except Exception:
                status, payload2, stats = ("err", "RuntimeError('Child exited before reporting')", None)  # queue was empty

            if status == "ok":
                outdir = str(payload2)  # temp output folder from worker
                if not outdir or not os.path.exists(outdir):
                    err = f"RuntimeError('Worker reported success but output directory is invalid: {outdir!r}')"
                    ok = 0
                    print(f"  ERROR: {err}")
                else:
                    err = ""  # clear error
                    ok = 1  # success
                    print(f"  Runtime: {runtime:.2f} s")
                    break  # stop retrying
            else:
                err = payload2  # worker-side error string
                print(f"  ERROR: {err}")
                print(f"  Runtime: {runtime:.2f} s")

            if attempt < int(max_attempts):
                wait = int(base_backoff_s) * (2 ** (attempt - 1))  # exponential retry delay
                print(f"  Retrying after {wait}s...")
                time.sleep(wait)
                continue  # retry

            ok = 0  # exhausted retries
            break  # exit attempt loop

        if ok and outdir and os.path.exists(outdir):
            run_dir = Path(outdir)  # temp comove folder
            copied = collect_csv_from_run_dir(run_dir, friends_dir, host_label)  # copy summary csv into FRIENDS

            if copied:
                final_csv_path = str((friends_dir / f"{host_label}.csv").resolve())  # record final csv path
                print(f"  Saved CSV -> {final_csv_path}")

            try:
                shutil.rmtree(run_dir)  # delete temp *_friends folder after csv was copied
                print(f"  Deleted temp folder -> {run_dir}")
            except Exception as e:
                print(f"  WARNING: Could not delete temp folder {run_dir}: {e}")

        if stats is None:
            stats = {
                "distance_pc": "",
                "search_radius_deg": "",
                "min_parallax_mas": "",
                "parallax_error_cut_mas": "",
                "n_query_rows": "",
                "n_base": "",
                "n_not_self": "",
                "n_convergcut": "",
                "n_cmd_clean": "",
                "n_good_rv": "",
                "n_rv_comoving": "",
                "n_rv_outlier": "",
                "n_final": "",
            }  # blank stats if worker failed before returning diagnostics

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f'{idx},{host_label},{gaia_id},{radvel},"{query_timestamp}",{attempt_completed_on},'
                f'{float(srad):.6f},{float(srad_eff):.6f},{ok},{runtime:.2f},"{final_csv_path}",'
                f'{stats.get("distance_pc","")},{stats.get("search_radius_deg","")},'
                f'{stats.get("min_parallax_mas","")},{stats.get("parallax_error_cut_mas","")},'
                f'{stats.get("n_query_rows","")},{stats.get("n_base","")},{stats.get("n_not_self","")},'
                f'{stats.get("n_convergcut","")},{stats.get("n_cmd_clean","")},{stats.get("n_good_rv","")},'
                f'{stats.get("n_rv_comoving","")},{stats.get("n_rv_outlier","")},{stats.get("n_final","")},'
                f'"{err}"\n'
            )  # append one row to the run log

        time.sleep(float(sleep_between_targets_s))  # small pause between targets


def run_stars(**kwargs):
    return run(**kwargs)  # convenience wrapper using your preferred function name