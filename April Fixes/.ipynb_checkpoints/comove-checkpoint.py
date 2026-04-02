from __future__ import annotations  # allow forward-looking type hints / cleaner annotations

import csv  # for optional LocalRV.csv overrides
import math  # trig / angle conversions / safety caps
import os  # folder creation / path checks
import warnings  # suppress non-fatal warnings cleanly

import galpy.util.coords as bc  # coordinate and velocity transforms
import numpy as np  # vectorized math / masks / arrays
import pandas as pd  # dataframe writing for summary csv
from astropy import units as u  # physical units
from astropy.coordinates import SkyCoord  # coordinate containers / separations
from astropy.utils.data import conf  # remote timeout control
from astroquery.gaia import Gaia  # Gaia TAP querying
from astroquery.simbad import Simbad  # SIMBAD fallback name resolution


Simbad.reset_votable_fields()  # clear any previously added SIMBAD fields
Simbad.TIMEOUT = 300  # allow SIMBAD some time before timing out
Simbad.server = "simbad.harvard.edu"  # pick the preferred SIMBAD server

conf.remote_timeout = 120.0  # astropy remote timeout for network-backed calls


SUMMARY_COLUMNS = [
    "Catalog",
    "Type",
    "GaiaDR3",
    "RA",
    "DEC",
    "Gmag",
    "Bp-Rp",
    "Voff(km/s)",
    "Sep(deg)",
    "3D(pc)",
    "Vr(pred)",
    "Vr(obs)",
    "Vrerr",
    "Plx(mas)",
    "SpT",
    "FnuvJ",
    "W1-W3",
    "RUWE",
    "XCrate",
    "RVsrc",
    "PMRApred",
    "PMDecpred",
    "PMRA",
    "PMRAerr",
    "PMDec",
    "PMDecerr",
]  # this is the only csv product we write now


def _masked_nanargmin(masked_arr):
    arr = np.ma.array(masked_arr)  # force masked-array behavior even if input is plain ndarray
    if arr.count() == 0:
        return None  # no valid values exist
    return int(np.ma.argmin(arr))  # index of smallest valid entry


def _designation_parts(designation):
    parts = str(designation).split(maxsplit=2)  # split designation into at most 3 chunks
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]  # catalog, type, id
    if len(parts) == 2:
        return parts[0], parts[1], ""  # missing final id chunk
    if len(parts) == 1:
        return "", "", parts[0]  # only one chunk, treat it like the id
    return "", "", ""  # fully empty fallback


def _mk_outdir(targname: str, output_directory: str | None) -> str:
    if output_directory is None:
        outdir = "./" + targname.replace(" ", "") + "_friends/"  # default folder naming scheme
    else:
        outdir = output_directory  # caller supplied an explicit folder

    if os.path.isdir(outdir):
        raise FileExistsError(
            f"Output directory already exists: {outdir}. Move/delete it or choose another."
        )  # avoid silently overwriting previous runs

    os.mkdir(outdir)  # create the output directory
    return outdir  # return created path


def _simbad_radec_deg(targname: str):
    result_table = Simbad.query_object(targname)  # ask SIMBAD to resolve the target name
    if result_table is None or len(result_table) == 0:
        raise RuntimeError(f"SIMBAD returned no coordinates for {targname!r}")  # fail loudly if not found

    ra_str = result_table["ra"][0]  # SIMBAD RA string
    dec_str = result_table["dec"][0]  # SIMBAD Dec string

    c = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")  # parse into coordinate object
    return float(c.ra.deg), float(c.dec.deg)  # return decimal degrees


def _gaia_target_by_source_id(source_id: int):
    sql = f"""
    SELECT
        source_id, designation, ra, dec,
        phot_g_mean_mag,
        parallax, parallax_error,
        pmra, pmdec, pmra_error, pmdec_error,
        radial_velocity, radial_velocity_error,
        ruwe,
        bp_rp, phot_rp_mean_mag, phot_bp_rp_excess_factor,
        teff_gspphot, grvs_mag
    FROM gaiadr3.gaia_source
    WHERE source_id = {int(source_id)}
    """  # exact target lookup by Gaia source id

    job = Gaia.launch_job_async(sql, dump_to_file=False)  # async is fine here too
    r = job.get_results()  # fetch results
    if len(r) == 0:
        raise RuntimeError(f"No Gaia row found for source_id={source_id}")  # invalid or missing source id
    return r[0]  # return the only expected row


def _gaia_target_by_cone(ra_deg: float, dec_deg: float, radius_arcsec: float = 6.0):
    sql = f"""
    SELECT
        source_id, designation, ra, dec,
        phot_g_mean_mag,
        parallax, parallax_error,
        pmra, pmdec, pmra_error, pmdec_error,
        radial_velocity, radial_velocity_error,
        ruwe,
        bp_rp, phot_rp_mean_mag, phot_bp_rp_excess_factor,
        teff_gspphot, grvs_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_arcsec/3600.0})
    ) = 1
    """  # small cone around input coordinates to identify the target row

    job = Gaia.launch_job_async(sql, dump_to_file=False)  # run query
    r = job.get_results()  # fetch returned rows

    if len(r) == 0:
        raise RuntimeError("Gaia target cone search returned 0 rows")  # target not found in Gaia near those coords

    minpos = _masked_nanargmin(r["phot_g_mean_mag"])  # choose brightest source in the tiny cone
    if minpos is None:
        raise RuntimeError("Gaia target cone search had no valid G mags")  # no usable target row

    return r[minpos]  # return best target row


def _table_has_columns(table_name: str, wanted_cols: list[str]) -> bool:
    query = f"""
    SELECT column_name
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{table_name}'
    """  # inspect Gaia TAP schema to see which columns are present

    try:
        job = Gaia.launch_job(query, dump_to_file=False)  # schema lookup can be synchronous
        r = job.get_results()  # get schema rows
        have = {str(x["column_name"]) for x in r}  # build set of available column names
        return all(c in have for c in wanted_cols)  # confirm every requested column is available
    except Exception:
        return False  # if schema lookup fails, just fall back to full table


def _fetch_neighbors_lite(center_ra_deg: float, center_dec_deg: float, searchraddeg: float, minpar_mas: float, plxerr_cut: float):
    lite_cols = [
        "source_id", "ra", "dec", "parallax", "parallax_error",
        "pmra", "pmdec", "pmra_error", "pmdec_error",
        "phot_g_mean_mag", "bp_rp", "ruwe"
    ]  # only columns needed for first-pass candidate finding

    use_lite = _table_has_columns("gaiadr3.gaia_source_lite", lite_cols)  # check whether lite table can satisfy this query
    table_name = "gaiadr3.gaia_source_lite" if use_lite else "gaiadr3.gaia_source"  # prefer lite if possible

    sql = f"""
    SELECT {", ".join(lite_cols)}
    FROM {table_name}
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center_ra_deg}, {center_dec_deg}, {searchraddeg})
    ) = 1
      AND parallax > {minpar_mas}
      AND parallax_error < {plxerr_cut}
    """  # first-pass local cone search with parallax filters

    job = Gaia.launch_job_async(sql, dump_to_file=False)  # heavy neighbor query, so async
    return job.get_results()  # return first-pass results


def _fetch_full_by_source_ids(source_ids: np.ndarray):
    if len(source_ids) == 0:
        return pd.DataFrame(columns=[
            "source_id", "designation", "phot_rp_mean_mag", "phot_bp_rp_excess_factor",
            "radial_velocity", "radial_velocity_error", "teff_gspphot", "grvs_mag"
        ])  # empty enrichment table if there are no rows

    chunks = []  # store enrichment chunks
    src = [str(int(x)) for x in source_ids]  # stringify source ids for SQL

    for i in range(0, len(src), 4000):
        sub = src[i:i+4000]  # moderate IN-clause chunk size
        id_list = ",".join(sub)  # comma-separated source ids

        sql = f"""
        SELECT
            source_id, designation,
            phot_rp_mean_mag, phot_bp_rp_excess_factor,
            radial_velocity, radial_velocity_error,
            teff_gspphot, grvs_mag
        FROM gaiadr3.gaia_source
        WHERE source_id IN ({id_list})
        """  # second-pass enrichment query only on rows we already kept

        job = Gaia.launch_job_async(sql, dump_to_file=False)  # run chunk query
        r = job.get_results()  # fetch chunk
        chunks.append(r.to_pandas())  # convert to pandas for merge

    if not chunks:
        return pd.DataFrame()  # defensive fallback

    return pd.concat(chunks, ignore_index=True)  # merge all chunks


def _build_summary_df(df: pd.DataFrame):
    rows = []  # accumulate rows in legacy output format

    for _, row in df.iterrows():
        catalog, kind, gaia_id = _designation_parts(row.get("designation", ""))  # split designation for old-style columns

        rows.append({
            "Catalog": catalog,
            "Type": kind,
            "GaiaDR3": gaia_id if gaia_id else str(int(row["source_id"])) if pd.notna(row["source_id"]) else "",
            "RA": row["ra"],
            "DEC": row["dec"],
            "Gmag": row["phot_g_mean_mag"],
            "Bp-Rp": row["bp_rp"],
            "Voff(km/s)": row["voff_kms"],
            "Sep(deg)": row["sep_deg"],
            "3D(pc)": row["sep3d_pc"],
            "Vr(pred)": row["vr_pred"],
            "Vr(obs)": row["vr_obs"],
            "Vrerr": row["vr_err"],
            "Plx(mas)": row["parallax"],
            "SpT": "nan",  # placeholder retained for compatibility with your old csv structure
            "FnuvJ": np.nan,  # placeholder retained for compatibility
            "W1-W3": np.nan,  # placeholder retained for compatibility
            "RUWE": row["ruwe"],
            "XCrate": np.nan,  # placeholder retained for compatibility
            "RVsrc": row["rvsrc"],
            "PMRApred": row["pmra_pred"],
            "PMDecpred": row["pmdec_pred"],
            "PMRA": row["pmra"],
            "PMRAerr": row["pmra_error"],
            "PMDec": row["pmdec"],
            "PMDecerr": row["pmdec_error"],
        })  # append one final candidate row

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)  # dataframe in fixed desired order


def _write_outputs(outdir: str, targname: str, summary_df: pd.DataFrame):
    base = os.path.join(outdir, targname.replace(" ", ""))  # base output path
    summary_path = base + ".csv"  # only output file now
    summary_df.to_csv(summary_path, index=False)  # write summary csv only


def findfriends(
    targname,
    radial_velocity,
    velocity_limit=5.0,
    search_radius=25.0,
    rvcut=5.0,
    convergcut=5.0,
    radec=[None, None],
    gaia_source_id=None,
    output_directory=None,
    showplots=False,
    verbose=False,
    DoGALEX=False,
    DoWISE=False,
    DoROSAT=False,
    force_max_angle_deg=45.0,
):
    del showplots, DoGALEX, DoWISE, DoROSAT  # explicitly unused in this search-only version

    warnings.filterwarnings("ignore", category=UserWarning)  # suppress non-fatal pandas / astropy style warnings

    outdir = _mk_outdir(str(targname), output_directory)  # create output directory first

    radvel = float(radial_velocity) * u.km / u.s  # target radial velocity with units
    vlim = float(velocity_limit) * u.km / u.s  # tangential velocity cut with units
    searchradpc = float(search_radius) * u.pc  # requested physical search radius
    convergcut = 0.0 if convergcut is None else float(convergcut)  # normalize None to zero

    if gaia_source_id is not None:
        target_row = _gaia_target_by_source_id(int(gaia_source_id))  # exact target fetch if Gaia id is known
    else:
        if (radec[0] is not None) and (radec[1] is not None):
            ra_deg = float(radec[0])  # use caller-provided RA
            dec_deg = float(radec[1])  # use caller-provided Dec
        else:
            if verbose:
                print("Asking SIMBAD for RA/DEC")  # note fallback path
            ra_deg, dec_deg = _simbad_radec_deg(str(targname))  # resolve target name if coordinates absent

        target_row = _gaia_target_by_cone(ra_deg, dec_deg)  # get target row by small cone if source id absent

    if np.ma.is_masked(target_row["parallax"]) or target_row["parallax"] is None or float(target_row["parallax"]) <= 0:
        raise RuntimeError(f"Target {targname!r} has invalid/non-positive parallax")  # cannot do 3D search with bad distance

    Pcoord = SkyCoord(
        ra=float(target_row["ra"]) * u.deg,
        dec=float(target_row["dec"]) * u.deg,
        distance=(1000.0 / float(target_row["parallax"])) * u.pc,
        frame="icrs",
        radial_velocity=radvel,
        pm_ra_cosdec=float(target_row["pmra"]) * u.mas / u.yr,
        pm_dec=float(target_row["pmdec"]) * u.mas / u.yr,
    )  # target coordinate object with distance, PM, and RV

    d_pc = float(Pcoord.distance.to_value(u.pc))  # target distance in pc
    angle_cap_pc = d_pc * math.sin(math.radians(float(force_max_angle_deg)))  # max physical radius implied by angle cap
    srad_eff_pc = min(float(searchradpc.value), 0.95 * d_pc, angle_cap_pc)  # enforce safe cap to avoid all-sky searches

    if srad_eff_pc <= 0:
        raise RuntimeError("Effective search radius became non-positive")  # sanity guard

    if verbose and srad_eff_pc < float(searchradpc.value):
        print(f"Search radius capped: {searchradpc.value:.3f} pc -> {srad_eff_pc:.3f} pc")  # tell user when cap changed radius

    searchradpc = srad_eff_pc * u.pc  # replace requested radius with safe effective radius
    ratio = min(max(srad_eff_pc / d_pc, 0.0), 0.999999)  # clamp arcsin input safely
    searchraddeg = math.degrees(math.asin(ratio))  # convert physical radius to angular cone radius
    minpar = (1000.0 / (d_pc + srad_eff_pc))  # nearest allowed parallax bound for stars inside sphere
    Pllbb = bc.radec_to_lb(Pcoord.ra.value, Pcoord.dec.value, degree=True)  # target galactic lon/lat

    if abs(Pllbb[1]) > 10.0:
        plxcut = max(0.5, (1000.0 / d_pc / 10.0))  # off-plane, allow scaled parallax error cut
    else:
        plxcut = 0.5  # near plane, use baseline strict cut

    if verbose:
        print(f"Target Gaia source_id: {int(target_row['source_id'])}")  # show exact target row selected
        print(f"Distance: {d_pc:.3f} pc")  # show target distance
        print(f"searchraddeg = {searchraddeg:.4f} deg")  # show angular radius
        print(f"minpar      = {minpar:.4f} mas")  # show parallax floor
        print(f"plxerr cut  = {plxcut:.4f} mas")  # show parallax-error ceiling

    lite = _fetch_neighbors_lite(
        center_ra_deg=Pcoord.ra.value,
        center_dec_deg=Pcoord.dec.value,
        searchraddeg=searchraddeg,
        minpar_mas=minpar,
        plxerr_cut=plxcut,
    )  # first-pass query for local neighbors

    if len(lite) == 0:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)  # empty summary if no rows returned
        _write_outputs(outdir, targname, summary_df)  # still write output csv so caller gets expected file
        stats = {
            "target_source_id": int(target_row["source_id"]),
            "distance_pc": d_pc,
            "search_radius_input_pc": float(search_radius),
            "search_radius_used_pc": float(srad_eff_pc),
            "search_radius_deg": float(searchraddeg),
            "min_parallax_mas": float(minpar),
            "parallax_error_cut_mas": float(plxcut),
            "n_query_rows": 0,
            "n_base": 0,
            "n_not_self": 0,
            "n_convergcut": 0,
            "n_cmd_clean": 0,
            "n_good_rv": 0,
            "n_rv_comoving": 0,
            "n_rv_outlier": 0,
            "n_final": 0,
        }  # empty stats bundle for logging
        return outdir, stats  # return normal output path plus stats

    lite_df = lite.to_pandas()  # convert first-pass results to dataframe
    enrich_df = _fetch_full_by_source_ids(lite_df["source_id"].to_numpy())  # get extra columns only for those rows
    df = lite_df.merge(enrich_df, on="source_id", how="left", suffixes=("", "_full"))  # combine first-pass and enrichment data

    if "designation_full" in df.columns:
        df["designation"] = df["designation_full"].combine_first(df.get("designation"))  # prefer enriched designation if present
        df.drop(columns=["designation_full"], inplace=True)  # drop temp merged column

    for col in ["phot_rp_mean_mag", "phot_bp_rp_excess_factor", "radial_velocity",
                "radial_velocity_error", "teff_gspphot", "grvs_mag"]:
        if f"{col}_full" in df.columns:
            df[col] = df[f"{col}_full"].combine_first(df.get(col))  # prefer enriched values when available
            df.drop(columns=[f"{col}_full"], inplace=True)  # clean up temp merged column

    gaiacoord = SkyCoord(
        ra=np.asarray(df["ra"], dtype=float) * u.deg,
        dec=np.asarray(df["dec"], dtype=float) * u.deg,
        distance=(1000.0 / np.asarray(df["parallax"], dtype=float)) * u.pc,
        frame="icrs",
        pm_ra_cosdec=np.asarray(df["pmra"], dtype=float) * u.mas / u.yr,
        pm_dec=np.asarray(df["pmdec"], dtype=float) * u.mas / u.yr,
    )  # coordinate object for all returned neighbors

    sep = gaiacoord.separation(Pcoord)  # on-sky separations from target
    sep3d = gaiacoord.separation_3d(Pcoord)  # 3D separations from target

    Ppmllpmbb = bc.pmrapmdec_to_pmllpmbb(
        Pcoord.pm_ra_cosdec.value,
        Pcoord.pm_dec.value,
        Pcoord.ra.value,
        Pcoord.dec.value,
        degree=True,
    )  # target proper motion converted into galactic components

    Pvxvyvz = bc.vrpmllpmbb_to_vxvyvz(
        Pcoord.radial_velocity.value,
        Ppmllpmbb[0],
        Ppmllpmbb[1],
        Pllbb[0],
        Pllbb[1],
        Pcoord.distance.value / 1000.0,
        XYZ=False,
        degree=True,
    )  # target space velocity vector

    Cll = (math.degrees(math.atan2(Pvxvyvz[1], Pvxvyvz[0]))) % 360.0  # convergent point galactic longitude
    Cbb = math.degrees(math.atan2(Pvxvyvz[2], math.sqrt(Pvxvyvz[0]**2 + Pvxvyvz[1]**2)))  # convergent point galactic latitude
    Cradec = bc.lb_to_radec(Cll, Cbb, degree=True, epoch=2000.0)  # back to RA/Dec
    Ccoord = SkyCoord(ra=Cradec[0] * u.deg, dec=Cradec[1] * u.deg, distance=999999.9 * u.pc, frame="icrs")  # distant fake coordinate for angle calc

    Cangle = gaiacoord.separation(Ccoord)  # angle from each star to convergent point
    flip = np.where(Cangle.degree > 90.0)[0]  # reflect beyond 90 deg to symmetric side

    if flip.size > 0:
        ctmp = Cangle.degree.copy()  # plain ndarray copy
        ctmp[flip] = 180.0 - ctmp[flip]  # reflect values into [0, 90]
        cangle_deg = ctmp  # final convergent-angle array
    else:
        cangle_deg = Cangle.degree  # no reflection needed

    Gllbb = bc.radec_to_lb(gaiacoord.ra.value, gaiacoord.dec.value, degree=True)  # neighbor galactic coords
    Gxyz = bc.lbd_to_XYZ(Gllbb[:, 0], Gllbb[:, 1], gaiacoord.distance.value / 1000.0, degree=True)  # neighbor XYZ positions
    Gvrpmllpmbb = bc.vxvyvz_to_vrpmllpmbb(
        Pvxvyvz[0] * np.ones(len(Gxyz[:, 0])),
        Pvxvyvz[1] * np.ones(len(Gxyz[:, 1])),
        Pvxvyvz[2] * np.ones(len(Gxyz[:, 2])),
        Gxyz[:, 0], Gxyz[:, 1], Gxyz[:, 2],
        XYZ=True,
    )  # predicted RV + PM components if each star were truly comoving

    Gpmrapmdec = bc.pmllpmbb_to_pmrapmdec(
        Gvrpmllpmbb[:, 1],
        Gvrpmllpmbb[:, 2],
        Gllbb[:, 0],
        Gllbb[:, 1],
        degree=True,
    )  # convert predicted PM into equatorial components

    Gvtanerr = np.ones(len(df))  # assume 1 km/s tangential error scale
    Gpmerr = Gvtanerr * 206265000.0 * 3.154e7 / (gaiacoord.distance.value * 3.086e13)  # convert that to PM uncertainty scale
    Gchi2 = np.sqrt(
        (Gpmrapmdec[:, 0] - gaiacoord.pm_ra_cosdec.value) ** 2
        + (Gpmrapmdec[:, 1] - gaiacoord.pm_dec.value) ** 2
    ) / Gpmerr  # tangential velocity mismatch in km/s-like units

    sep_deg = sep.degree  # angular separation in degrees
    sep3d_pc = sep3d.value  # 3D separation in pc

    bp_rp = np.asarray(df["bp_rp"], dtype=float)  # Gaia color
    ruwe = np.asarray(df["ruwe"], dtype=float)  # RUWE quality metric

    phot_excess = np.asarray(df["phot_bp_rp_excess_factor"], dtype=float)  # Gaia BP/RP excess
    phot_excess = np.where(np.isfinite(phot_excess), phot_excess, np.inf)  # missing values should fail cmd-clean cut

    base_mask = (sep3d_pc < srad_eff_pc) & (Gchi2 < vlim.value)  # core candidate cut: spatial + tangential mismatch
    not_self_mask = sep_deg > 1e-5  # reject the target itself
    conv_mask = cangle_deg > convergcut  # convergent-angle cut
    finite_bprp_mask = np.isfinite(bp_rp)  # require finite color
    cmd_clean_mask = finite_bprp_mask & (phot_excess < (1.3 + 0.06 * bp_rp**2))  # Gaia CMD-clean cut

    RV = np.full(len(df), np.nan)  # observed RV array
    RVerr = np.full(len(df), np.nan)  # RV uncertainty array
    RVsrc = np.array(["None"] * len(df), dtype=object)  # provenance label array

    rv_col = np.asarray(df["radial_velocity"], dtype=float)  # Gaia RV column
    rverr_col = np.asarray(df["radial_velocity_error"], dtype=float)  # Gaia RV error column
    teff_col = np.asarray(df["teff_gspphot"], dtype=float)  # Teff for hot-star correction
    grvs_col = np.asarray(df["grvs_mag"], dtype=float)  # GRVS mag for hot-star correction
    rp_col = np.asarray(df["phot_rp_mean_mag"], dtype=float)  # RP mag fallback for hot-star correction

    cand = np.where(base_mask)[0]  # rows passing base candidate cut
    if cand.size > 0:
        cand = cand[np.argsort(sep3d_pc[cand])]  # sort by spatial separation

    for idx in cand:
        if np.isfinite(rv_col[idx]):
            RV[idx] = rv_col[idx]  # start from Gaia DR3 RV
            if np.isfinite(teff_col[idx]) and teff_col[idx] >= 8500.0:
                if np.isfinite(grvs_col[idx]):
                    RV[idx] = rv_col[idx] - (7.98 - 1.135 * grvs_col[idx])  # hot-star correction using GRVS
                elif np.isfinite(rp_col[idx]):
                    RV[idx] = rv_col[idx] - (7.98 - 1.135 * rp_col[idx])  # fallback hot-star correction using RP
            RVerr[idx] = rverr_col[idx]  # store uncertainty
            RVsrc[idx] = "Gaia_DR3"  # mark provenance

    if os.path.isfile("LocalRV.csv"):
        with open("LocalRV.csv", newline="", encoding="utf-8") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")  # local override RV file
            next(readCSV, None)  # skip header if present
            for row in readCSV:
                ww = np.where(df["designation"].astype(str).to_numpy() == row[0])[0]  # match by designation
                if ww.size == 1 and (not np.isfinite(RVerr[ww[0]]) or RVerr[ww[0]] > float(row[3])):
                    RV[ww[0]] = float(row[2])  # replace with better external RV
                    RVerr[ww[0]] = float(row[3])  # replace uncertainty
                    RVsrc[ww[0]] = row[4]  # provenance label

    has_good_rv = np.isfinite(RV) & np.isfinite(RVerr) & (RVerr > 0)  # stars with usable observed RV
    rv_delta = np.abs(RV - Gvrpmllpmbb[:, 0])  # observed minus predicted RV
    is_rv_outlier = has_good_rv & (rv_delta > rvcut) & ((rv_delta / RVerr) > 2.0)  # clearly inconsistent RVs
    is_rv_comoving = has_good_rv & (rv_delta <= rvcut)  # RV-compatible stars

    candidate_mask = base_mask & not_self_mask & conv_mask  # final candidate definition

    target_sid = int(target_row["source_id"])  # target Gaia DR3 id
    is_target = df["source_id"].astype("int64").to_numpy() == target_sid  # identify target row if present

    n_query_rows = int(len(df))  # total Gaia rows returned by local query
    n_base = int(np.sum(base_mask))  # rows passing base spatial + tangential cut
    n_not_self = int(np.sum(base_mask & not_self_mask))  # rows after removing target itself
    n_convergcut = int(np.sum(base_mask & not_self_mask & conv_mask))  # rows after convergent-angle cut
    n_cmd_clean = int(np.sum(base_mask & not_self_mask & conv_mask & cmd_clean_mask))  # rows that also pass Gaia CMD-clean cut
    n_good_rv = int(np.sum(has_good_rv))  # rows with usable RV information
    n_rv_comoving = int(np.sum(is_rv_comoving))  # rows RV-consistent with comoving prediction
    n_rv_outlier = int(np.sum(is_rv_outlier))  # rows RV-inconsistent with comoving prediction
    n_final = int(np.sum(candidate_mask))  # final output candidate count

    df["sep_deg"] = sep_deg  # store separations for output building
    df["sep3d_pc"] = sep3d_pc
    df["cangle_deg"] = cangle_deg
    df["voff_kms"] = Gchi2
    df["vr_pred"] = Gvrpmllpmbb[:, 0]
    df["vr_obs"] = RV
    df["vr_err"] = RVerr
    df["rvsrc"] = RVsrc
    df["pmra_pred"] = Gpmrapmdec[:, 0]
    df["pmdec_pred"] = Gpmrapmdec[:, 1]
    df["is_target"] = is_target
    df["is_candidate"] = candidate_mask  # only this one is actually used to form final summary table

    summary_source = df[df["is_candidate"]].copy()  # final candidate subset only
    summary_df = _build_summary_df(summary_source)  # build old-style summary table

    _write_outputs(outdir, targname, summary_df)  # write only the summary csv

    stats = {
        "target_source_id": int(target_row["source_id"]),
        "distance_pc": float(d_pc),
        "search_radius_input_pc": float(search_radius),
        "search_radius_used_pc": float(srad_eff_pc),
        "search_radius_deg": float(searchraddeg),
        "min_parallax_mas": float(minpar),
        "parallax_error_cut_mas": float(plxcut),
        "n_query_rows": n_query_rows,
        "n_base": n_base,
        "n_not_self": n_not_self,
        "n_convergcut": n_convergcut,
        "n_cmd_clean": n_cmd_clean,
        "n_good_rv": n_good_rv,
        "n_rv_comoving": n_rv_comoving,
        "n_rv_outlier": n_rv_outlier,
        "n_final": n_final,
    }  # structured diagnostic stats for logging in run_stars.py

    if verbose:
        print(f"Wrote summary CSV to {outdir}")  # completion message
        print(stats)  # print stats if verbose so you can inspect filtering behavior immediately

    return outdir, stats  # return output folder plus logging stats