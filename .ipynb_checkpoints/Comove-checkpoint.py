# import pkg_resources
# datapath = pkg_resources.resource_filename('Comove', 'resources')
datapath = './resources'

import csv
import math
import os
import warnings

import galpy.util.coords as bc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import conf
from astroquery.exceptions import NoResultsWarning
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier


print('DEBUG 0')
Simbad.reset_votable_fields()
Simbad.TIMEOUT = 1500
Simbad.server = "simbad.harvard.edu"
customSimbad = Simbad()
customSimbad.add_votable_fields('rvz_radvel', 'rvz_err', 'rvz_bibcode')
Irsa.TIMEOUT = 600
conf.remote_timeout = 60.0
Vizier.TIMEOUT = 600
print('DEBUG 1')


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['axes.labelweight'] = 'semibold'
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.titleweight'] = 'semibold'
mpl.rcParams['font.weight'] = 'semibold'
plt.rcParams['figure.facecolor'] = 'white'


HEADER_TXT = (
    "GaiaDR3                               RA         DEC   Gmag  Bp-Rp  Voff(km/s) Sep(deg)   3D(pc) "
    "Vr(pred)  Vr(obs)    Vrerr Plx(mas)  SpT    FnuvJ  W1-W3    RUWE  XCrate                               "
    "RVsrc    PMRApred   PMDecpred        PMRA     PMRAerr       PMDec    PMDecerr"
)

CSV_COLUMNS = [
    'Catalog', 'Type', 'GaiaDR3', 'RA', 'DEC', 'Gmag', 'Bp-Rp', 'Voff(km/s)', 'Sep(deg)', '3D(pc)',
    'Vr(pred)', 'Vr(obs)', 'Vrerr', 'Plx(mas)', 'SpT', 'FnuvJ', 'W1-W3', 'RUWE', 'XCrate',
    'RVsrc', 'PMRApred', 'PMDecpred', 'PMRA', 'PMRAerr', 'PMDec', 'PMDecerr'
]

FMT_ROW = (
    "%28s %11.7f %11.7f %6.3f %6.3f %11.3f %8.4f %8.4f %8.2f %8.2f %8.2f %8.3f %4s %8.6f %6.2f "
    "%7.3f %7.3f %35s %11.3f %11.3f %11.3f %11.3f %11.3f %11.3f"
)


def _load_cmd_resources(basepath):
    names = {
        'mamajek': 'sptGBpRp.txt',
        'pleiades': 'PleGBpRp.txt',
        'tuchor': 'TucGBpRp.txt',
        'usco': 'UScGBpRp.txt',
        'chai': 'ChaGBpRp.txt',
    }
    out = {}
    for key, fname in names.items():
        fpath = os.path.join(basepath, fname)
        try:
            out[key] = np.loadtxt(fpath)
        except Exception as exc:
            print(f"DEBUG: Failed to load {fpath} ({type(exc).__name__}: {exc})")
            out[key] = None
    return out


CMD_RESOURCES = _load_cmd_resources(datapath)


def _designation_parts(designation):
    parts = str(designation).split(maxsplit=2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ''
    if len(parts) == 1:
        return '', '', parts[0]
    return '', '', ''


def _rows_to_dataframe(rows):
    parsed = []
    for row in rows:
        catalog, kind, gaia_id = _designation_parts(row['designation'])
        parsed.append([
            catalog,
            kind,
            gaia_id,
            row['ra'],
            row['dec'],
            row['gmag'],
            row['bp_rp'],
            row['voff'],
            row['sep_deg'],
            row['sep3d_pc'],
            row['vr_pred'],
            row['vr_obs'],
            row['vr_err'],
            row['plx_mas'],
            row['spt'],
            row['fnuvj'],
            row['w1_w3'],
            row['ruwe'],
            row['xcrate'],
            row['rvsrc'],
            row['pmra_pred'],
            row['pmdec_pred'],
            row['pmra'],
            row['pmra_err'],
            row['pmdec'],
            row['pmdec_err'],
        ])
    return pd.DataFrame(parsed, columns=CSV_COLUMNS)


def _write_outputs(outdir, targname, rows):
    base = os.path.join(outdir, targname.replace(' ', ''))
    txt_filename = base + '.txt'
    csv_filename = base + '.csv'

    with open(txt_filename, 'w') as f:
        f.write(HEADER_TXT + '\n')
        for row in rows:
            f.write(FMT_ROW % (
                row['designation'],
                row['ra'],
                row['dec'],
                row['gmag'],
                row['bp_rp'],
                row['voff'],
                row['sep_deg'],
                row['sep3d_pc'],
                row['vr_pred'],
                row['vr_obs'],
                row['vr_err'],
                row['plx_mas'],
                row['spt'],
                row['fnuvj'],
                row['w1_w3'],
                row['ruwe'],
                row['xcrate'],
                row['rvsrc'],
                row['pmra_pred'],
                row['pmdec_pred'],
                row['pmra'],
                row['pmra_err'],
                row['pmdec'],
                row['pmdec_err'],
            ))
            f.write('\n')

    df = _rows_to_dataframe(rows)
    df.to_csv(csv_filename, index=False)


def _write_empty_outputs(outdir, targname):
    _write_outputs(outdir, targname, [])


def _masked_nanargmin(masked_arr):
    arr = np.ma.array(masked_arr)
    if arr.count() == 0:
        return None
    return int(np.ma.argmin(arr))


def _save_or_skip(fig, fname, showplots=False):
    try:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.2, dpi=200)
        if showplots:
            plt.show()
    finally:
        plt.close(fig)


def findfriends(
    targname,
    radial_velocity,
    velocity_limit=5.0,
    search_radius=25.0,
    rvcut=5.0,
    convergcut=5.0,
    radec=[None, None],
    output_directory=None,
    showplots=False,
    verbose=False,
    DoGALEX=False,
    DoWISE=False,
    DoROSAT=False,
):
    radvel = radial_velocity * u.kilometer / u.second
    if convergcut is None:
        convergcut = 0.0

    if output_directory is None:
        outdir = './' + targname.replace(' ', '') + '_friends/'
    else:
        outdir = output_directory

    if os.path.isdir(outdir):
        print('Output directory ' + outdir + ' Already Exists!!')
        print('Either Move it, Delete it, or input a different [output_directory] Please!')
        return
    os.mkdir(outdir)

    if velocity_limit < 1e-5:
        print('input velocity_limit is too small, try something else')
        print('velocity_limit: ' + str(velocity_limit))
    if search_radius < 1e-7:
        print('input search_radius is too small, try something else')
        print('search_radius: ' + str(search_radius))

    vlim = velocity_limit * u.kilometer / u.second
    searchradpc = search_radius * u.parsec

    if (radec[0] is not None) and (radec[1] is not None):
        usera, usedec = radec[0], radec[1]
    else:
        print('Asking Simbad for RA and DEC')
        result_table = Simbad.query_object(targname)
        usera, usedec = result_table['ra'][0], result_table['dec'][0]

    if verbose:
        print('Target name: ', targname)
        print('Coordinates: ' + str(usera) + ' ' + str(usedec))
        print()

    c = SkyCoord(ra=usera, dec=usedec, unit=(u.deg, u.deg), frame='icrs')
    if verbose:
        print(c)

    print('Asking Gaia for precise coordinates')
    target_sql = f"""
    SELECT
        source_id, ra, dec,
        phot_g_mean_mag,
        parallax,
        pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
        CIRCLE('ICRS', {c.ra.value}, {c.dec.value}, {6.0/3600.0})
    )=1
    """
    target_job = Gaia.launch_job(target_sql, dump_to_file=False)
    Pgaia = target_job.get_results()

    if len(Pgaia) == 0:
        print('DEBUG: Gaia target cone search returned 0 rows. Writing empty outputs.')
        _write_empty_outputs(outdir, targname)
        return outdir

    minpos = _masked_nanargmin(Pgaia['phot_g_mean_mag'])
    if minpos is None:
        print('DEBUG: Gaia target cone search had no valid G mags. Writing empty outputs.')
        _write_empty_outputs(outdir, targname)
        return outdir

    Pcoord = SkyCoord(
        ra=Pgaia['ra'][minpos] * u.deg,
        dec=Pgaia['dec'][minpos] * u.deg,
        distance=(1000.0 / Pgaia['parallax'][minpos]) * u.parsec,
        frame='icrs',
        radial_velocity=radvel,
        pm_ra_cosdec=Pgaia['pmra'][minpos] * u.mas / u.year,
        pm_dec=Pgaia['pmdec'][minpos] * u.mas / u.year,
    )

    searchraddeg = np.arcsin(searchradpc / Pcoord.distance).to(u.deg)
    minpar = (1000.0 * u.parsec) / (Pcoord.distance + searchradpc) * u.mas

    if verbose:
        print(Pcoord)
        print('Search radius in deg: ', searchraddeg)
        print('Minimum parallax: ', minpar)

    print('Querying Gaia for neighbors')
    Pllbb = bc.radec_to_lb(Pcoord.ra.value, Pcoord.dec.value, degree=True)
    if np.abs(Pllbb[1]) > 10.0:
        plxcut = max(0.5, (1000.0 / Pcoord.distance.value / 10.0))
    else:
        plxcut = 0.5
    print('Parallax cut: ', plxcut)

    neighbor_cols = """
        source_id, designation,
        ra, dec,
        parallax, parallax_error,
        pmra, pmdec, pmra_error, pmdec_error,
        phot_g_mean_mag, phot_rp_mean_mag, bp_rp,
        phot_bp_rp_excess_factor,
        ruwe,
        radial_velocity, radial_velocity_error,
        teff_gspphot, grvs_mag
    """

    if searchradpc < Pcoord.distance:
        neighbor_sql = f"""
        SELECT {neighbor_cols}
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
            CIRCLE('ICRS', {Pcoord.ra.value}, {Pcoord.dec.value}, {searchraddeg.value})
        ) = 1
        AND parallax > {minpar.value}
        AND parallax_error < {plxcut}
        """
    else:
        neighbor_sql = f"""
        SELECT {neighbor_cols}
        FROM gaiadr3.gaia_source
        WHERE parallax > {minpar.value}
        AND parallax_error < {plxcut}
        """
        print('Note, using all-sky search')

    neighbor_job = Gaia.launch_job_async(neighbor_sql, dump_to_file=False)
    r = neighbor_job.get_results()

    if len(r) == 0:
        print('DEBUG: Gaia neighbor query returned 0 rows. Writing empty outputs.')
        _write_empty_outputs(outdir, targname)
        return outdir

    if verbose:
        print('Number of records: ', len(r))

    gaiacoord = SkyCoord(
        ra=r['ra'],
        dec=r['dec'],
        distance=(1000.0 / r['parallax']) * u.parsec,
        frame='icrs',
        pm_ra_cosdec=r['pmra'],
        pm_dec=r['pmdec'],
    )

    sep = gaiacoord.separation(Pcoord)
    sep3d = gaiacoord.separation_3d(Pcoord)

    Ppmllpmbb = bc.pmrapmdec_to_pmllpmbb(
        Pcoord.pm_ra_cosdec.value,
        Pcoord.pm_dec.value,
        Pcoord.ra.value,
        Pcoord.dec.value,
        degree=True,
    )
    Pvxvyvz = bc.vrpmllpmbb_to_vxvyvz(
        Pcoord.radial_velocity.value,
        Ppmllpmbb[0],
        Ppmllpmbb[1],
        Pllbb[0],
        Pllbb[1],
        Pcoord.distance.value / 1000.0,
        XYZ=False,
        degree=True,
    )

    Cll = (math.atan2(Pvxvyvz[1], Pvxvyvz[0]) * 180.0 / np.pi) % 360
    Cbb = math.atan2(Pvxvyvz[2], np.sqrt(Pvxvyvz[0] ** 2 + Pvxvyvz[1] ** 2)) * 180.0 / np.pi
    Cradec = bc.lb_to_radec(Cll, Cbb, degree=True, epoch=2000.0)
    Ccoord = SkyCoord(ra=Cradec[0] * u.deg, dec=Cradec[1] * u.deg, distance=999999.9, frame='icrs')

    Cangle = gaiacoord.separation(Ccoord)
    zzflip = np.where(Cangle.degree > 90.0)
    if np.array(zzflip).size > 0:
        Cangle[zzflip] = (180.0 - Cangle[zzflip].degree) * u.deg

    Gllbb = bc.radec_to_lb(gaiacoord.ra.value, gaiacoord.dec.value, degree=True)
    Gxyz = bc.lbd_to_XYZ(Gllbb[:, 0], Gllbb[:, 1], gaiacoord.distance / 1000.0, degree=True)
    Gvrpmllpmbb = bc.vxvyvz_to_vrpmllpmbb(
        Pvxvyvz[0] * np.ones(len(Gxyz[:, 0])),
        Pvxvyvz[1] * np.ones(len(Gxyz[:, 1])),
        Pvxvyvz[2] * np.ones(len(Gxyz[:, 2])),
        Gxyz[:, 0],
        Gxyz[:, 1],
        Gxyz[:, 2],
        XYZ=True,
    )
    Gpmrapmdec = bc.pmllpmbb_to_pmrapmdec(
        Gvrpmllpmbb[:, 1],
        Gvrpmllpmbb[:, 2],
        Gllbb[:, 0],
        Gllbb[:, 1],
        degree=True,
    )

    Gvtanerr = np.ones(len(Gxyz[:, 0]))
    Gpmerr = Gvtanerr * 206265000.0 * 3.154e7 / (gaiacoord.distance.value * 3.086e13)
    Gchi2 = np.sqrt(
        (Gpmrapmdec[:, 0] - gaiacoord.pm_ra_cosdec.value) ** 2
        + (Gpmrapmdec[:, 1] - gaiacoord.pm_dec.value) ** 2
    ) / Gpmerr

    sep3d_pc = sep3d.value
    sep_deg = sep.degree
    cangle_deg = Cangle.degree
    bp_rp = np.asarray(r['bp_rp'])
    phot_excess = np.asarray(r['phot_bp_rp_excess_factor'])
    ruwe = np.asarray(r['ruwe'])

    base_mask = (sep3d_pc < searchradpc.value) & (Gchi2 < vlim.value)
    not_self_mask = sep_deg > 1e-5
    conv_mask = cangle_deg > convergcut
    finite_bprp_mask = np.isfinite(bp_rp)
    cmd_clean_mask = finite_bprp_mask & (phot_excess < (1.3 + 0.06 * bp_rp ** 2))

    RV = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    RVerr = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    RVsrc = np.array(['                             None' for _ in range(np.array(r['ra']).size)])

    cand = np.where(base_mask)[0]
    if cand.size > 0:
        cand = cand[np.argsort(sep3d_pc[cand])]

    print('Populating RV table')
    for idx in cand:
        if np.isnan(r['radial_velocity'][idx]) is False:
            RV[idx] = r['radial_velocity'][idx]
            if np.ma.is_masked(r['teff_gspphot'][idx]) is False:
                if (r['teff_gspphot'][idx] >= 8500.0) and (np.ma.is_masked(r['grvs_mag'][idx]) is False):
                    RV[idx] = r['radial_velocity'][idx] - (7.98 - 1.135 * r['grvs_mag'][idx])
                elif (r['teff_gspphot'][idx] >= 8500.0) and (np.ma.is_masked(r['phot_rp_mean_mag'][idx]) is False):
                    RV[idx] = r['radial_velocity'][idx] - (7.98 - 1.135 * r['phot_rp_mean_mag'][idx])
            RVerr[idx] = r['radial_velocity_error'][idx]
            RVsrc[idx] = 'Gaia_DR3'

    if os.path.isfile('LocalRV.csv'):
        with open('LocalRV.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV)
            for row in readCSV:
                ww = np.where(r['designation'] == row[0])[0]
                if (np.array(ww).size == 1) and (RVerr[ww] > float(row[3])):
                    RV[ww] = float(row[2])
                    RVerr[ww] = float(row[3])
                    RVsrc[ww] = row[4]

    sptstring = np.full(np.array(r['bp_rp']).size, 'nan', dtype=object)
    fnuvj = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    W13 = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    W13err = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    ROSATflux = np.full(np.array(r['ra']).size, np.nan, dtype=float)

    try:
        mamajek = CMD_RESOURCES['mamajek']
        pleiades = CMD_RESOURCES['pleiades']
        tuchor = CMD_RESOURCES['tuchor']
        usco = CMD_RESOURCES['usco']
        chai = CMD_RESOURCES['chai']

        if any(arr is None for arr in (mamajek, pleiades, tuchor, usco, chai)):
            print('DEBUG: CMD reference arrays missing; skipping CMD plot.')
        else:
            yy = cand[finite_bprp_mask[cand]] if cand.size > 0 else np.array([], dtype=int)
            yy2 = np.where(base_mask & not_self_mask & conv_mask & cmd_clean_mask)[0]
            if yy2.size > 0:
                yy2 = yy2[np.argsort((-Gchi2)[yy2])]

            if yy.size == 0 or yy2.size == 0:
                print('DEBUG: CMD selection empty; skipping CMD plot.')
            else:
                figname = outdir + targname.replace(' ', '') + 'cmd.png'
                fig, ax1 = plt.subplots(figsize=(12, 8))
                ccc = None

                absG_yy = r['phot_g_mean_mag'][yy] - (5.0 * np.log10(gaiacoord.distance[yy].value) - 5.0)
                ax1.axis([
                    math.floor(np.nanmin(r['bp_rp'][yy])),
                    math.ceil(np.nanmax(r['bp_rp'][yy])),
                    math.ceil(np.nanmax(absG_yy)) + 1,
                    math.floor(np.nanmin(absG_yy)) - 1,
                ])

                ax1.set_xlabel(r'$B_p-R_p$ (mag)', fontsize=16)
                ax1.set_ylabel(r'$M_G$ (mag)', fontsize=16)
                ax1.set_title(f"{targname} CMD")

                ax2 = ax1.twiny()
                ax2.set_xlim(ax1.get_xlim())
                spttickvals = np.array([-0.037, 0.377, 0.782, 0.980, 1.84, 2.50, 3.36, 4.75])
                sptticklabs = np.array(['A0', 'F0', 'G0', 'K0', 'M0', 'M3', 'M5', 'M7'])
                xx = np.where((spttickvals >= math.floor(np.nanmin(r['bp_rp'][yy]))) & (spttickvals <= math.ceil(np.nanmax(r['bp_rp'][yy]))))[0]
                ax2.set_xticks(spttickvals[xx])
                ax2.set_xticklabels(sptticklabs[xx])
                ax2.set_xlabel('SpT', fontsize=16, labelpad=15)

                ax1.plot(chai[:, 1], chai[:, 0], zorder=1, label='Cha-I (0-5 Myr)')
                ax1.plot(usco[:, 1], usco[:, 0], zorder=2, label='USco (11 Myr)')
                ax1.plot(tuchor[:, 1], tuchor[:, 0], zorder=3, label='Tuc-Hor (40 Myr)')
                ax1.plot(pleiades[:, 1], pleiades[:, 0], zorder=4, label='Pleiades (125 Myr)')
                ax1.plot(mamajek[:, 2], mamajek[:, 1], zorder=5, label='Mamajek MS')

                for idx in yy2:
                    msize = (17 - 12.0 * (sep3d_pc[idx] / searchradpc.value)) ** 2
                    mcolor = Gchi2[idx]
                    medge = 'black'
                    mzorder = 7
                    mshape = 'o' if (ruwe[idx] < 1.2) else 's'

                    if rvcut is not None and np.isnan(RV[idx]) is False:
                        rv_delta = np.abs(RV[idx] - Gvrpmllpmbb[idx, 0])
                        if (rv_delta > rvcut) and ((rv_delta / RVerr[idx]) > 2.0):
                            mshape = '+'
                            mcolor = 'black'
                            mzorder = 6
                        if rv_delta <= rvcut:
                            medge = 'blue'

                    sc = ax1.scatter(
                        [r['bp_rp'][idx]],
                        [r['phot_g_mean_mag'][idx] - (5.0 * np.log10(gaiacoord.distance[idx].value) - 5.0)],
                        s=msize,
                        c=('black' if mcolor == 'black' else mcolor),
                        marker=mshape,
                        edgecolors=medge,
                        zorder=mzorder,
                        vmin=0.0,
                        vmax=vlim.value,
                        cmap='cubehelix',
                        label='_nolabel',
                    )
                    if mcolor != 'black':
                        ccc = sc

                ax1.scatter([], [], c='white', edgecolors='black', marker='o', s=12 ** 2, label='RUWE < 1.2')
                ax1.scatter([], [], c='white', edgecolors='black', marker='s', s=12 ** 2, label='RUWE >= 1.2')
                ax1.scatter([], [], c='white', edgecolors='blue', marker='o', s=12 ** 2, label='RV Comoving')
                ax1.scatter([], [], c='black', marker='+', s=12 ** 2, label='RV Outlier')

                ax1.plot(
                    r['bp_rp'][cand[0]],
                    r['phot_g_mean_mag'][cand[0]] - (5.0 * np.log10(gaiacoord.distance[cand[0]].value) - 5.0),
                    'rx',
                    markersize=18,
                    mew=3,
                    markeredgecolor='red',
                    zorder=10,
                    label=targname,
                )
                ax1.legend(fontsize=11)

                if ccc is not None:
                    cb = plt.colorbar(ccc, ax=ax1)
                    cb.set_label(label='Velocity Difference (km/s)', fontsize=14)

                _save_or_skip(fig, figname, showplots=showplots)

    except Exception as e:
        print(f'DEBUG: CMD plot failed ({type(e).__name__}: {e}); continuing.')

    try:
        yy2 = np.where(base_mask & not_self_mask & conv_mask)[0]
        if yy2.size > 0:
            yy2 = yy2[np.argsort((-Gchi2)[yy2])]
        zz3 = np.where((sep3d_pc < searchradpc.value) & not_self_mask)[0]

        if yy2.size == 0:
            print('DEBUG: PM plot selection empty; skipping PM plot.')
        else:
            figname = outdir + targname.replace(' ', '') + 'pmd.png'
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ccc = None

            pmra_sel = np.array(r['pmra'][yy2])
            pmdec_sel = np.array(r['pmdec'][yy2])
            pmra_span = np.ptp(pmra_sel)
            if pmra_span == 0:
                pmra_span = 1.0

            ax1.axis([
                pmra_sel.max() + 0.05 * pmra_span,
                pmra_sel.min() - 0.05 * pmra_span,
                pmdec_sel.min() - 0.05 * pmra_span,
                pmdec_sel.max() + 0.05 * pmra_span,
            ])

            ax1.errorbar(
                r['pmra'][yy2],
                r['pmdec'][yy2],
                yerr=r['pmdec_error'][yy2],
                xerr=r['pmra_error'][yy2],
                fmt='none',
                ecolor='k',
            )

            ax1.scatter(
                r['pmra'][zz3],
                r['pmdec'][zz3],
                s=(0.5) ** 2,
                marker='o',
                c='black',
                zorder=2,
                label='Field',
            )

            for idx in yy2:
                msize = (17 - 12.0 * (sep3d_pc[idx] / searchradpc.value)) ** 2
                mcolor = Gchi2[idx]
                medge = 'black'
                mzorder = 7
                mshape = 'o' if (ruwe[idx] < 1.2) else 's'

                if rvcut is not None and np.isnan(RV[idx]) is False:
                    rv_delta = np.abs(RV[idx] - Gvrpmllpmbb[idx, 0])
                    if (rv_delta > rvcut) and ((rv_delta / RVerr[idx]) > 2.0):
                        mshape = '+'
                        mcolor = 'black'
                        mzorder = 6
                    if rv_delta <= rvcut:
                        medge = 'blue'

                sc = ax1.scatter(
                    [r['pmra'][idx]],
                    [r['pmdec'][idx]],
                    s=msize,
                    c=('black' if mcolor == 'black' else mcolor),
                    marker=mshape,
                    edgecolors=medge,
                    zorder=mzorder,
                    vmin=0.0,
                    vmax=vlim.value,
                    cmap='cubehelix',
                    label='_nolabel',
                )
                if mcolor != 'black':
                    ccc = sc

            ax1.scatter([], [], c='white', edgecolors='black', marker='o', s=12 ** 2, label='RUWE < 1.2')
            ax1.scatter([], [], c='white', edgecolors='black', marker='s', s=12 ** 2, label='RUWE >= 1.2')
            ax1.scatter([], [], c='white', edgecolors='blue', marker='o', s=12 ** 2, label='RV Comoving')
            ax1.scatter([], [], c='black', marker='+', s=12 ** 2, label='RV Outlier')

            ax1.plot(
                Pgaia['pmra'][minpos],
                Pgaia['pmdec'][minpos],
                'rx',
                markersize=18,
                mew=3,
                markeredgecolor='red',
                zorder=3,
                label=targname,
            )

            ax1.set_xlabel(r'$\mu_{RA}$ (mas/yr)', fontsize=22, labelpad=10)
            ax1.set_ylabel(r'$\mu_{DEC}$ (mas/yr)', fontsize=22, labelpad=10)
            ax1.legend(fontsize=12)

            if ccc is not None:
                cb = plt.colorbar(ccc, ax=ax1)
                cb.set_label(label='Tangential Velocity Difference (km/s)', fontsize=18, labelpad=10)

            _save_or_skip(fig, figname, showplots=showplots)

    except Exception as e:
        print(f'DEBUG: PM plot failed ({type(e).__name__}: {e}); continuing.')

    print('Creating Output Tables with Results')
    yy = cand

    warnings.filterwarnings('ignore', category=UserWarning)
    rows = []
    for idx in yy:
        rows.append({
            'designation': r['designation'][idx],
            'ra': gaiacoord.ra[idx].value,
            'dec': gaiacoord.dec[idx].value,
            'gmag': r['phot_g_mean_mag'][idx],
            'bp_rp': r['bp_rp'][idx],
            'voff': Gchi2[idx],
            'sep_deg': sep[idx].value,
            'sep3d_pc': sep3d[idx].value,
            'vr_pred': Gvrpmllpmbb[idx, 0],
            'vr_obs': RV[idx],
            'vr_err': RVerr[idx],
            'plx_mas': r['parallax'][idx],
            'spt': sptstring[idx],
            'fnuvj': fnuvj[idx],
            'w1_w3': W13[idx],
            'ruwe': r['ruwe'][idx],
            'xcrate': ROSATflux[idx],
            'rvsrc': RVsrc[idx],
            'pmra_pred': Gpmrapmdec[idx, 0],
            'pmdec_pred': Gpmrapmdec[idx, 1],
            'pmra': gaiacoord.pm_ra_cosdec.value[idx],
            'pmra_err': r['pmra_error'][idx],
            'pmdec': gaiacoord.pm_dec.value[idx],
            'pmdec_err': r['pmdec_error'][idx],
        })

    _write_outputs(outdir, targname, rows)

    if verbose:
        print('All output can be found in ' + outdir)

    return outdir