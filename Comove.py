# import pkg_resources
##figure out where the big fits files are in this installation
datapath = './resources' #pkg_resources.resource_filename('Comove','resources')
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
print('DEBUG 0')
Simbad.reset_votable_fields()
Simbad.TIMEOUT = 1500
Simbad.server = "simbad.harvard.edu"
#Simbad.add_votable_fields('typed_id')
customSimbad = Simbad()
customSimbad.add_votable_fields('rvz_radvel','rvz_err','rvz_bibcode')
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.coordinates import ICRS
from astroquery.gaia import Gaia
from astroquery.exceptions import NoResultsWarning
import galpy.util.coords as bc
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from astroquery.mast import Catalogs
from astroquery.ipac.irsa import Irsa
Irsa.TIMEOUT = 600
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.io import readsav
from astroquery.vizier import Vizier
from astropy.utils.data import conf
conf.remote_timeout = 60.0
Vizier.TIMEOUT = 600
import os,warnings,sys
import urllib.request
import csv
import pickle
import matplotlib as mpl

print('DEBUG 1')

#if 'dustmaps.bayestar' in sys.modules: print('Bayestar already imported, skipping 30-second load time.')

"""
if 'dustmaps.bayestar' not in sys.modules:
    print('Bayestar not imported, doing so now. Will require 30 seconds or so.')
    from dustmaps.config import config
    datadir = '~/Dropbox/Malmquist/'
    bayestarver = 'bayestar2019'
    testname = datadir + 'bayestar/' + bayestarver + '.h5'
    config['data_dir'] = datadir
    from dustmaps.bayestar import BayestarQuery
    bayestar = BayestarQuery(version=bayestarver)
    if ((os.path.isfile(os.path.expanduser(testname))) == True) : print('Already downloaded Bayestar files.')
    if ((os.path.isfile(os.path.expanduser(testname))) == False):
        import dustmaps.bayestar # Only uncomment if running in a new place to download dust maps again
        dustmaps.bayestar.fetch()
"""

mpl.rcParams['lines.linewidth']   = 2
mpl.rcParams['axes.linewidth']    = 2
mpl.rcParams['xtick.major.width'] =2
mpl.rcParams['ytick.major.width'] =2
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['axes.labelweight']='semibold'
mpl.rcParams['axes.titlesize']=9
mpl.rcParams['axes.titleweight']='semibold'
mpl.rcParams['font.weight'] = 'semibold'
plt.rcParams['figure.facecolor'] = 'white'

def findfriends(targname, radial_velocity, velocity_limit=5.0, search_radius=25.0,
                rvcut=5.0, convergcut=5.0, radec=[None, None],
                output_directory=None, showplots=False, verbose=False,
                DoGALEX=False, DoWISE=False, DoROSAT=False):

    radvel = radial_velocity * u.kilometer / u.second
    if (convergcut is None): convergcut = 0.0

    # ---------- Output directory ----------
    if output_directory is None:
        outdir = './' + targname.replace(" ", "") + '_friends/'
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

    # ---------- Resolve coordinates ----------
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
    if verbose: print(c)

    # ---------- Gaia: precise target ----------
    print('Asking Gaia for precise coordinates')
    sqltext = f"""
    SELECT *
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
        CIRCLE('ICRS', {c.ra.value}, {c.dec.value}, {6.0/3600.0})
    )=1
    """
    job = Gaia.launch_job_async(sqltext, dump_to_file=False)
    Pgaia = job.get_results()
    job = Gaia.launch_job_async(sqltext, dump_to_file=False)
    Pgaia = job.get_results()

    if len(Pgaia) == 0:
        print("DEBUG: Gaia target cone search returned 0 rows. Writing empty outputs.")
        # Write an empty header-only file
        filename = outdir + targname.replace(" ", "") + ".txt"
        with open(filename, 'w') as f:
            f.write("GaiaDR3 RA DEC Gmag Bp-Rp Voff(km/s) Sep(deg) 3D(pc) Vr(pred) Vr(obs) Vrerr Plx(mas) SpT FnuvJ W1-W3 RUWE XCrate RVsrc PMRApred PMDecpred PMRA PMRAerr PMDec PMDecerr\n")
        # Convert to csv
        csv_filename = outdir + targname.replace(" ", "") + ".csv"
        original_headers = pd.read_csv(filename, sep=r'\s+', nrows=0).columns.tolist()
        new_names = ['Catalog', 'Type'] + original_headers
        csv_file = pd.read_csv(filename, sep=r'\s+', names=new_names, skiprows=1, index_col=False)
        csv_file.to_csv(csv_filename, index=False)
        return outdir

    # pick brightest unmasked
    minpos = Pgaia['phot_g_mean_mag'].tolist().index(
        min(Pgaia['phot_g_mean_mag'][~Pgaia['phot_g_mean_mag'].mask])
    )

    Pcoord = SkyCoord(
        ra=Pgaia['ra'][minpos]*u.deg,
        dec=Pgaia['dec'][minpos]*u.deg,
        distance=(1000.0/Pgaia['parallax'][minpos])*u.parsec,
        frame='icrs',
        radial_velocity=radvel,
        pm_ra_cosdec=Pgaia['pmra'][minpos]*u.mas/u.year,
        pm_dec=Pgaia['pmdec'][minpos]*u.mas/u.year
    )

    searchraddeg = np.arcsin(searchradpc / Pcoord.distance).to(u.deg)
    minpar = (1000.0*u.parsec) / (Pcoord.distance + searchradpc) * u.mas

    if verbose:
        print(Pcoord)
        print('Search radius in deg: ', searchraddeg)
        print('Minimum parallax: ', minpar)

    # ---------- Gaia: neighbor query ----------
    print('Querying Gaia for neighbors')
    Pllbb = bc.radec_to_lb(Pcoord.ra.value, Pcoord.dec.value, degree=True)
    if np.abs(Pllbb[1]) > 10.0:
        plxcut = max(0.5, (1000.0 / Pcoord.distance.value / 10.0))
    else:
        plxcut = 0.5
    print('Parallax cut: ', plxcut)

    # Build neighbor query safely (avoids broken quote bugs)
    if (searchradpc < Pcoord.distance):
        sqltext = f"""
        SELECT *
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
            CIRCLE('ICRS', {Pcoord.ra.value}, {Pcoord.dec.value}, {searchraddeg.value})
        ) = 1
        AND parallax > {minpar.value}
        AND parallax_error < {plxcut};
        """
    else:
        sqltext = f"""
        SELECT *
        FROM gaiadr3.gaia_source
        WHERE parallax > {minpar.value}
        AND parallax_error < {plxcut};
        """
        print('Note, using all-sky search')

    if verbose:
        print(sqltext)
        print()

    job = Gaia.launch_job_async(sqltext, dump_to_file=False)
    r = job.get_results()

    if len(r) == 0:
        print("DEBUG: Gaia neighbor query returned 0 rows. Writing empty outputs.")
        filename = outdir + targname.replace(" ", "") + ".txt"
        with open(filename, 'w') as f:
            f.write("GaiaDR3 RA DEC Gmag Bp-Rp Voff(km/s) Sep(deg) 3D(pc) Vr(pred) Vr(obs) Vrerr Plx(mas) SpT FnuvJ W1-W3 RUWE XCrate RVsrc PMRApred PMDecpred PMRA PMRAerr PMDec PMDecerr\n")
        csv_filename = outdir + targname.replace(" ", "") + ".csv"
        original_headers = pd.read_csv(filename, sep=r'\s+', nrows=0).columns.tolist()
        new_names = ['Catalog', 'Type'] + original_headers
        csv_file = pd.read_csv(filename, sep=r'\s+', names=new_names, skiprows=1, index_col=False)
        csv_file.to_csv(csv_filename, index=False)
        return outdir

    if verbose: print('Number of records: ', len(r))

    # ---------- Build coordinate arrays ----------
    gaiacoord = SkyCoord(
        ra=r['ra'],
        dec=r['dec'],
        distance=(1000.0/r['parallax'])*u.parsec,
        frame='icrs',
        pm_ra_cosdec=r['pmra'],
        pm_dec=r['pmdec']
    )

    sep = gaiacoord.separation(Pcoord)
    sep3d = gaiacoord.separation_3d(Pcoord)

    # ---------- Convergent point + predicted PM ----------
    Pllbb = bc.radec_to_lb(Pcoord.ra.value, Pcoord.dec.value, degree=True)
    Ppmllpmbb = bc.pmrapmdec_to_pmllpmbb(
        Pcoord.pm_ra_cosdec.value, Pcoord.pm_dec.value,
        Pcoord.ra.value, Pcoord.dec.value, degree=True
    )
    Pvxvyvz = bc.vrpmllpmbb_to_vxvyvz(
        Pcoord.radial_velocity.value, Ppmllpmbb[0], Ppmllpmbb[1],
        Pllbb[0], Pllbb[1], Pcoord.distance.value/1000.0,
        XYZ=False, degree=True
    )

    Cll = (math.atan2(Pvxvyvz[1], Pvxvyvz[0]) * 180.0/np.pi) % 360
    Cbb = math.atan2(Pvxvyvz[2], np.sqrt(Pvxvyvz[0]**2 + Pvxvyvz[1]**2)) * 180.0/np.pi
    Cradec = bc.lb_to_radec(Cll, Cbb, degree=True, epoch=2000.0)
    Ccoord = SkyCoord(ra=Cradec[0]*u.deg, dec=Cradec[1]*u.deg, distance=999999.9, frame='icrs')

    Cangle = gaiacoord.separation(Ccoord)
    zzflip = np.where((Cangle.degree > 90.0))
    if np.array(zzflip).size > 0:
        Cangle[zzflip] = (180.0 - Cangle[zzflip].degree)*u.deg

    Gllbb = bc.radec_to_lb(gaiacoord.ra.value, gaiacoord.dec.value, degree=True)
    Gxyz = bc.lbd_to_XYZ(Gllbb[:, 0], Gllbb[:, 1], gaiacoord.distance/1000.0, degree=True)
    Gvrpmllpmbb = bc.vxvyvz_to_vrpmllpmbb(
        Pvxvyvz[0]*np.ones(len(Gxyz[:, 0])),
        Pvxvyvz[1]*np.ones(len(Gxyz[:, 1])),
        Pvxvyvz[2]*np.ones(len(Gxyz[:, 2])),
        Gxyz[:, 0], Gxyz[:, 1], Gxyz[:, 2], XYZ=True
    )
    Gpmrapmdec = bc.pmllpmbb_to_pmrapmdec(
        Gvrpmllpmbb[:, 1], Gvrpmllpmbb[:, 2],
        Gllbb[:, 0], Gllbb[:, 1], degree=True
    )

    # predicted PM error model
    Gvtanerr = 1.0 * np.ones(len(Gxyz[:, 0]))
    Gpmerr = Gvtanerr * 206265000.0 * 3.154e7 / (gaiacoord.distance.value * 3.086e13)

    Gchi2 = np.sqrt(
        (Gpmrapmdec[:, 0] - gaiacoord.pm_ra_cosdec.value)**2 +
        (Gpmrapmdec[:, 1] - gaiacoord.pm_dec.value)**2
    ) / Gpmerr

    # ---------- RV arrays always exist ----------
    RV = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    RVerr = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    RVsrc = np.array(['                             None' for _ in range(np.array(r['ra']).size)])

    # candidates for RV population
    cand = np.where((sep3d.value < searchradpc.value) & (Gchi2 < vlim.value))[0]
    cand = cand[np.argsort(sep3d[cand])] if cand.size > 0 else cand

    print('Populating RV table')
    for idx in cand:
        if np.isnan(r['radial_velocity'][idx]) == False:
            RV[idx] = r['radial_velocity'][idx]
            if (np.ma.is_masked(r['teff_gspphot'][idx]) == False):
                if (r['teff_gspphot'][idx] >= 8500.0) and (np.ma.is_masked(r['grvs_mag'][idx]) == False):
                    RV[idx] = r['radial_velocity'][idx] - (7.98 - 1.135 * r['grvs_mag'][idx])
                elif (r['teff_gspphot'][idx] >= 8500.0) and (np.ma.is_masked(r['phot_rp_mean_mag'][idx]) == False):
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

    # ---------- Defaults for outputs used later ----------
    sptstring = ["nan" for _ in range(np.array(r['bp_rp']).size)]
    fnuvj = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    W13 = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    W13err = np.full(np.array(r['ra']).size, np.nan, dtype=float)
    ROSATflux = np.full(np.array(r['ra']).size, np.nan, dtype=float)

    # ---------- Plot helpers ----------
    def _save_or_skip(fig, fname):
        try:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.2, dpi=200)
            if showplots: plt.show()
        finally:
            plt.close('all')

    # ---------- CMD plot ----------
    try:
        mamajek = np.loadtxt(datapath + '/sptGBpRp.txt')
        pleiades = np.loadtxt(datapath + '/PleGBpRp.txt')
        tuchor = np.loadtxt(datapath + '/TucGBpRp.txt')
        usco = np.loadtxt(datapath + '/UScGBpRp.txt')
        chai = np.loadtxt(datapath + '/ChaGBpRp.txt')

        zz = np.where((sep3d.value < searchradpc.value) & (Gchi2 < vlim.value) & (np.isnan(r['bp_rp']) == False))[0]
        yy = zz[np.argsort(sep3d[zz])] if zz.size > 0 else np.array([], dtype=int)

        zz2 = np.where((sep3d.value < searchradpc.value) & (Gchi2 < vlim.value) &
                       (sep.degree > 1e-5) &
                       (r['phot_bp_rp_excess_factor'] < (1.3 + 0.06*r['bp_rp']**2)) &
                       (Cangle.degree > convergcut) &
                       (np.isnan(r['bp_rp']) == False))[0]
        yy2 = zz2[np.argsort((-Gchi2)[zz2])] if zz2.size > 0 else np.array([], dtype=int)

        if yy.size == 0 or yy2.size == 0:
            print("DEBUG: CMD selection empty; skipping CMD plot.")
        else:
            figname = outdir + targname.replace(" ", "") + "cmd.png"
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ccc = None
            ddd = None

            ax1.axis([
                math.floor(min(r['bp_rp'][zz])),
                math.ceil(max(r['bp_rp'][zz])),
                math.ceil(max((r['phot_g_mean_mag'][zz] - (5.0*np.log10(gaiacoord.distance[zz].value)-5.0)))) + 1,
                math.floor(min((r['phot_g_mean_mag'][zz] - (5.0*np.log10(gaiacoord.distance[zz].value)-5.0)))) - 1
            ])

            ax1.set_xlabel(r'$B_p-R_p$ (mag)', fontsize=16)
            ax1.set_ylabel(r'$M_G$ (mag)', fontsize=16)

            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            spttickvals = np.array([-0.037, 0.377, 0.782, 0.980, 1.84, 2.50, 3.36, 4.75])
            sptticklabs = np.array(['A0', 'F0', 'G0', 'K0', 'M0', 'M3', 'M5', 'M7'])
            xx = np.where((spttickvals >= math.floor(min(r['bp_rp'][zz]))) & (spttickvals <= math.ceil(max(r['bp_rp'][zz]))))[0]
            ax2.set_xticks(spttickvals[xx])
            ax2.set_xticklabels(sptticklabs[xx])
            ax2.set_xlabel('SpT', fontsize=16, labelpad=15)

            ax1.plot(chai[:, 1], chai[:, 0], zorder=1, label='Cha-I (0-5 Myr)')
            ax1.plot(usco[:, 1], usco[:, 0], zorder=2, label='USco (11 Myr)')
            ax1.plot(tuchor[:, 1], tuchor[:, 0], zorder=3, label='Tuc-Hor (40 Myr)')
            ax1.plot(pleiades[:, 1], pleiades[:, 0], zorder=4, label='Pleiades (125 Myr)')
            ax1.plot(mamajek[:, 2], mamajek[:, 1], zorder=5, label='Mamajek MS')

            for idx in yy2:
                msize = (17 - 12.0*(sep3d[idx].value/searchradpc.value))**2
                mcolor = Gchi2[idx]
                medge = 'black'
                mzorder = 7
                mshape = 'o' if (r['ruwe'][idx] < 1.2) else 's'

                if rvcut is not None:
                    if (np.isnan(RV[idx]) == False) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0]) > rvcut) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0])/RVerr[idx] > 2.0):
                        mshape = '+'
                        mcolor = 'black'
                        mzorder = 6
                    if (np.isnan(RV[idx]) == False) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0]) <= rvcut):
                        medge = 'blue'

                sc = ax1.scatter([r['bp_rp'][idx]],
                                 [(r['phot_g_mean_mag'][idx] - (5.0*np.log10(gaiacoord.distance[idx].value)-5.0))],
                                 s=msize, c=('black' if mcolor == 'black' else mcolor),
                                 marker=mshape, edgecolors=medge, zorder=mzorder,
                                 vmin=0.0, vmax=vlim.value, cmap='cubehelix', label='_nolabel')
                if mcolor == 'black':
                    ddd = sc
                else:
                    ccc = sc

            ax1.scatter([], [], c='white', edgecolors='black', marker='o', s=12**2, label='RUWE < 1.2')
            ax1.scatter([], [], c='white', edgecolors='black', marker='s', s=12**2, label='RUWE >= 1.2')
            ax1.scatter([], [], c='white', edgecolors='blue', marker='o', s=12**2, label='RV Comoving')
            ax1.scatter([], [], c='black', marker='+', s=12**2, label='RV Outlier')

            ax1.plot(r['bp_rp'][yy[0]],
                     (r['phot_g_mean_mag'][yy[0]] - (5.0*np.log10(gaiacoord.distance[yy[0]].value)-5.0)),
                     'rx', markersize=18, mew=3, markeredgecolor='red', zorder=10, label=targname)

            ax1.legend(fontsize=11)
            if ccc is not None:
                cb = plt.colorbar(ccc, ax=ax1)
                cb.set_label(label='Velocity Difference (km/s)', fontsize=14)

            _save_or_skip(fig, figname)

    except Exception as e:
        print(f"DEBUG: CMD plot failed ({type(e).__name__}: {e}); continuing.")

    # ---------- PM plot (THIS is where your max(empty) was happening) ----------
    try:
        zz2 = np.where((sep3d.value < searchradpc.value) & (Gchi2 < vlim.value) &
                       (sep.degree > 1e-5) & (Cangle.degree > convergcut))[0]
        yy2 = zz2[np.argsort((-Gchi2)[zz2])] if zz2.size > 0 else np.array([], dtype=int)
        zz3 = np.where((sep3d.value < searchradpc.value) & (sep.degree > 1e-5))[0]

        if yy2.size == 0:
            print("DEBUG: PM plot selection empty; skipping PM plot.")
        else:
            figname = outdir + targname.replace(" ", "") + "pmd.png"
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ccc = None
            ddd = None

            pmra_sel = np.array(r['pmra'][zz2])
            pmdec_sel = np.array(r['pmdec'][zz2])

            ax1.axis([
                (pmra_sel.max() + 0.05*np.ptp(pmra_sel)),
                (pmra_sel.min() - 0.05*np.ptp(pmra_sel)),
                (pmdec_sel.min() - 0.05*np.ptp(pmra_sel)),
                (pmdec_sel.max() + 0.05*np.ptp(pmra_sel))
            ])

            ax1.errorbar(r['pmra'][yy2], r['pmdec'][yy2],
                         yerr=r['pmdec_error'][yy2], xerr=r['pmra_error'][yy2],
                         fmt='none', ecolor='k')

            ax1.scatter([r['pmra'][zz3]], [r['pmdec'][zz3]],
                        s=(0.5)**2, marker='o', c='black', zorder=2, label='Field')

            for idx in yy2:
                msize = (17 - 12.0*(sep3d[idx].value/searchradpc.value))**2
                mcolor = Gchi2[idx]
                medge = 'black'
                mzorder = 7
                mshape = 'o' if (r['ruwe'][idx] < 1.2) else 's'

                if rvcut is not None:
                    if (np.isnan(RV[idx]) == False) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0]) > rvcut) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0])/RVerr[idx] > 2.0):
                        mshape = '+'
                        mcolor = 'black'
                        mzorder = 6
                    if (np.isnan(RV[idx]) == False) and (np.abs(RV[idx]-Gvrpmllpmbb[idx, 0]) <= rvcut):
                        medge = 'blue'

                sc = ax1.scatter([r['pmra'][idx]], [r['pmdec'][idx]],
                                 s=msize, c=('black' if mcolor == 'black' else mcolor),
                                 marker=mshape, edgecolors=medge, zorder=mzorder,
                                 vmin=0.0, vmax=vlim.value, cmap='cubehelix', label='_nolabel')
                if mcolor == 'black':
                    ddd = sc
                else:
                    ccc = sc

            ax1.scatter([], [], c='white', edgecolors='black', marker='o', s=12**2, label='RUWE < 1.2')
            ax1.scatter([], [], c='white', edgecolors='black', marker='s', s=12**2, label='RUWE >= 1.2')
            ax1.scatter([], [], c='white', edgecolors='blue', marker='o', s=12**2, label='RV Comoving')
            ax1.scatter([], [], c='black', marker='+', s=12**2, label='RV Outlier')

            ax1.plot(Pgaia['pmra'][minpos], Pgaia['pmdec'][minpos],
                     'rx', markersize=18, mew=3, markeredgecolor='red', zorder=3, label=targname)

            ax1.set_xlabel(r'$\mu_{RA}$ (mas/yr)', fontsize=22, labelpad=10)
            ax1.set_ylabel(r'$\mu_{DEC}$ (mas/yr)', fontsize=22, labelpad=10)
            ax1.legend(fontsize=12)

            if ccc is not None:
                cb = plt.colorbar(ccc, ax=ax1)
                cb.set_label(label='Tangential Velocity Difference (km/s)', fontsize=18, labelpad=10)

            _save_or_skip(fig, figname)

    except Exception as e:
        print(f"DEBUG: PM plot failed ({type(e).__name__}: {e}); continuing.")

    # ---------- (Optional) You can keep your remaining plots/GALEX/WISE/ROSAT code as-is ----------
    # The key change for batch stability is: NEVER early-return on empty plot selections.
    # We now always proceed to output tables.

    # ---------- Output tables ----------
    print('Creating Output Tables with Results')

    zz = np.where((sep3d.value < searchradpc.value) & (Gchi2 < vlim.value))[0]
    sortlist = np.argsort(sep3d[zz]) if zz.size > 0 else np.array([], dtype=int)
    yy = zz[sortlist] if zz.size > 0 else np.array([], dtype=int)

    fmt1 = "%28s %11.7f %11.7f %6.3f %6.3f %11.3f %8.4f %8.4f %8.2f %8.2f %8.2f %8.3f %4s %8.6f %6.2f %7.3f %7.3f %35s %11.3f %11.3f %11.3f %11.3f %11.3f %11.3f"
    fmt2 = fmt1
    filename = outdir + targname.replace(" ", "") + ".txt"

    warnings.filterwarnings("ignore", category=UserWarning)
    header = ("GaiaDR3                               RA         DEC   Gmag  Bp-Rp  Voff(km/s) Sep(deg)   3D(pc) "
              "Vr(pred)  Vr(obs)    Vrerr Plx(mas)  SpT    FnuvJ  W1-W3    RUWE  XCrate                               "
              "RVsrc    PMRApred   PMDecpred        PMRA     PMRAerr       PMDec    PMDecerr\n")
    with open(filename, 'w') as f:
        f.write(header)

    for idx in yy:
        with open(filename, 'a') as f:
            f.write(fmt2 % (
                r['designation'][idx],
                gaiacoord.ra[idx].value, gaiacoord.dec[idx].value,
                r['phot_g_mean_mag'][idx], r['bp_rp'][idx],
                Gchi2[idx], sep[idx].value, sep3d[idx].value,
                Gvrpmllpmbb[idx, 0], RV[idx], RVerr[idx],
                r['parallax'][idx],
                sptstring[idx],
                fnuvj[idx],
                W13[idx],
                r['ruwe'][idx],
                ROSATflux[idx],
                RVsrc[idx],
                Gpmrapmdec[idx, 0], Gpmrapmdec[idx, 1],
                gaiacoord.pm_ra_cosdec.value[idx], r['pmra_error'][idx],
                gaiacoord.pm_dec.value[idx], r['pmdec_error'][idx]
            ))
            f.write("\n")

    # convert to CSV the same way you already do
    csv_filename = outdir + targname.replace(" ", "") + ".csv"
    original_headers = pd.read_csv(filename, sep=r'\s+', nrows=0).columns.tolist()
    new_names = ['Catalog', 'Type'] + original_headers
    csv_file = pd.read_csv(filename, sep=r'\s+', names=new_names, skiprows=1, index_col=False)
    csv_file.to_csv(csv_filename, index=False)

    if verbose:
        print('All output can be found in ' + outdir)

    return outdir


# def binprob(targname,targfilt,targDmag,targDmagerr,targsep,targDPM=None,targDPMerr=None,targDPI=None,Pradvel=None,Pdist=None,Pdisterr=None,PdistU=None,PdistL=None,Pmass=None,PT=None):


#     targfilt = np.array(targfilt)
#     targDmag = np.array(targDmag)
#     targDmagerr = np.array(targDmagerr)
#     if isinstance(targDPM , np.ndarray):
#         if (len(targDPM) == 2): targDPM = np.array(targDPM)

# ### Defining standard color-magnitude and extinction relations

#     SpT_Mama = []
#     T_Mama   = []
#     M_Mama   = []
#     R_Mama   = []
#     MG_Mama  = []
#     MJ_Mama  = []
#     MH_Mama  = []
#     MK_Mama  = []
#     Phot_Mama= []
#     Filt_Mama= []

#     mamaurl = 'http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt'
#     mamafile= urllib.request.urlopen(mamaurl)
#     print()
#     print('This is querying a table maintained by Eric Mamajek, based on an initial compilation by Pecaut & Mamajek (2013).')
#     print('Cite them in your paper, and footnote the URL:')
#     print(mamaurl)
#     print('Go there and read the warnings. Note, they come after the spells.')
#     print()
#     print('You should also footnote the date that the compilation was accessed.')

#     for i in range(0,3):
#         txt = mamafile.readline()
#     print(mamafile.readline())
#     for i in range(4,22): txt = mamafile.readline()
#     hdr = str(mamafile.readline())
#     print(hdr)
#     for i in range(24,142):
#         line = str(mamafile.readline())
#         strloc = hdr.find('SpT')
#         SpT_Mama.append(           line[(strloc-1):(strloc+4)].strip() )
#         strloc = hdr.find('Teff')
#         T_Mama.append(      float( line[(strloc+0):(strloc+5)].strip()) )
#         strloc = hdr.find('R_Rsun')
#         if (line[(strloc+1):(strloc+3)] != '..'):
#             R_Mama.append(      float( line[(strloc+0):(strloc+6)].strip()) )
#         else:
#             R_Mama.append(np.nan)
#         strloc = hdr.find('Msun')
#         if (line[(strloc+0):(strloc+2)] != '..'):
#             M_Mama.append(  float( line[(strloc+0):(strloc+5)].strip()) )
#         else:
#             M_Mama.append(np.nan)
#         strloc = hdr.find('M_G')
#         if (line[(strloc+1):(strloc+3)] != '..'):
#             MG_Mama.append( float( line[(strloc-1):(strloc+5)].replace(':',' ').strip()) )
#         else:
#             MG_Mama.append(np.nan)
#         strloc = hdr.find('M_Ks')
#         if (line[(strloc+0):(strloc+2)] != '..'):
#             MK_Mama.append( float( line[(strloc-1):(strloc+5)].strip()) )
#         else:
#             MK_Mama.append(np.nan)
#         strloc = hdr.find('H-Ks')
#         if (line[(strloc+0):(strloc+2)] != '..') and (np.isnan(MK_Mama[-1]) == False):
#             MH_Mama.append( float( line[(strloc-1):(strloc+5)].strip()) + MK_Mama[-1] )
#         else:
#             MH_Mama.append(np.nan)
#         strloc = hdr.find('J-H')
#         if (line[(strloc+0):(strloc+2)] != '..') and (np.isnan(MH_Mama[-1]) == False):
#             MJ_Mama.append( float( line[(strloc-1):(strloc+5)].strip()) + MH_Mama[-1] )
#         else:
#             MJ_Mama.append(np.nan)

#     SpT_Mama = np.array(SpT_Mama)
#     T_Mama   = np.array(T_Mama)
#     R_Mama   = np.array(R_Mama)
#     M_Mama   = np.array(M_Mama)
#     MG_Mama  = np.array(MG_Mama)
#     MJ_Mama  = np.array(MJ_Mama)
#     MH_Mama  = np.array(MH_Mama)
#     MK_Mama  = np.array(MK_Mama)
#     logg_Mama = np.log10( 27400.0 * M_Mama / R_Mama**2)
#     print('Done parsing Mamajek table.')

#     print('Now parsing color tables from Kraus+2022')
#     krausurl = datapath+'/TableSynColors.txt'
#     f = open(krausurl,"r")
#     Krausphot = []
#     Krausfilt = np.array([ 'G'     , 'Ks'    , 'r'     ,     'i' , 'z'     , 'Bp'    , 'Rp' , 'Kp' , 'LP600' , \
#               '[467]' , '[562]' , '[692]' , '[716]' , '[832]' , '[880]' ])
#     for s in f:
#         Krausphot.append( [ float(s[15:22].strip()) , \
#                         (float(s[15:22].strip())-float(s[23:31].strip())) , \
#                         float(s[15:22].strip())-float(s[32:40].strip()) , \
#                         float(s[15:22].strip())-float(s[41:49].strip()) , \
#                         float(s[15:22].strip())-float(s[50:58].strip()) , \
#                         float(s[15:22].strip())-float(s[59:67].strip()) , \
#                         float(s[15:22].strip())-float(s[68:76].strip()) , \
#                         float(s[15:22].strip())-float(s[77:85].strip()) , \
#                         float(s[15:22].strip())-float(s[86:94].strip()) , \
#                         float(s[15:22].strip())-float(s[95:103].strip()) , \
#                         float(s[15:22].strip())-float(s[104:112].strip()) , \
#                         float(s[15:22].strip())-float(s[113:121].strip()) , \
#                         float(s[15:22].strip())-float(s[122:130].strip()) , \
#                         float(s[15:22].strip())-float(s[131:139].strip()) , \
#                         float(s[15:22].strip())-float(s[140:148].strip()) ] )
#     f.close
#     Krausphot = np.array(Krausphot)

#     print('Now parsing extinction tables from Kraus+2022')
#     krausurl = datapath+'/TableAXAV.txt'
#     f = open(krausurl,"r")
#     KrausAXAV = []
#     s1=[]
#     for s in f:
#         s1.append(s[20:25])
#     f.close
#     s1 = np.array([ 0.0,float(s1[1]),float(s1[2]),float(s1[3]),float(s1[5]),float(s1[6]),\
#                 float(s1[7]),float(s1[8]),float(s1[9]),float(s1[10]),float(s1[11]),\
#                 float(s1[12]),float(s1[13]),float(s1[14]),0.276,0.176,float(s1[4]) ])
#     KrausAXAV = np.stack([s1 for n in range(0,72)],axis=0)

#     krausurl = datapath+'/TableATeff.txt'
#     f = open(krausurl,"r")
#     nline=0
#     for s in f:
#         KrausAXAV[nline,0] = float( s[6:13].strip() )
#         KrausAXAV[nline,1] = float( s[23:31].strip() )
#         KrausAXAV[nline,2] = float( s[32:40].strip() )
#         KrausAXAV[nline,3] = float( s[41:49].strip() )
#         KrausAXAV[nline,7] = float( s[50:58].strip() )
#         nline=nline+1
#     f.close

#     print('Parsing all to phot table')
#     filtarr = np.array([ 'G' , 'Bp' , 'Rp' , 'r' , 'i' , 'z' , 'LP600' , \
#               '[467]' , '[562]' , '[692]' , '[716]' , '[832]' , '[880]' , 'J' , 'H' , 'Ks' ])
#     photarr = np.zeros( (SpT_Mama.size,filtarr.size) )

#     GlocP = np.where( filtarr   == 'G')[0][0]
#     KlocP = np.where( filtarr   == 'Ks')[0][0]
#     GlocK = np.where( Krausfilt == 'G')[0][0]
#     KlocK = np.where( Krausfilt == 'Ks')[0][0]
#     for i in np.arange(0,SpT_Mama.size):
#         photarr[i,0]  = MG_Mama[i]
#         photarr[i,13] = MJ_Mama[i]
#         photarr[i,14] = MH_Mama[i]
#         photarr[i,15] = MK_Mama[i]
#     for i in np.arange(0,SpT_Mama.size):
#         for j in np.arange(1,13):
#             filtloc = np.where( Krausfilt == filtarr[j])[0][0]
#             photarr[i,j]  = np.interp( photarr[i,GlocP] , \
#                               Krausphot[:,GlocK] , Krausphot[:,filtloc] , \
#                               left = np.nan , right = np.nan )

#     print('Parsing all to AXAV table.')
#     AXAVarr = np.zeros( (SpT_Mama.size,filtarr.size) )
#     for i in np.arange(0,SpT_Mama.size):
#         for j in np.arange(0,16):
#             AXAVarr[i,j] = np.interp( T_Mama[i] , np.flip(KrausAXAV[:,0]) , np.flip(KrausAXAV[:,(j+1)]))



# ### Defining Primary Star Properties

#     print('Looking up primary star RA/DEC in SIMBAD')
#     result_table = customSimbad.query_object(targname)
#     ('Target name: ',targname)
#     print(result_table['RA','DEC','RVZ_RADVEL','RVZ_ERROR','RVZ_BIBCODE'])
#     c = SkyCoord( ra=result_table['RA'][0] , dec=result_table['DEC'][0] , unit=(u.hourangle, u.deg) , frame='icrs')
#     print(c)
#     print()
    
#     print('Look up primary star astrometry/photometry in Gaia FULL DR3')
#     sqltext = "SELECT * FROM gaiadr3.gaia_source WHERE CONTAINS( \
#         POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec), \
#         CIRCLE('ICRS'," + str(c.ra.value) +","+ str(c.dec.value) +","+ str(6.0/3600.0) +"))=1;"
#     job = Gaia.launch_job_async(sqltext , dump_to_file=False)
#     Pgaia = job.get_results()
#     print(Pgaia['ra','dec','phot_g_mean_mag','pmra','pmdec','parallax','ruwe'].pprint_all())
#     minpos = Pgaia['phot_g_mean_mag'].tolist().index(min(Pgaia['phot_g_mean_mag']))
#     Pcoord = SkyCoord( ra=Pgaia['ra'][minpos]*u.deg , dec=Pgaia['dec'][minpos]*u.deg , \
#         distance=1.0*u.parsec , frame='icrs' , \
#         radial_velocity=0.0*u.kilometer/u.second , \
#         pm_ra_cosdec=Pgaia['pmra'][minpos]*u.mas/u.year , pm_dec=Pgaia['pmdec'][minpos]*u.mas/u.year )
#     Pgaia = Pgaia[minpos]
#     print(Pcoord)
#     print()

#     print('Looking up primary star photometry in 2MASS')
#     PcoordTM = Pcoord.apply_space_motion(dt=(-15.0*u.year))  
#     print(PcoordTM)
#     tmass = Irsa.query_region(PcoordTM,catalog='fp_psc' , radius='0d0m10s')
#     if ((np.where(tmass['j_m'] > -10.0)[0]).size > 0):
#         ww = np.where( (tmass['j_m'] == min(tmass['j_m'][np.where(tmass['j_m'] > 0.0)])))
#         Ptmass = tmass[ww][0]
#     print(Ptmass['ra','dec','j_m','j_cmsig','h_m','h_cmsig','k_m','k_cmsig','dist','angle'])
#     print()

#     print('Parsing primary star RV')
#     print(result_table['RVZ_RADVEL'].filled(np.nan)[0])
#     if (Pradvel != None): 
#         print('Using user-provided RV.')
#     if (Pradvel == None):
#         if (np.isnan(result_table['RVZ_RADVEL'].filled(np.nan)[0]) == False): 
#             Pradvel = result_table['RVZ_RADVEL'][0]
#             print('Using RV from SIMBAD: ',Pradvel)
#         if (np.isnan(result_table['RVZ_RADVEL'].filled(np.nan)[0]) == True) : 
#             Pradvel = 0.0
#             print('No RV provided, setting to 0.0. Be cautious of projection effects.')
#     print('Adopted radvel: ',Pradvel)
#     print()

#     print('Parsing primary star distance')
#     if (Pdist != None): 
#         print('Using user-provided distance.')
#     if (Pdist == None):
#         print(Pgaia['parallax'])
#         if ( np.isnan(Pgaia['parallax']) == False):
#             v = Vizier(columns=["id","r_med_geo","r_lo_geo","r_hi_geo"] ,\
#                             ).query_constraints(id=Pgaia['source_id'],catalog='I/352/gedr3dis')
#             vtable = v['I/352/gedr3dis']
#             distfrac = (vtable['B_rgeo'][0]-vtable['b_rgeo'][0])/(2.0*vtable['rgeo'][0])
#             if (distfrac < 0.3):
#                 Pdist = vtable['rgeo'][0]
#                 PdistU= vtable['B_rgeo'][0]- vtable['rgeo'][0]
#                 PdistL= vtable['rgeo'][0]  - vtable['b_rgeo'][0]
#                 Pdisterr = 0.5 * (PdistU + PdistL)
#                 print('Distances of Bailer-Jones21: ',Pdist,Pdisterr,PdistU,PdistL)
#     if (Pdist == None):
#         print('Can infer primary distance from G-K color. Not implemented yet. Will probably crash imminently.')

#     Pcoord = SkyCoord( ra=Pgaia['ra']*u.deg , dec=Pgaia['dec']*u.deg , \
#                   distance=Pdist*u.parsec , frame='icrs' , \
#                   radial_velocity=Pradvel*u.kilometer/u.second , \
#                   pm_ra_cosdec=Pgaia['pmra']*u.mas/u.year , pm_dec=Pgaia['pmdec']*u.mas/u.year )
#     print(Pcoord)

#     print('Parsing primary star extinction')
#     PAV = bayestar(Pcoord , mode='median') * 2.742
#     if (np.isnan(PAV) == True): 
#         print('NaN was returned, resetting to 0.0')
#         PAV = 0.0
#     print('Extinction to primary star: ',PAV)

#     print('Parsing primary star mass')
#     print(Pmass)
#     if (Pmass != None):
#         print('Using user-provided mass.')
#     zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(M_Mama) == False) )
#     if (Pmass == None):
#         PMG = Pgaia['phot_g_mean_mag'] - (5.0*np.log10(Pdist)-5.0) - PAV*0.822
#         Pmass = np.interp( PMG , MG_Mama[zz] , M_Mama[zz])
#         print('Mass from M_G: ',Pmass)
#     zz = np.where( (np.isnan(MK_Mama) == False) & (np.isnan(M_Mama) == False) )
#     if (Pmass == None):
#         PMK = Ptmass['k_m'] - (5.0*np.log10(Pdist)-5.0) - PAV*0.120
#         Pmass = np.interp( PMK , MK_Mama[zz] , M_Mama[zz])
#         print('Mass from M_K: ',Pmass)
#     if (Pmass == None):
#         print('Can infer primary mass from G-K color. Not implemented yet. Will probably crash imminently.')
#     print(Pmass)

#     print('Parsing primary star temperature')
#     if (PT != None):
#         print('Using user-privided Teff')
#     if (PT == None):
#         zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(M_Mama) == False) )
#         PT = np.interp( Pmass , np.flip(M_Mama[zz]) , np.flip(T_Mama[zz]) )
#     print(PT)

#     print('Parsing primary AX/AV extinction ratios in dmag filters')
#     PAXAV = np.zeros(targDmag.size)
#     zz = np.where( (np.isnan(T_Mama) == False) )
#     for i in range(0,targDmag.size):
#         filtloc = np.where( filtarr == targfilt[i] )[0][0]
#         PAXAV[i] = np.interp( PT , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#     print(PAXAV)

#     print('Computing primary star magnitudes in dmag filters')
#     Pmag = np.zeros(targDmag.size)
#     zz = np.where( (np.isnan(T_Mama) == False) )
#     filtloc = np.where( filtarr == 'G' )[0][0]
#     MamaG = np.interp( PT , np.flip(T_Mama[zz]) , np.flip(photarr[:,filtloc]))
#     for i in range(0,targDmag.size):
#         filtloc = np.where( filtarr == targfilt[i] )[0][0]
#         MamaX = np.interp( PT , np.flip(T_Mama[zz]) , np.flip(photarr[:,filtloc]))
#         Pmag[i] = Pgaia['phot_g_mean_mag'] - PAV*0.822 - (MamaG-MamaX) + PAV*PAXAV[i]
#     print(Pmag)
#     print(targfilt)



# #### Simulate binary population

#     print('Simulating Binary Population') # Note, someday can make this mass dependent

#     Pmass2=1.0
#     nsim = 1000000
#     print('Primary mass used: ' , Pmass2)
#     BF0   = np.interp( Pmass2 , [0.1 , 0.2 , 0.6 , 1.0 , 2.0] , [0.25 , 0.35 , 0.45 , 0.45 , 0.70] )
#     gamma = np.interp( Pmass2 , [0.1 , 0.2 , 0.6 , 1.0 , 2.0] , [4.0  , 1.02 , 0.18 , 0.00 ,-2.30] )
#     mu    = 1.48 * np.log10(Pmass2) + 2.11
#     sigma = -0.93*(np.log10(Pmass2)+0.18)**2 + 1.01

#     print('Total binary fraction adopted:   ',BF0)
#     print('Mass ratio power law adopted:    ',gamma)
#     print('Semimajor axis mu/sigma adopted: ',mu,sigma)

#     # Populate random orbital elements

#     binfrac = BF0			# Note, assuming BD secondaries aren't included in binary fraction.
#     ag , bg = (0.08/Pmass)**(gamma+1.0) , 1.0**(gamma+1.0)
#     qq      = ( ag + (bg-ag)*np.random.random(                  nsim ))**(1.0/(gamma+1.0))
#     aa      = 10**(np.random.normal(        mu , sigma      , nsim ))

# #    binfrac = 0.46 * (1.0 - 0.080/Pmass) / (1.0 - 0.1) # Account for BD secondaries that can't be simulated
# #    qq     = np.random.power(      0.080/Pmass , 1.0       , nsim )
# #    aa     = 10**(np.random.normal(       1.70 , 1.52      , nsim ))
#     ee      = np.random.uniform(           0.0  , 0.95      , nsim )
#     littleo = np.random.uniform(           0.0  , np.pi     , nsim )
#     bigO    = np.random.uniform(           0.0  , 2.0*np.pi , nsim )
#     ii      = np.arccos(np.random.uniform( 0.0  , 1.0       , nsim ))
#     MM      = np.random.uniform(           0.0  , 2.0*np.pi , nsim )

#     zz = np.where( (qq >= 0.8) & (qq <= 1.0))[0]
#     print('Number with q of 0.8 to 1.0: ',zz.size)
#     zz = np.where( (qq >= 0.6) & (qq <= 0.8))[0]
#     print('Number with q of 0.6 to 0.8: ',zz.size)
#     zz = np.where( (qq >= 0.4) & (qq <= 0.6))[0]
#     print('Number with q of 0.4 to 0.6: ',zz.size)
#     zz = np.where( (qq >= 0.2) & (qq <= 0.4))[0]
#     print('Number with q of 0.2 to 0.4: ',zz.size)
#     zz = np.where( (qq >= 0.1) & (qq <= 0.2))[0]
#     print('Number with q of 0.1 to 0.2: ',zz.size)
#     zz = np.where( (qq >= 0.08) & (qq <= 0.1))[0]
#     print('Number with q of 0.08 to 0.1: ',zz.size)



#     i = np.where( (MM<0.000001) )[0]
#     for j in i: MM[j] = MM[j] + 0.000001
#     i = np.where( (np.abs(MM-np.pi)<0.000001) )[0]
#     for j in i: MM[j] = MM[j] + 0.000002
#     i = np.where( (np.abs(MM-2.0*np.pi)<0.000001) )[0]
#     for j in i: MM[j] = MM[j] - 0.000001

#     print('Computing projected separations from orbital elements')
#     EEL  = 0.0
#     EE   = MM
#     niter= 0
#     while np.max(np.abs(EE-EEL)) > 0.00001:
#         niter=niter+1
#         EEL  = EE
#         gEE  = EE - ee*np.sin(EE) - MM
#         ggEE = 1.0 - ee*np.cos(EE)
#         EE   = EE - gEE/ggEE
#     print('Iterations required for Keplers Eqn:',niter)
#     rr = aa*(1 - ee*np.cos(EE))
#     ff = np.arccos( ((aa*(1.0-ee**2))/rr - 1.0)/ee )
#     XX = rr * (np.cos(bigO)*np.cos(littleo+ff) - np.sin(bigO)*np.sin(littleo+ff)*np.cos(ii) )
#     YY = rr * (np.sin(bigO)*np.cos(littleo+ff) - np.cos(bigO)*np.sin(littleo+ff)*np.cos(ii) )
#     rhoAU = (XX**2 + YY**2)**0.5
#     rhoAS = rhoAU/Pdist

#     print('Diagnostics')
#     print(qq[0:10])
#     # Convert mass ratios to mass and Teff, compute AG and AK
#     sM   = Pmass*qq
#     print(sM[0:10])

#     zz = np.where( (np.isnan(M_Mama) == False) & (np.isnan(T_Mama) == False) )
#     sT   = np.interp( sM , np.flip(M_Mama[zz]) , np.flip(T_Mama[zz]))
#     print(sT[0:10])

#     sAXAV = np.zeros([nsim,targfilt.size])
#     zz = np.where( (np.isnan(T_Mama) == False) )
#     for i in range(0,targfilt.size):
#         filtloc = np.where( filtarr == targfilt[i] )[0][0]
#         sAXAV[:,i] = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )

#     filtloc = np.where( filtarr == 'G' )[0][0]
#     sAGAV = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#     filtloc = np.where( filtarr == 'Ks' )[0][0]
#     sAKAV = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )

#     SMX = np.zeros([nsim,targfilt.size])
#     SX = np.zeros([nsim,targfilt.size])
#     dX = np.zeros([nsim,targfilt.size])
#     # Compute photometry
#     zz = np.where( (np.isnan(T_Mama) == False) )
#     for i in range(0,targfilt.size):
#         filtloc = np.where( filtarr == targfilt[i] )[0][0]
#         SMX[:,i] = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(np.ravel(photarr[zz,filtloc])) )
#         SX[:,i]  = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(np.ravel(photarr[zz,filtloc])) ) + \
#                                 (5.0*np.log10(Pdist)-5.0) - PAV*sAXAV[:,i]
#         dX[:,i]  = SX[:,i] - Pmag[i]
#     zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(MG_Mama) == False) )
#     sMG   = np.interp( sT , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]))
#     sG    = sMG + (5.0*np.log10(Pdist)-5.0) + PAV*sAGAV
#     with np.printoptions(threshold=np.inf):
#         print(SMX[0:10,:])
#         print(SX[0:10,:])
#         print(dX[0:10,:])

#     vorb   = 29.78*np.sqrt((Pmass + sM)/rhoAU)
#     if (targDPMerr is not None): 
#         sPMerr = np.interp( sG , [ 15.0 , 17.0 , 20.0 , 21.0] , [0.03 , 0.07 , 0.5 , 1.4])
#         print(sPMerr)
#         sPMerr = targDPMerr * np.ones(vorb.size)
#         print(sPMerr)
#         sPMerr = np.sqrt( sPMerr**2 + (vorb*210.0/Pdist)**2 )
#         print(sPMerr)
#         sPIerr = np.interp( sG , [ 15.0 , 17.0 , 20.0 , 21.0] , [0.03 , 0.07 , 0.5 , 1.3])



# ### Create binary figures

#     if (os.path.isdir('binprob/' + targname) == False): os.mkdir('binprob/' + targname)

# ############
#     if (targfilt.size > 1):
#         for i in range(1,targfilt.size):

#             fig,ax1 = plt.subplots(figsize=(9,8))
#             ax1.axis([ 0.0 , (max(dX[:,0])+0.2) , 0.0 , (max(dX[:,i])+0.2) ])
#             ax1.set_xlabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#             ax1.set_ylabel(r'$\Delta ' + targfilt[i] + '$ (mag)' , fontsize=16)
#             ax1.tick_params(axis='both',which='major',labelsize=12)

#             ccc = ax1.scatter( dX[:,0] + np.random.normal( 0.0 , (targDmagerr[0]+0.1) , nsim ) , \
#                            dX[:,i] + np.random.normal( 0.0 , (targDmagerr[i]+0.1) , nsim )  , \
#                            s=2 , c='black' , marker='+' , label='Simulated bins')

#             plt.plot( targDmag[0] , targDmag[i] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )
#             plt.savefig( 'binprob/' + targname + '/' + (targname + '_Bin_'+targfilt[0]+'_'+targfilt[i]+'_D.png') , \
#                             bbox_inches='tight', pad_inches=0.2 , dpi=200)
#             plt.close('all')


#     if (targfilt.size > 1):
#         for i in range(1,targfilt.size):

#             nx=30
#             ny=30
#             xstart = 0.0
#             xend   = max(dX[:,0])+0.2
#             ystart = 0.0
#             yend   = max(dX[:,i])+0.2
#             xstep = (xend - xstart) / (nx)
#             ystep = (yend - ystart) / (ny)
#             renorm = np.abs((1.0 / xstep) * (1.0 / ystep))

#             X3,Y3 = np.meshgrid( np.linspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#             DXDYmap = np.zeros([nx+1,ny+1])
#             for xxx in np.arange(nx+1):
#                 for yyy in np.arange(ny+1):
#                     testcont1 = xstart + xstep*(xxx+0.5)
#                     testcont2 = ystart + ystep*(yyy+0.5)
#                     logBF = np.zeros(nsim)
#                     DeltalogBF = np.log10(np.e) * ( (-0.5) * ((testcont1 - dX[:,0])/(targDmagerr[0]+0.2))**2) \
#                             / (np.sqrt(2.0*np.pi)*(targDmagerr[0]+0.2))
#                     logBF = logBF + DeltalogBF
#                     DeltalogBF = np.log10(np.e) * ( (-0.5) * ((testcont2 - dX[:,i])/(targDmagerr[i]+0.2))**2) \
#                             / (np.sqrt(2.0*np.pi)*(targDmagerr[i]+0.2))
#                     logBF  = logBF  + DeltalogBF
#                     BF  = 10**logBF
#                     yy = np.where(np.isnan(BF) == False)[0]
#                     BFtot = binfrac*np.sum(BF[yy])   / (yy.size)
#                     DXDYmap[yyy,xxx] = BFtot # Number of companions per mag of contrast per mag of contrast

#             DXDYmap = DXDYmap * renorm

#             fig,ax1 = plt.subplots(figsize=(9,8))
#             ax1.axis([ xstart , xend , ystart , yend ])
#             ax1.set_xlabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#             ax1.set_ylabel(r'$\Delta ' + targfilt[i] + '$ (mag)' , fontsize=16)
#             ax1.tick_params(axis='both',which='major',labelsize=12)

#             vvmax = np.max(np.log10(DXDYmap))
#             sortarr = np.sort(DXDYmap , axis=None)[::-1]
#             for j in np.arange(len(sortarr)):
# #                print(j , np.sum(sortarr[0:j])/np.sum(sortarr))
#                 if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                     vvmin = np.log10(sortarr[j])
#                     break
#             print(vvmax,vvmin)
#             im = plt.pcolor( X3 , Y3 , np.log10(DXDYmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#             cb = plt.colorbar(im , orientation='vertical' )
#             cb.set_label(label=r'log$(\frac{N}{mag**2})$',fontsize=18)

#             plt.plot( targDmag[0] , targDmag[i] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )
#             plt.savefig( 'binprob/' + targname + '/' + (targname + '_Bin_'+targfilt[0]+'_'+targfilt[i]+'_Dkde.png') , \
#                             bbox_inches='tight', pad_inches=0.2 , dpi=200)
#             plt.close('all')


# #############
#     if (targDPM is not None):
#         fig,ax1 = plt.subplots(figsize=(9,8))
        
#         pmsize = np.amax(targDPM) + 5.0
        
#         #if ( np.linalg.norm(targDPM) > 5.0):
# 	    #    pmsize = np.max( abs(targDPM) ) + 3.0
#         ax1.axis([ Pgaia['pmra']-pmsize , Pgaia['pmra']+pmsize , Pgaia['pmdec']-pmsize , Pgaia['pmdec']+pmsize ])
#         ax1.set_xlabel(r'PMRA (mas)' , fontsize=16)
#         ax1.set_ylabel(r'PMDE (mas)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         ccc = ax1.scatter( Pgaia['pmra'] + np.random.normal( 0.0 , sPMerr[0:10000] )  , \
#                   Pgaia['pmdec'] + np.random.normal( 0.0 , sPMerr[0:10000] ) , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( Pgaia['pmra'] + targDPM[0] , Pgaia['pmdec'] + targDPM[1] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_BinPMD.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')

#     if (targDPM is not None):

#         nx=30
#         ny=30
#         xstart = Pgaia['pmra']-pmsize
#         xend   = Pgaia['pmra']+pmsize
#         ystart = Pgaia['pmdec']-pmsize
#         yend   = Pgaia['pmdec']+pmsize
#         xstep = (xend - xstart) / (nx)
#         ystep = (yend - ystart) / (ny)
#         renorm = np.abs((1.0 / xstep) * (1.0 / ystep))
        
#         print(xstart,xend,xstep )
#         print(ystart,yend,ystep)


#         X3,Y3 = np.meshgrid( np.linspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#         PMmap = np.zeros([nx+1,ny+1])
#         for xxx in np.arange(nx+1):
#             for yyy in np.arange(ny+1):
#                 testPMx = xstart + xstep*(xxx+0.5)
#                 testPMy = ystart + ystep*(yyy+0.5)
#                 logBF = np.zeros(nsim)
# #                print(testPMx,testPMy)
#                 DeltalogBF = np.log10(np.e) * ( (-0.5) * ( ((Pgaia['pmra'] -testPMx)/sPMerr)**2 +  \
#                                                            ((Pgaia['pmdec']-testPMy)/sPMerr)**2 ) )\
#                     - np.log10(2.0*np.pi*sPMerr**2)
#                 logBF  = logBF  + DeltalogBF
#                 BF  = 10**logBF
#                 yy = np.where(np.isnan(BF) == False)[0]
#                 BFtot = binfrac*np.sum(BF[yy])   / (yy.size)
#                 PMmap[yyy,xxx] = BFtot # Number of companions per mag of contrast per dex of separation

#         PMmap = PMmap * renorm	


#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ xstart , xend , ystart , yend ])
#         ax1.set_xlabel(r'PMRA (mas)' , fontsize=16)
#         ax1.set_ylabel(r'PMDE (mas)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         vvmax = np.log10(np.max(PMmap))
#         sortarr = np.sort(PMmap , axis=None)[::-1]
#         for j in np.arange(len(sortarr)):
# #            print(j , np.sum(sortarr[0:j])/np.sum(sortarr))
#             if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                 vvmin = np.log10(sortarr[j])
#                 break
#         print(10**vvmin)
#         yyy = np.where(PMmap < 10**vvmin)
#         print(yyy)
#         PMmap[yyy[0]] = vvmin
#         print(vvmax,vvmin)
#         im = plt.pcolor( X3 , Y3 , np.log10(PMmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#         cb = plt.colorbar(im , orientation='vertical' )
#         cb.set_label(label=r'log$(\frac{N}{(mas/yr)**2})$',fontsize=18)

# #        ccc = ax1.scatter( Pgaia['pmra'] + np.random.normal( 0.0 , sPMerr )  , \
# #                  Pgaia['pmdec'] + np.random.normal( 0.0 , sPMerr ) , \
# #                  s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( Pgaia['pmra'] + targDPM[0] , Pgaia['pmdec'] + targDPM[1] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_BinPMDkde.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')

#     print('zzzz')



# #############
#     if (targDPI is not None):
#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ Pgaia['parallax']-4.0 , Pgaia['parallax']+4.0 , 0.0 , (max(dX[:,0])+0.2) ])
#         ax1.set_xlabel(r'Parallax (mas)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         ccc = ax1.scatter((1000.0/Pdist) + np.random.normal( 0.0 , sPIerr )  , \
#                   dX[:,0] + np.random.normal( 0.0 , (targDmagerr[0]+0.1) , nsim ) , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')

#         plt.plot( Pgaia['parallax'] + targDPI , targDmag[0] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )


#         plt.savefig('binprob/' + targname + '/' + targname + '_BinGPID.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')

# #############
#     if (targfilt.size > 0): # Note, I think one contrast and a separation are always needed.

#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ 10**-5.0, (100000.0/Pdist) , (max(dX[:,0])+0.2) , 0.0 ])
#         ax1.set_xlabel(r'Proj. Sep. (arcsec)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)
#         ax1.set_xscale('log')

#         ccc = ax1.scatter(rhoAS[0:10000]  , \
#                   dX[0:10000,0] + np.random.normal( 0.0 , (targDmagerr[0]+0.1) , 10000 ) , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( targsep , targDmag[0] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_BinGrhoD.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')

#     if (targfilt.size > 0): # Note, I think one contrast and a separation are always needed.

#         nx=30
#         ny=30
#         xstart = -3.0
#         xend   = np.log10(100000.0/Pdist)
#         ystart = max(dX[:,0])+0.2
#         yend   = 0.0
#         xstep = (xend - xstart) / (nx)
#         ystep = (yend - ystart) / (ny)
#         renorm = np.abs((1.0 / xstep) * (1.0 / ystep))

#         X3,Y3 = np.meshgrid( np.logspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#         GrhoDmap = np.ones([nx+1,ny+1])
#         for xxx in np.arange(nx+1):
#             for yyy in np.arange(ny+1):
#                 testlogsep = xstart + xstep*(xxx+0.5)
#                 testcont   = ystart + ystep*(yyy+0.5)
#                 logBF = np.zeros(nsim)
#                 DeltalogBF = np.log10(np.e) * ( (-0.5) * ((testlogsep-np.log10(rhoAS))/0.2)**2 ) / (np.sqrt(2.0*np.pi)*0.2)
#                 logBF = logBF + DeltalogBF
#                 DeltalogBF = np.log10(np.e) * ( (-0.5) * ((testcont - dX[:,0])/(targDmagerr[0]+0.2))**2) \
#                         / (np.sqrt(2.0*np.pi)*(targDmagerr[0]+0.2))
#                 logBF  = logBF  + DeltalogBF
#                 BF  = 10**logBF
#                 yy = np.where(np.isnan(BF) == False)[0]
#                 BFtot = binfrac*np.sum(BF[yy])   / (yy.size)
#                 GrhoDmap[yyy,xxx] = BFtot # Number of companions per mag of contrast per dex of separation

#         GrhoDmap = GrhoDmap[:,:] * renorm

#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ 10**(xstart) , 10**(xend) , ystart , yend ])
#         ax1.set_xlabel(r'Proj. Sep. (arcsec)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)
#         ax1.set_xscale('log')

#         vvmax = np.max(np.log10(GrhoDmap))
#         sortarr = np.sort(GrhoDmap , axis=None)[::-1]
#         for j in np.arange(len(sortarr)):
# #            print(j , np.sum(sortarr[0:j])/np.sum(sortarr))
#             if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                 vvmin = np.log10(sortarr[j])
#                 break
#         print(vvmax,vvmin)
#         im = plt.pcolor( X3 , Y3 , np.log10(GrhoDmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#         cb = plt.colorbar(im , orientation='vertical' )
#         cb.set_label(label=r'log$(\frac{N}{mag \times dex})$',fontsize=18)

#         print(np.log10(GrhoDmap[:,15]))
#         print('wwwww')


#         plt.plot( targsep , targDmag[0] , 'rx' , \
#                   markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_BinGrhoDkde.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')


# ### Query Gaia for field population

#     smin = (40000.0/Pdist)
#     smax = max([3600.0 , 206265.0/Pdist])
#     print('Search inner/outer radii in arcsec: ',smin,smax)
#     Garea = np.pi * ( smax**2 - smin**2 )
#     print('Total search area: ',Garea)
#     print('Note, for binaries simulating one million but expect 0.46. So can rescale field down by  within 100,000 AU. So can compute a separation rescaling.')
#     Gscale = np.sqrt( 0.46 / 10000.0)
#     print('Rescaling inward of field population: ',Gscale)

#     # Query Gaia FULL DR3 unless already downloaded
#     testname = 'binprob/GaiaDL/' + targname + '_DR3.pickle'
#     if ((os.path.isfile(os.path.expanduser(testname))) == True) : 
#         print('Already downloaded Gaia DR3 query.')
#         with open(testname , 'rb') as gaiadata:
#             r = pickle.load(gaiadata)
#     if ((os.path.isfile(os.path.expanduser(testname))) == False):
#         sqltext = "SELECT * FROM gaiadr3.gaia_source WHERE CONTAINS( \
#             POINT('ICRS',gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec), \
#             CIRCLE('ICRS'," + str(Pcoord.ra.value) +","+ str(Pcoord.dec.value) +","+ str(smax/3600.0) +"))=1;"
#         print(sqltext)
#         job = Gaia.launch_job_async(sqltext , dump_to_file=False)
#         r = job.get_results()
#         with open(testname , 'wb') as gaiadata:
#             pickle.dump(r['ra','dec','parallax','parallax_error','parallax_over_error','pmra','pmdec',\
#                           'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag', \
#                           'phot_bp_mean_flux_over_error','phot_rp_mean_flux_over_error'] , gaiadata)


#     r2 = r
#     #print(r[0:10].pprint_all())
#     print('Number of entries: ',np.array(r['ra']).size)



# ### Compute field population properties

#     r = r2

#     zz = np.where(np.isnan(r['phot_g_mean_mag']) == False)[0]
#     print('Number of entries without Gmag: ', (np.array(r['ra']).size - zz.size))
#     r = r[zz]

#     # Predict mags in other filters
#     fT = np.zeros([len(r['ra'])])
#     fT[:] = np.nan
#     fdist = np.zeros([len(r['ra'])])
#     fdist[:] = np.nan
#     fAV = np.zeros([len(r['ra'])])
#     fAV[:] = np.nan
#     fMG = np.zeros([len(r['ra'])])
#     fMG[:] = np.nan
#     fAGAV = np.zeros([len(r['ra'])])
#     fAGAV[:] = 0.822
#     fAXAV = np.zeros([len(r['ra']),targfilt.size])
#     fAXAV[:,:] = np.nan
#     fMX = np.zeros([len(r['ra']),targfilt.size])
#     fMX[:,:] = np.nan
#     fX = np.zeros([len(r['ra']),targfilt.size])
#     fX[:,:] = np.nan
#     fdX = np.zeros([len(r['ra']),targfilt.size])
#     fdX[:,:] = np.nan
#     fdXerr = np.zeros([len(r['ra']),targfilt.size])
#     fdXerr[:,:] = np.nan
#     fcalctype = np.array(["    " for i in range(len(r['ra']))])
#     frhoAS = np.empty(np.array(r['ra']).size)
#     frhoAS[:] = np.nan 

#     yy = np.where( (np.isnan(fdX[:,0]) == True) & (r['parallax_over_error'] > 10.0))[0]
#     print('Number with distance-based properties: ',yy.size)
#     if (yy.size > 0):
#         fdist[yy] = (1000.0/r['parallax'][yy])
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         frhoAS[yy] = Pcoord.separation(fcoord).to(u.arcsecond).value
#         # Rough props
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         fMG[yy] = r['phot_g_mean_mag'][yy] - (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAGAV[yy]
#         zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(T_Mama) == False) )[0]
#         fT[yy]  = np.interp( fMG[yy] , MG_Mama[zz] , T_Mama[zz])
#         # Compute extinctions
#         zz = np.where( (np.isnan(T_Mama) == False) )[0]
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         filtloc = np.where( filtarr == 'G' )[0][0]
#         fAGAV[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         # Compute final MG and final temperature
#         fMG[yy] = r['phot_g_mean_mag'][yy] - (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAGAV[yy]
#         zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(T_Mama) == False) )[0]
#         fT[yy]   = np.interp( fMG[yy] , MG_Mama[zz] , T_Mama[zz])
#         zz = np.where( (np.isnan(T_Mama) == False) )[0]
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(np.ravel(photarr[:,filtloc])) == False))[0]
#             fMX[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , \
#                             np.flip(np.ravel(photarr[zz,filtloc])) , left=np.nan , right=np.nan )
#             fX[yy,j]  = fMX[yy,j] + (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAXAV[yy,j]
#             fdX[yy,j] = fX[yy,j] - Pmag[j]
#             fdXerr[yy,j] = 0.1
#             fcalctype[yy] = 'Dist'

#     yy = np.where( (np.isnan(fdX[:,0]) == True) & (r['phot_bp_mean_flux_over_error'][:] > 10.0) & \
#                                               (r['phot_rp_mean_flux_over_error'][:] > 10.0))[0]
#     print('Number with BpRp-based properties: ',yy.size)
#     if (yy.size > 0):     
#         # Estimate preliminary fT from color
#         filtloc1 = np.where( filtarr == 'Bp' )[0][0]
#         filtloc2 = np.where( filtarr == 'Rp' )[0][0]
#         fBpRp = (r['phot_bp_mean_mag'][yy] - r['phot_rp_mean_mag'][yy])
#         zz = np.where( (np.isnan(np.ravel(photarr[:,filtloc1])) == False) & \
#                        (np.isnan(np.ravel(photarr[:,filtloc2])) == False) & \
#                        (np.isnan(T_Mama) == False) )
#         fT[yy] = np.interp( fBpRp , (np.ravel(photarr[zz,filtloc1]) - np.ravel(photarr[zz,filtloc2])) , T_Mama[zz])
#         # Estimate preliminary extinctions coefficients and preliminary distance from fT
#         zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(T_Mama) == False) )
#         filtloc = np.where( filtarr == 'G' )[0][0]
#         fAGAV[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc] )) )
#         fABAV     = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc1])) )
#         fARAV     = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc2])) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         fMG[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]) )
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+5.0 )/5.0)
#         # Estimate AV and compute projected separation
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         frhoAS[yy] = Pcoord.separation(fcoord).to(u.arcsecond).value
#         # Estimate final fT
#         zz = np.where( (np.isnan(np.ravel(photarr[:,filtloc1])) == False) & \
#                        (np.isnan(np.ravel(photarr[:,filtloc2])) == False) & \
#                        (np.isnan(T_Mama) == False) )
#         fBpRp = (r['phot_bp_mean_mag'][yy] - r['phot_rp_mean_mag'][yy]) - fAV[yy]*(fABAV-fARAV)
#         fT[yy] = np.interp( fBpRp , (np.ravel(photarr[zz,filtloc1]) - np.ravel(photarr[zz,filtloc2])) , T_Mama[zz])
#         # Estimate final extinction coefficients and distance
#         zz = np.where( (np.isnan(T_Mama) == False) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         fMG[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]) )
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+fAV[yy]*fAGAV[yy]+5.0 )/5.0)
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         # Compute final abs mag, apparent mag, and contrast
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(np.ravel(photarr[:,filtloc])) == False))[0]
#             fMX[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , \
#                             np.flip(np.ravel(photarr[zz,filtloc])) , left=np.nan , right=np.nan )
#             fX[yy,j]  = fMX[yy,j] + (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAXAV[yy,j]
#             fdX[yy,j] = fX[yy,j] - Pmag[j]
#             fdXerr[yy,j] = 0.1
#         fcalctype[yy] = 'BpRp'

#     yy = np.where( (np.isnan(fdX[:,0]) == True) & (r['phot_rp_mean_flux_over_error'][:] > 10.0))[0]
#     print('Number with mGRp-based properties: ',yy.size)
#     if (yy.size > 0):     
#         # Estimate preliminary fT from color
#         filtloc1 = np.where( filtarr == 'G' )[0][0]
#         filtloc2 = np.where( filtarr == 'Rp' )[0][0]
#         fmGRp = (r['phot_g_mean_mag'][yy] - r['phot_rp_mean_mag'][yy])
#         zz = np.where( (np.isnan(np.ravel(photarr[:,filtloc1])) == False) & \
#                        (np.isnan(np.ravel(photarr[:,filtloc2])) == False) & \
#                        (np.isnan(T_Mama) == False) )
#         fT[yy] = np.interp( fmGRp , (np.ravel(photarr[zz,filtloc1]) - np.ravel(photarr[zz,filtloc2])) , T_Mama[zz])
#         # Estimate preliminary extinctions coefficients and preliminary distance from fT
#         zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(T_Mama) == False) )
#         filtloc = np.where( filtarr == 'G' )[0][0]
#         fAGAV[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc] )) )
#         fARAV     = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc2])) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         fMG[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]) )
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+5.0 )/5.0)
#         # Estimate AV and compute projected separation
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         frhoAS[yy] = Pcoord.separation(fcoord).to(u.arcsecond).value
#         # Estimate final fT
#         zz = np.where( (np.isnan(np.ravel(photarr[:,filtloc1])) == False) & \
#                        (np.isnan(np.ravel(photarr[:,filtloc2])) == False) & \
#                        (np.isnan(T_Mama) == False) )
#         fmGRp = (r['phot_g_mean_mag'][yy] - r['phot_rp_mean_mag'][yy]) - fAV[yy]*(fAGAV[yy]-fARAV)
#         fT[yy] = np.interp( fmGRp , (np.ravel(photarr[zz,filtloc1]) - np.ravel(photarr[zz,filtloc2])) , T_Mama[zz])
#         # Estimate final extinction coefficients and distance
#         zz = np.where( (np.isnan(T_Mama) == False) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         fMG[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]) )
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+fAV[yy]*fAGAV[yy]+5.0 )/5.0)
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         # Compute final abs mag, apparent mag, and contrast
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(np.ravel(photarr[:,filtloc])) == False))[0]
#             fMX[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , \
#                             np.flip(np.ravel(photarr[zz,filtloc])) , left=np.nan , right=np.nan )
#             fX[yy,j]  = fMX[yy,j] + (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAXAV[yy,j]
#             fdX[yy,j] = fX[yy,j] - Pmag[j]
#             fdXerr[yy,j] = 0.1
#         fcalctype[yy] = 'mGRp' 

#     yy = np.where( (np.isnan(fdX[:,0]) == True) )[0]
#     print('Number with Teff-based properties: ',yy.size)
#     if (yy.size > 0):     
#         fT[yy] = 4500.0
#         # Compute extinction coefficients
#         zz = np.where( (np.isnan(T_Mama) == False) )
#         filtloc = np.where( filtarr == 'G' )[0][0]
#         fAGAV[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             fAXAV[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(np.ravel(AXAVarr[zz,filtloc])) )
#         # Compute MG and then fdist
#         zz = np.where( (np.isnan(MG_Mama) == False) & (np.isnan(T_Mama) == False) )
#         fMG[yy] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , np.flip(MG_Mama[zz]) )
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+5.0 )/5.0)
#         # Compute actual AV, then update fdist and compute projected sep
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         print(fcoord[0])
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         frhoAS[yy] = Pcoord.separation(fcoord).to(u.arcsecond).value
#         fdist[yy] = 10**(( r['phot_g_mean_mag'][yy]-fMG[yy]+fAV[yy]*fAGAV[yy]+5.0 )/5.0)
#         fcoord = SkyCoord( ra=np.array(r['ra'][yy])*u.deg , dec=np.array(r['dec'][yy])*u.deg , \
#                                distance=np.array(fdist[yy])*u.parsec , frame='icrs' )
#         fAV[yy] = bayestar(fcoord , mode='median') * 2.742
#         zz = np.where( np.isnan(fAV[yy]) == True )[0]
#         if (zz.size > 0):
#             print('Nan is reset to zero for: ',zz.size)
#             fAV[yy[zz]] = 0.0

#         # Compute final abs mag, apparent mag, and contrast
#         for j in range(0,targfilt.size):
#             filtloc = np.where( filtarr == targfilt[j] )[0][0]
#             zz = np.where( (np.isnan(T_Mama) == False) & (np.isnan(np.ravel(photarr[:,filtloc])) == False))[0]
#             fMX[yy,j] = np.interp( fT[yy] , np.flip(T_Mama[zz]) , \
#                             np.flip(np.ravel(photarr[zz,filtloc])) , left=np.nan , right=np.nan )
#             fX[yy,j]  = fMX[yy,j] + (5.0*np.log10(fdist[yy])-5.0) - fAV[yy]*fAXAV[yy,j]
#             fdX[yy,j] = fX[yy,j] - Pmag[j]
#             fdXerr[yy,j] = 0.5
#         fcalctype[yy] = 'Teff'


# ### Field interloper plots

# #######################
#     if (targfilt.size > 1):
#         for i in range(1,targfilt.size):
#             fig,ax1 = plt.subplots(figsize=(9,8))
#             ax1.axis([ 0.0 , (max(fdX[:,0])+0.2) , 0.0 , (max(fdX[:,i])+0.2) ])
#             ax1.set_xlabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#             ax1.set_ylabel(r'$\Delta ' + targfilt[i] + '$ (mag)' , fontsize=16)
#             ax1.tick_params(axis='both',which='major',labelsize=12)

#             zz = np.where( fcalctype == 'Teff')[0]
#             ccc = ax1.scatter( fdX[zz,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[zz,0]) , fdX[zz,0].size ) , \
#                            fdX[zz,i] + np.random.normal( 0.0 , (targDmagerr[i]+fdXerr[zz,i]) , fdX[zz,0].size)  , \
#                            s=2 , c='black' , marker='o' , label='Field (Assumed Teff)')
#             zz = np.where( fcalctype == 'mGRp')[0]
#             ccc = ax1.scatter( fdX[zz,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[zz,0]) , fdX[zz,0].size ) , \
#                            fdX[zz,i] + np.random.normal( 0.0 , (targDmagerr[i]+fdXerr[zz,i]) , fdX[zz,0].size)  , \
#                            s=2 , c='orange' , marker='o' , label='Field (Using G-Rp)')
#             zz = np.where( fcalctype == 'BpRp')[0]
#             ccc = ax1.scatter( fdX[zz,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[zz,0]) , fdX[zz,0].size ) , \
#                            fdX[zz,i] + np.random.normal( 0.0 , (targDmagerr[i]+fdXerr[zz,i]) , fdX[zz,0].size)  , \
#                            s=2 , c='green' , marker='o' , label='Field (Using Bp-Rp)')
#             zz = np.where( fcalctype == 'Dist')[0]
#             ccc = ax1.scatter( fdX[zz,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[zz,0]) , fdX[zz,0].size ) , \
#                            fdX[zz,i] + np.random.normal( 0.0 , (targDmagerr[i]+fdXerr[zz,i]) , fdX[zz,0].size)  , \
#                            s=2 , c='blue' , marker='o' , label='Field (Using Dist)')
        
        
#             plt.plot( targDmag[0] , targDmag[i] , 'rx' , \
#                  markersize=15 , mew=3 , markeredgecolor='red' , zorder=3 , label=(targname+' cand'))

#             lgnd = plt.legend()
#             for lh in lgnd.legendHandles:
#                 lh._sizes = [25.0]
#             plt.savefig( 'binprob/' + targname + '/' + (targname + '_Fld_'+targfilt[0]+'_'+targfilt[i]+'_D.png') , \
#                             bbox_inches='tight', pad_inches=0.2 , dpi=200)
#             plt.close('all')


#             nx=30
#             ny=30
#             xstart = 0.0
#             xend   = max(fdX[:,0])+0.2
#             ystart = 0.0
#             yend   = max(fdX[:,i])+0.2
#             xstep = (xend - xstart) / (nx)
#             ystep = (yend - ystart) / (ny)
#             renorm = np.abs((1.0 / xstep) * (1.0 / ystep))

#             fig,ax1 = plt.subplots(figsize=(9,8))
#             ax1.axis([ xstart , xend , ystart , yend ])
#             ax1.set_xlabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#             ax1.set_ylabel(r'$\Delta ' + targfilt[i] + '$ (mag)' , fontsize=16)
#             ax1.tick_params(axis='both',which='major',labelsize=12)


#             X3,Y3 = np.meshgrid( np.linspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#             DXDYmap = np.zeros([nx+1,ny+1])
#             for xxx in np.arange(nx+1):
#                 for yyy in np.arange(ny+1):
#                     testcont1 = xstart + xstep*(xxx+0.5)
#                     testcont2 = ystart + ystep*(yyy+0.5)
#                     logFF = np.zeros(frhoAS.size)
#                     DeltalogFF = np.log10(np.e) * ( (-0.5) * ((testcont1 - fdX[:,0])/(targDmagerr[0]+fdXerr[:,0]))**2) \
#                               / (np.sqrt(2.0*np.pi)*(targDmagerr[0]+fdXerr[:,0]))
#                     logFF = logFF + DeltalogFF
#                     DeltalogFF = np.log10(np.e) * ( (-0.5) * ((testcont2 - fdX[:,i])/(targDmagerr[i]+fdXerr[:,i]))**2) \
#                               / (np.sqrt(2.0*np.pi)*(targDmagerr[i]+fdXerr[:,i]))
#                     logFF = logFF + DeltalogFF


#                     FF  = 10**logFF
#                     yy = np.where(np.isnan(FF) == False)[0]
#                     FFtot = binfrac*np.sum(FF[yy])   / (yy.size)
#                     DXDYmap[yyy,xxx] = FFtot # Number of companions per mag of contrast per mag of contrast

#             DXDYmap = DXDYmap * renorm	

#             vvmax = np.max(np.log10(DXDYmap))
#             sortarr = np.sort(DXDYmap , axis=None)[::-1]
#             for j in np.arange(len(sortarr)):
# #                print(j , np.sum(sortarr[0:j])/np.sum(sortarr))
#                 if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                     vvmin = np.log10(sortarr[j])
#                     break
#             print(vvmax,vvmin)
#             im = plt.pcolor( X3 , Y3 , np.log10(DXDYmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#             cb = plt.colorbar(im , orientation='vertical' )
#             cb.set_label(label=r'log$(\frac{N}{mag**2})$',fontsize=18)
        
#             plt.plot( targDmag[0] , targDmag[i] , 'rx' , \
#                  markersize=15 , mew=3 , markeredgecolor='red' , zorder=3 , label=(targname+' cand'))

#             lgnd = plt.legend()
#             for lh in lgnd.legendHandles:
#                 lh._sizes = [25.0]
#             plt.savefig( 'binprob/' + targname + '/' + (targname + '_Fld_'+targfilt[0]+'_'+targfilt[i]+'_Dkde.png') , \
#                             bbox_inches='tight', pad_inches=0.2 , dpi=200)
#             plt.close('all')


# #################################
#     if (targDPM is not None):
#         fig,ax1 = plt.subplots(figsize=(9,8))
#         pmsize = np.amax(abs(targDPM)) + 5.0
#         ax1.axis([ Pgaia['pmra']-pmsize , Pgaia['pmra']+pmsize , Pgaia['pmdec']-pmsize , Pgaia['pmdec']+pmsize ])
#         ax1.set_xlabel(r'PMRA (mas)' , fontsize=16)
#         ax1.set_ylabel(r'PMDE (mas)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         ccc = ax1.scatter( r['pmra']  , r['pmdec'] , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( Pgaia['pmra'] + targDPM[0] , Pgaia['pmdec'] + targDPM[1] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_FldPMD.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')


#         fig,ax1 = plt.subplots(figsize=(9,8))

#         nx=30
#         ny=30
#         xstart = Pgaia['pmra']-pmsize
#         xend   = Pgaia['pmra']+pmsize
#         ystart = Pgaia['pmdec']-pmsize
#         yend   = Pgaia['pmdec']+pmsize
#         xstep = (xend - xstart) / (nx)
#         ystep = (yend - ystart) / (ny)
#         renorm = np.abs((1.0 / xstep) * (1.0 / ystep))
        
#         print(xstart,xend,xstep )
#         print(ystart,yend,ystep)


#         X3,Y3 = np.meshgrid( np.linspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#         PMmap = np.zeros([nx+1,ny+1])
#         for xxx in np.arange(nx+1):
#             for yyy in np.arange(ny+1):
#                 testPMx = xstart + xstep*(xxx+0.5)
#                 testPMy = ystart + ystep*(yyy+0.5)
#                 logFF = np.zeros(frhoAS.size)
#                 DeltalogFF = np.log10(np.e) * ( (-0.5) * (((Pgaia['pmra']  + testPMx - r['pmra'] )/np.amax(np.array([1.0 , np.median(sPMerr)])) )**2 + \
#                                                           ((Pgaia['pmdec'] + testPMy - r['pmdec'])/np.amax(np.array([1.0 , np.median(sPMerr)])) )**2)) \
#                                                          - np.log10(2.0*np.pi*np.amax(np.array([1.0 , np.median(sPMerr)]))**2)
# #                                                        - np.log10(2.0*np.pi*1.0**2)
#                 logFF  = logFF  + DeltalogFF
#                 FF  = 10**logFF
#                 yy = np.where(np.isnan(FF) == False)[0]
#                 FFtot = binfrac*np.sum(FF[yy])   / (yy.size)
#                 PMmap[yyy,xxx] = FFtot # Number of companions per mag of contrast per dex of separation

#         PMmap = PMmap * renorm	

#         ax1.axis([ xstart , xend , ystart , yend ])
#         ax1.set_xlabel(r'PMRA (mas)' , fontsize=16)
#         ax1.set_ylabel(r'PMDE (mas)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         vvmax = np.log10(np.max(PMmap))
#         sortarr = np.sort(PMmap , axis=None)[::-1]
#         print(np.sum(sortarr))
#         print('wwww')
#         for j in np.arange(len(sortarr)):
# #            print(j , np.sum(sortarr[0:j])/np.sum(sortarr) , sortarr[j])
#             if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                 print(sortarr[j])
#                 vvmin = np.log10(sortarr[j])
#                 break
#         print(vvmax,vvmin)
#         im = plt.pcolor( X3 , Y3 , np.log10(PMmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#         cb = plt.colorbar(im , orientation='vertical' )
#         cb.set_label(label=r'log$(\frac{N}{(mas/yr)**2})$',fontsize=18)

# #        ccc = ax1.scatter( r['pmra']  , r['pmdec'] , \
# #                  s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( Pgaia['pmra'] + targDPM[0] , Pgaia['pmdec'] + targDPM[1] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_FldPMDkde.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')



# #################################
#     if (targDPI is not None):
#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ Pgaia['parallax']-4.0 , Pgaia['parallax']+4.0 , 0.0 , (max(fdX[:,0])+0.2) ])
#         ax1.set_xlabel(r'Parallax (mas)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)

#         ccc = ax1.scatter(r['parallax']  , \
#                   r['phot_g_mean_mag'] - Pgaia['phot_g_mean_mag'] , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( Pgaia['parallax'] + targDPI , targDmag[0] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_FldGPID.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')



# ###########################
#     if (targfilt.size > 0):
#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ 10**-5.0, (100000.0/Pdist) , (max(dX[:,0])+0.2) , 0.0 ])
# #        ax1.axis([ 1.0, smax , 0.0 , (max(fdX[:,0])+0.2) ])
#         ax1.set_xlabel(r'Proj. Sep. (arcsec)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)
#         ax1.set_xscale('log')

# #        ccc = ax1.scatter(frhoAS  , fdX[:,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[:,0]) , fdX[:,0].size ) , \
#         ccc = ax1.scatter(frhoAS*Gscale  , fdX[:,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[:,0]) , fdX[:,0].size ) , \
#                   s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( targsep , targDmag[0] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_FldGrhoD.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')


#     if (targfilt.size > 0):
#         nx=30
#         ny=30
#         xstart = -3.0
#         xend   = np.log10(100000.0/Pdist)
#         ystart = max(dX[:,0])+0.2
#         yend   = 0.0
#         xstep = (xend - xstart) / (nx)
#         ystep = (yend - ystart) / (ny)
#         renorm = np.abs((1.0 / xstep) * (1.0 / ystep))

#         X3,Y3 = np.meshgrid( np.logspace(xstart,xend,nx+1) , \
#                              np.linspace(ystart,yend,ny+1) )
#         GrhoDmap = np.ones([nx+1,ny+1])
#         for xxx in np.arange(nx+1):
#             for yyy in np.arange(ny+1):
#                 testlogsep = xstart + xstep*(xxx+0.5)
#                 testcont   = ystart + ystep*(yyy+0.5)
#                 logFF = np.zeros(frhoAS.size)
#                 DeltalogFF = np.ones(frhoAS.size) * np.log10((frhoAS.size / (np.pi * (smax**2 - smin**2))) \
#                                      * 2.0 * np.pi * (10**testlogsep)**2 / np.log10(np.e))
#                 logFF = logFF + DeltalogFF
#                 DeltalogFF = np.log10(np.e) * ( (-0.5) * ((testcont - fdX[:,i])/(targDmagerr[i]+fdXerr[:,i]))**2) \
#                              / (np.sqrt(2.0*np.pi)*(targDmagerr[i]+fdXerr[:,i]))
#                 logFF  = logFF  + DeltalogFF
#                 FF  = 10**logFF
#                 yy = np.where(np.isnan(FF) == False)[0]
#                 FFtot = binfrac*np.sum(FF[yy])   / (yy.size)
#                 GrhoDmap[yyy,xxx] = FFtot # Number of companions per mag of contrast per dex of separation

#         GrhoDmap = GrhoDmap * renorm	

#         fig,ax1 = plt.subplots(figsize=(9,8))
#         ax1.axis([ 10**(xstart) , 10**(xend) , ystart , yend ])
#         ax1.set_xlabel(r'Proj. Sep. (arcsec)' , fontsize=16)
#         ax1.set_ylabel(r'$\Delta ' + targfilt[0] + '$ (mag)' , fontsize=16)
#         ax1.tick_params(axis='both',which='major',labelsize=12)
#         ax1.set_xscale('log')

#         vvmax = np.max(np.log10(GrhoDmap))
#         sortarr = np.sort(GrhoDmap , axis=None)[::-1]
#         for j in np.arange(len(sortarr)):
# #            print(j , np.sum(sortarr[0:j])/np.sum(sortarr))
#             if ( (np.sum(sortarr[0:j])/np.sum(sortarr)) > 0.999 ):
#                 vvmin = np.log10(sortarr[j])
#                 break
#         print(vvmax,vvmin)
#         im = plt.pcolor( X3 , Y3 , np.log10(GrhoDmap) , cmap='cubehelix_r' , vmax=vvmax, vmin=vvmin )
#         cb = plt.colorbar(im , orientation='vertical' )
#         cb.set_label(label=r'log$(\frac{N}{mag \times dex})$',fontsize=18)

# #        ccc = ax1.scatter(frhoAS*Gscale  , fdX[:,0] + np.random.normal( 0.0 , (targDmagerr[0]+fdXerr[:,0]) , fdX[:,0].size ) , \
# #                  s=2 , c='black' , marker='+' , label='Simulated bins')
#         plt.plot( targsep , targDmag[0] , 'rx' , \
#                  markersize=12 , mew=3 , markeredgecolor='red' , zorder=3 , label=targname )

#         plt.savefig('binprob/' + targname + '/' + targname + '_FldGrhoDkde.png',bbox_inches='tight', pad_inches=0.2 , dpi=200)
#         plt.close('all')


# ### Create vectorized KDE

#     logBF = np.zeros(nsim)
#     logFF = np.zeros(frhoAS.size)

#     # Projected separation term
#     DeltalogBF = np.log10(np.e) * ( (-0.5) * ((np.log10(targsep)-np.log10(rhoAS))/0.2)**2 ) / (np.sqrt(2.0*np.pi)*0.2)
#     DeltalogFF = np.ones(frhoAS.size) * np.log10((frhoAS.size / (np.pi * (smax**2 - smin**2))) * 2.0 * np.pi * targsep**2 / np.log10(np.e))
#     logBF = logBF + DeltalogBF
#     logFF = logFF + DeltalogFF
#     logBFA = logBF
#     logBFP = logBF
#     logBFS = logBF
#     logFFA = logFF
#     logFFP = logFF
#     logFFS = logFF

#     # Contrast term
#     for i in range(0,targfilt.size):
#         DeltalogBF = np.log10(np.e) * ( (-0.5) * ((targDmag[i] - dX[:,i])/(targDmagerr[i]+0.2))**2) \
#                     / (np.sqrt(2.0*np.pi)*(targDmagerr[i]+0.2))
#         DeltalogFF = np.log10(np.e) * ( (-0.5) * ((targDmag[i] - fdX[:,i])/(targDmagerr[i]+fdXerr[:,i]))**2) \
#                     / (np.sqrt(2.0*np.pi)*(targDmagerr[i]+fdXerr[:,i]))
#         logBF  = logBF  + DeltalogBF
#         logFF  = logFF  + DeltalogFF
#         logBFP = logBFP + DeltalogBF
#         logFFP = logFFP + DeltalogFF
#         if (i == 0):
#             logBFA = logBFA + DeltalogBF
#             logFFA = logFFA + DeltalogFF
#             logBFS = logBFS + DeltalogBF
#             logFFS = logFFS + DeltalogFF
    
#     # Proper motion term
#     if (targDPM is not None):
#         DeltalogBF = np.log10(np.e) * ( (-0.5) * ((targDPM[0]/sPMerr)**2 + (targDPM[1]/sPMerr)**2)) \
#                 - np.log10(2.0*np.pi*sPMerr**2)

#         PMRA = Pgaia['pmra']
#         PMDec = Pgaia['pmdec']


#         print('zzzzzz')
#         print('Note, KDE bandwidth for binary PMD is: ',np.median(sPMerr))

#         DeltalogFF = np.log10(np.e) * ( (-0.5) * (((Pgaia['pmra']  + targDPM[0] - r['pmra'] )/np.amax(np.array([1.0 , np.median(sPMerr)])) )**2 + \
#                                               ((Pgaia['pmdec'] + targDPM[1] - r['pmdec'])    /np.amax(np.array([1.0 , np.median(sPMerr)])) )**2)) \
#                 - np.log10(2.0*np.pi*np.amax(np.array([1.0 , np.median(sPMerr)]))**2)
# #                - np.log10(2.0*np.pi*1.0**2)
#         print('Note, KDE bandwidth for field PMD is: ',np.amax(np.array([1.0 , np.median(sPMerr)])) )
#         print('I should think more about the KDE bandwidth for the field PMD calculation. Do I use the observational uncertainty? I think yes.')

#         logBF  = logBF  + DeltalogBF
#         logFF  = logFF  + DeltalogFF
#         logBFA = logBFA + DeltalogBF
#         logFFA = logFFA + DeltalogFF

#     # Parallax term
#     if (targDPI is not None):
#         DeltalogBF = np.log10(np.e) * ((-0.5)*(targDPI/sPIerr)**2) \
#                                         - np.log10(2.0*np.pi*sPIerr**2)
#         DeltalogFF = np.log10(np.e) * ((-0.5)*((targDPI+Pgaia['parallax']-r['parallax'])/(0.1+r['parallax_error']))**2) \
#                                         - np.log10(2.0*np.pi*1.0**2)
#         logBF  = logBF  + DeltalogBF
#         logFF  = logFF  + DeltalogFF
#         logBFA = logBFA + DeltalogBF
#         logFFA = logFFA + DeltalogFF

#     BF  = 10**logBF
#     BFP = 10**logBFP
#     BFA = 10**logBFA
#     BFS = 10**logBFS
#     FF  = 10**logFF
#     FFP = 10**logFFP
#     FFA = 10**logFFA
#     FFS = 10**logFFS

#     yy = np.where(np.isnan(BF) == False)[0]
#     BFtot = binfrac*np.sum(BF[yy])   / (yy.size)
#     yy = np.where(np.isnan(BFP) == False)[0]
#     BFtotP = binfrac*np.sum(BFP[yy]) / (yy.size)
#     yy = np.where(np.isnan(BFA) == False)[0]
#     BFtotA = binfrac*np.sum(BFA[yy]) / (yy.size)
#     yy = np.where(np.isnan(BFS) == False)[0]
#     BFtotS = binfrac*np.sum(BFS[yy]) / (yy.size)

#     yy = np.where(np.isnan(FF) == False)[0]
#     FFtot = np.sum(FF[yy])   / (yy.size)
#     yy = np.where(np.isnan(FFP) == False)[0]
#     FFtotP = np.sum(FFP[yy]) / (yy.size)
#     yy = np.where(np.isnan(FFA) == False)[0]
#     FFtotA = np.sum(FFA[yy]) / (yy.size)
#     yy = np.where(np.isnan(FFS) == False)[0]
#     FFtotS = np.sum(FFS[yy]) / (yy.size)
    
#     print('PDF for binary companions: ',BFtot) # units of companions per mag^2 per (mas/yr)^2 per dex of sep
#     print('PDF for field interlopers: ',FFtot) # units of companions per mag^2 per (mas/yr)^2 per dex of sep

#     print('All: ')
#     print('Probability of field:  ',FFtot/(FFtot+BFtot))
#     print('Probability of binary: ',BFtot/(FFtot+BFtot))

#     print('Survey detection: ')
#     print('Probability of field:  ',FFtotS/(FFtotS+BFtotS))
#     print('Probability of binary: ',BFtotS/(FFtotS+BFtotS))

#     print('Photometry: ')
#     print('Probability of field:  ',FFtotP/(FFtotP+BFtotP))
#     print('Probability of binary: ',BFtotP/(FFtotP+BFtotP))

#     print('Astrometry: ')
#     print('Probability of field:  ',FFtotA/(FFtotA+BFtotA))
#     print('Probability of binary: ',BFtotA/(FFtotA+BFtotA))

#     fmt9 = "%8s %17.15f %11.9f %11.9f %11.9f %17.15f %11.9f %11.9f %11.9f"
#     filename = 'binprobs.txt'
#     with open(filename,'a') as file1:
#         file1.write(fmt9 % (targname , BFtot/(FFtot+BFtot) , BFtotS/(FFtotS+BFtotS) , BFtotP/(FFtotP+BFtotP) , BFtotA/(FFtotA+BFtotA) , \
#                                        FFtot/(FFtot+BFtot) , FFtotS/(FFtotS+BFtotS) , FFtotP/(FFtotP+BFtotP) , FFtotA/(FFtotA+BFtotA) ) )
#         file1.write("\n")

#     print('Returning an array of Pbin (all, survey, phot, astro) and then Pfield (same order).')

#     return np.array([ [BFtot/(FFtot+BFtot) , BFtotS/(FFtotS+BFtotS) , BFtotP/(FFtotP+BFtotP) , BFtotA/(FFtotA+BFtotA)] , \
#                       [FFtot/(FFtot+BFtot) , FFtotS/(FFtotS+BFtotS) , FFtotP/(FFtotP+BFtotP) , FFtotA/(FFtotA+BFtotA)] ])