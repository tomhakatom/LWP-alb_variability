
import numpy as np
import netCDF4 as nc4
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
#import seviri_support
import misc_codes
from scipy import stats
import read_data
#from roipoly import roipoly
from csat import MODIS
from sklearn.metrics import jaccard_score
import copy
from scipy import signal
#import pysal as ps
import LWP_alb_variability_support
import glob
from scipy import spatial
import os
from matplotlib.colors import LinearSegmentedColormap
import time
import scipy
import LES_support

## Plotting ##
plt.rc('ytick', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('axes', labelsize=20)
plt.rc('axes', titlesize=20)
plt.rc('font', size=20)

########################################################################################################################
## READ LES DATA
########################################################################################################################

# Read LES trajectories
traj_STATS = {}
traj_optics = {}
traj_2D = {}
traj_3D = {}
traj_CF = {}
traj_time_series = {}
traj_time_series_1 = {}
traj_time_series_2 = {}
for NC in ['0001']: # 175,200,225, 250, 350
    path_LES_STATS = '/media/tgoren/TOSHIBA EXT/noaa_home_directory/research/analysis/continental_tracks/LES/data/2018-06-13/Azores_2010-01-30_' + NC + '/OUT_STAT/'
    f_name_STATS_000 = 'Azores_2010-01-30_' + NC + '_ERA5-001-00.nc'
    variables_STATS = ['LAT', 'LON', 'time', 'ALBEDO']
    traj_STATS[NC] = LES_support.get_LES_traj(path_LES_STATS, f_name_STATS_000, variables_STATS)

    path_LES_optics = '/media/tgoren/TOSHIBA EXT/noaa_home_directory/research/analysis/continental_tracks/LES/data/2018-06-13/Azores_2010-01-30_' + NC + '/OUT_3D/'
    f_name_optics = 'Azores_2010-01-30_' + NC + '.cloud_properties_time_series.nc'  # cloud_optics_time_series.nc'
    variables_optics = ['REFF_QOPD_1_3_OPD', 'NC_QOPD_1_3_OPD', 'time', 'REFF_QOPD_2_0_to_30_REFF', 'LWP_0_to_30_REFF']
    traj_optics[NC] = LES_support.get_LES_traj(path_LES_optics, f_name_optics, variables_optics)

    path_LES_2D = '/media/tgoren/TOSHIBA EXT/noaa_home_directory/research/analysis/continental_tracks/LES/data/2018-06-13/Azores_2010-01-30_' + NC + '/OUT_2D/'
    f_name_2D = 'Azores_2010-01-30_' + NC + '.CWP.nc'
    variables_2D = ['CWP', 'time']
    traj_2D[NC] = LES_support.get_LES_traj(path_LES_2D, f_name_2D, variables_2D)
    traj_STATS[NC]['CWP_short'] = traj_2D[NC]['CWP'][np.arange(0,4680,5)]

    path_LES_2D = '/media/tgoren/TOSHIBA EXT/noaa_home_directory/research/analysis/continental_tracks/LES/data/2018-06-13/Azores_2010-01-30_' + NC + '/OUT_2D/'
    f_name_2D = 'Azores_2010-01-30_' + NC + '.time_series.nc'
    variables_2D = ['CF', 'time']
    traj_CF[NC] = LES_support.get_LES_traj(path_LES_2D, f_name_2D, variables_2D)
    traj_STATS[NC]['CF_short'] = traj_CF[NC]['CF'][np.arange(0,4680,5)]

# Plotting
size = np.shape(traj_STATS[NC]['CWP_short'])
LWP_mean = np.zeros(size[0])*np.nan
LWP_std = np.zeros(size[0])*np.nan
CWP_sig_mu = np.zeros(size[0])*np.nan
for i,t in enumerate(np.arange(size[0])):
    mask = [traj_STATS[NC]['CWP_short'][t]*10**3 > 5]
    LWP_mean[i] = np.nanmean(traj_STATS[NC]['CWP_short'][t][mask]*10**3)
    LWP_std[i] = np.std(traj_STATS[NC]['CWP_short'][t][mask] * 10 ** 3)

CWP_sig_mu =  LWP_std / LWP_mean

plt.figure(); plt.scatter(CWP_sig_mu, traj_STATS[NC]['ALBEDO'], c=traj_STATS[NC]['CF_short']); plt.colorbar()
plt.ylabel('albedo'); plt.xlabel('sig/mu of LWP')

plt.figure(); plt.scatter(CWP_sig_mu, traj_STATS[NC]['ALBEDO'], c=traj_optics[NC]['NC_QOPD_1_3_OPD']); plt.colorbar()
plt.ylabel('albedo'); plt.xlabel('sig/mu of LWP')


CWP_sig_mu =  LWP_mean/LWP_std
plt.figure(); plt.scatter(traj_STATS[NC]['CF_short'], (CWP_sig_mu)**2, c=np.arange(size[0])); plt.colorbar()
plt.xlabel('CF'); plt.ylabel('sig/mu of LWP')