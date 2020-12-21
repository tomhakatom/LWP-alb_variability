
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

## Plotting ##
plt.rc('ytick', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('axes', labelsize=20)
plt.rc('axes', titlesize=20)
plt.rc('font', size=20)

########################################################################################################################
## Global analysis - L2
########################################################################################################################
# Time counter - start
t = time.time()

all_data = {}
#file_list = np.sort(glob.glob('/media/tgoren/TOSHIBA EXT/NOAA/data/MODIS/MODIS_L2/SEP/January/*.*'))
file_list = np.sort(glob.glob('/media/tgoren/intenso02/tgoren/data/MODIS_noaa/MODIS_L2/SEP/*.*'))
modis_vars = ['Cloud_Effective_Radius', 'Cloud_Water_Path', 'Cloud_Optical_Thickness', \
              'cloud_top_temperature_1km', 'Cloud_Multi_Layer_Flag', \
              'Cloud_Water_Path_Uncertainty', 'Cloud_Effective_Radius_16', \
              'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Effective_Radius_Uncertainty', \
              'Atm_Corr_Refl', 'Cloud_Fraction', 'Latitude', 'Longitude', \
              'Cloud_Mask_1km', 'Retrieval_Failure_Metric','Cloud_Top_Height',\
              'Solar_Zenith', 'Cloud_Phase_Optical_Properties']

output_all = LWP_alb_variability_support.creat_output_dict()

for count, f_name_modis in enumerate(file_list):
    print(f_name_modis)
    modis_data_all = read_data.read_modis(f_name_modis, modis_vars) # read modis data
    modis_data = copy.copy(read_data.shorten_modis_swath_data(modis_data_all, 500, 500))
    output_all = LWP_alb_variability_support.grid_data(modis_data, f_name_modis, output_all)

# Time counter - end
elapsed = time.time() - t
print(elapsed/60)




output_all['LWP_mean'] = output_all['LWP']/output_all['Num']
output_all['LWP_std'] = ((output_all['LWP_squared']/output_all['Num']) - (output_all['LWP']/output_all['Num'])**2)**(0.5)
output_all['LWP_skewness'] = ((output_all['LWP_cube']/(output_all['Num']-1)) - ((output_all['LWP']/(output_all['Num']-1))**3))**(1/3) / (output_all['LWP_std']**3)
output_all['reflectance_mean'] = output_all['reflectance']/output_all['Num']
output_all['CF_mean'] = output_all['CF_MODIS']/output_all['Num_all']
output_all['CDNC_mean'] = output_all['CDNC']/output_all['Num']
output_all['reff_mean'] = output_all['reff']/output_all['Num']


CF_mask = [(output_all['CF_mean'] > 0.7) *
           (output_all['CF_mean'] <=1 )][0]
plt.figure(); plt.scatter(output_all['LWP_std'][CF_mask]/output_all['LWP_mean'][CF_mask], output_all['reflectance_mean'][CF_mask], c=output_all['reff_mean'][CF_mask], cmap='jet')
plt.colorbar()
plt.xlim([-5,5]); plt.ylim(0,0.8);


CF_mask = [(output_all['CF_mean'] > 0.9) *
           (output_all['CF_mean'] <=1 )][0]
plt.figure(); plt.scatter(output_all['LWP_std'][CF_mask], output_all['reflectance_mean'][CF_mask], c=output_all['CF_mean'][CF_mask], cmap='jet')
plt.colorbar()
plt.xlim([-5,5]); plt.ylim(0,0.8);







    cld_mask_copy = np.zeros(np.shape(modis_data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_copy[modis_data['Cloud_Mask_1km'][:,:,0]==57] = 1
    cld_mask_copy[modis_data['Cloud_Mask_1km'][:, :, 0] == 41] = 1

    bins = {}
    bins['lat'] = np.arange(-90, 91, 1)
    bins['lon'] = np.arange(-180, 181, 1)

    mask = [  # (data['Cloud_Phase_Optical_Properties'] == 2) *
        (modis_data['cloud_top_temperature_1km'] > 268) *
        (modis_data['Cloud_Multi_Layer_Flag'] < 2) *
        #(np.isfinite(data['CDNC'] + data['Cloud_Water_Path'])) *
        #(modis_data['CDNC'] < 700)*
        (modis_data['Cloud_Optical_Thickness'] > 3)*
        (cld_mask_copy == 1)][0]

    # Swath into 100x100 pixels
    data = {}
    i = 0
    for lat in np.arange(0, modis_data['Latitude_1km'].shape[0] - 100, 100):
        for lon in np.arange(0, modis_data['Longitude_1km'].shape[1] - 100, 100):
            # for lon in np.arange(, modis_data['Longitude_1km'].shape[1] - 100, 100):
            # lon = modis_data['Longitude_1km'].shape[1]//2
            # lon_ind_temp = np.where(np.logical_and(modis_data['Longitude_1km'][lat,:]>= lon,modis_data['Longitude_1km'][lat,:]<lon + 1))
            data[i] = [np.arange(lat, lat + 101), np.arange(lon, lon + 101)]
            i += 1



    #  KDtree
    modis_lat_lon = zip(modis_data['Latitude_1km'].ravel(), modis_data['Longitude_1km'].ravel())
    modis_latlon_kd = spatial.cKDTree(list(modis_lat_lon))
    # Finding the MODIS indices that match
    ind_mod_cal_colloc = modis_latlon_kd.query(list([-61,-104]))  # MODIS indices of the pixels that within the middle of the swath (this is needed because I cut the calipso data by the min and max of modis). The length is the same as the calipso input - one modix index per calipso index
    cal_swath_ind = np.where(ind_mod_cal_colloc[0][:] < 0.01)[0]  # MODIS indices that match CALIPSO (distance limit) - this reduces the size because now it is the most accurate collocation





    Lat_temp = np.histogram2d(modis_data['Longitude_1km'][mask], modis_data['Latitude_1km'][mask], \
                                   bins=[bins['lon'], bins['lat']], weights=modis_data['Latitude_1km'][mask])[0]

    Lon_temp = np.histogram2d(modis_data['Longitude_1km'][mask], modis_data['Latitude_1km'][mask], \
                                   bins=[bins['lon'], bins['lat']], weights=modis_data['Longitude_1km'][mask])[0]





for count, f_name_modis in enumerate(file_list):
    # modis_data = averaging_problem_support.read_modis(f_name_modis, path_MODIS) # read modis data
    modis_data = temp.read_modis(f_name_modis, path_MODIS)  # read modis data

    # Swath into 100x100 pixels
    data = {}
    i = 0
    for lat in np.arange(0, modis_data['Latitude_1km'].shape[0] - 100, 100):
        for lon in np.arange(0, modis_data['Longitude_1km'].shape[1] - 100, 100):
            # for lon in np.arange(, modis_data['Longitude_1km'].shape[1] - 100, 100):
            # lon = modis_data['Longitude_1km'].shape[1]//2
            # lon_ind_temp = np.where(np.logical_and(modis_data['Longitude_1km'][lat,:]>= lon,modis_data['Longitude_1km'][lat,:]<lon + 1))
            data[i] = [np.arange(lat, lat + 101), np.arange(lon, lon + 101)]
            i += 1


def sub_grid_analysis(data, modis_data, factor):
    # reduce resolution and calculate albedo bias
    dims_data = len(data.keys())
    data_scene = {}
    product = {'LWP': np.zeros(dims_data) * np.nan,
               'albedo_subgrid_mean': np.zeros(dims_data) * np.nan}
    modis_data['tau_CF'] = np.zeros(dims_data) * np.nan
    modis_data['LWP'] = np.zeros(dims_data) * np.nan
    modis_data['reflectance'] = np.zeros(dims_data) * np.nan
    for ff, f in enumerate(range(dims_data)):
        ## Masking
        # Create tau CF
        tau_temp = modis_data['Cloud_Optical_Thickness'][data[ff][0][0]:data[ff][0][-1],
                   data[ff][1][0]:data[ff][1][-1]]
        modis_data['tau_CF'][ff] = sum(sum(~np.isnan(tau_temp))) / np.size(tau_temp)
        LWP_temp_box = modis_data['Cloud_Optical_Thickness'][data[ff][0][0]:data[ff][0][-1],
                   data[ff][1][0]:data[ff][1][-1]]
        modis_data['LWP'][ff] = scipy.stats.skew(np.ravel(LWP_temp_box[~np.isnan(LWP_temp_box)]))
        reflectance_temp_box = modis_data['Atm_Corr_Refl'][:,:,0][data[ff][0][0]:data[ff][0][-1],
                   data[ff][1][0]:data[ff][1][-1]]
        modis_data['reflectance'][ff] = np.nanmean(reflectance_temp_box)




    ## reduce resolution and calculate albedo bias
    factor = 20
    data_scene = averaging_problem_support.sub_grid_analysis(data, modis_data, factor)
    # Get all cases into one dictionary
    all_data[f_name_modis] = data_scene