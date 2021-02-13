
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
import imp

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
#file_list = np.sort(glob.glob('/media/tgoren/intenso02/tgoren/data/MODIS_noaa/MODIS_L2/SEP/*.*'))
file_list = np.sort(glob.glob('/media/tgoren/TOSHIBA EXT/NOAA/data/MODIS/MODIS_L2/SEP/January/*.*'))

modis_vars = ['Cloud_Effective_Radius', 'Cloud_Water_Path', 'Cloud_Optical_Thickness', \
              'cloud_top_temperature_1km', 'Cloud_Multi_Layer_Flag', \
              'Cloud_Water_Path_Uncertainty', 'Cloud_Effective_Radius_16', \
              'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Effective_Radius_Uncertainty', \
              'Atm_Corr_Refl', 'Cloud_Fraction', 'Latitude', 'Longitude', \
              'Cloud_Mask_1km', 'Retrieval_Failure_Metric','Cloud_Top_Height',\
              'Solar_Zenith', 'Cloud_Phase_Optical_Properties', 'Retrieval_Failure_Metric']

for count, f_name_modis in enumerate(file_list):
    print(f_name_modis)
    modis_data_all = read_data.read_modis(f_name_modis, modis_vars) # read modis data
    modis_data_all['CDNC'] = read_data.calculate_Nd_adjust(modis_data_all['Cloud_Effective_Radius'], modis_data_all['Cloud_Optical_Thickness'])
    # Including the excluded retrievals of re and tau
    modis_data_all['Cloud_Effective_Radius_all'] = copy.copy(modis_data_all['Cloud_Effective_Radius'])
    modis_data_all['Cloud_Optical_Thickness_all'] = copy.copy(modis_data_all['Cloud_Optical_Thickness'])
    modis_data_all['Cloud_Water_Path_all'] = copy.copy(modis_data_all['Cloud_Water_Path'])
    modis_data_all['Cloud_Effective_Radius_all'][np.isnan(modis_data_all['Cloud_Effective_Radius'])] = modis_data_all['Retrieval_Failure_Metric'][:,:,1][np.isnan(modis_data_all['Cloud_Effective_Radius'])]
    modis_data_all['Cloud_Optical_Thickness_all'][np.isnan(modis_data_all['Cloud_Optical_Thickness'])] = modis_data_all['Retrieval_Failure_Metric'][:,:,0][np.isnan(modis_data_all['Cloud_Optical_Thickness'])]
    modis_data_all['Cloud_Water_Path_all'] = 2/3 * (10**6 * modis_data_all['Cloud_Effective_Radius_all'] * 10**(-6) * modis_data_all['Cloud_Optical_Thickness_all'])
    modis_data = copy.copy(read_data.shorten_modis_swath_data(modis_data_all, 500, 500))
    modis_data['f_name'] = f_name_modis

    # Swath into 100x100 pixels
    data = {}
    i=0
    for lat in np.arange(0,modis_data['Latitude_1km'].shape[0]-100, 100):
        for lon in np.arange(0,modis_data['Longitude_1km'].shape[1]-100, 100):
            data[i] = [np.arange(lat,lat+101), np.arange(lon,lon+101)]
            i+=1

    data_scene = LWP_alb_variability_support.regridding_for_loop(data, modis_data)
    # Get all cases into one dictionary
    all_data[count] = data_scene

## Sum all cases into one array
processed_data = {'LWP_mean': [], 'LWP_std':[], 'LWP_median':[], 'LWP_skewness':[],'LWP_kurtosis':[],\
                  'LWP_all_mean': [], 'LWP_all_std':[],  'LWP_all_skewness':[],\
                  'LWP_mean_zeros': [], 'LWP_std_zeros':[],\
                  'tau_skewness': [], 'tau_kurtosis':[], 'CDNC_mean':[], 'CDNC_std':[],\
                  'CDNC_median':[], 're_mean':[],'CF_mean':[], 'FFT_1D_max':[],\
                  'CF_MODIS_mean':[], 'reflectance_mean':[], 'reflectance_std':[], 'reflectance_skewness':[],\
                  'albedo_mean':[], 'f_name':[],\
                  'lat_center':[], 'lon_center':[]}

for k in all_data.keys():
    processed_data['LWP_mean'] = np.append(processed_data['LWP_mean'], all_data[k]['LWP_mean'])
    processed_data['LWP_std'] = np.append(processed_data['LWP_std'], all_data[k]['LWP_std'])
    processed_data['LWP_median'] = np.append(processed_data['LWP_median'], all_data[k]['LWP_median'])
    processed_data['LWP_skewness'] = np.append(processed_data['LWP_skewness'], all_data[k]['LWP_skewness'])
    processed_data['LWP_kurtosis'] = np.append(processed_data['LWP_kurtosis'], all_data[k]['LWP_kurtosis'])
    processed_data['LWP_mean_zeros'] = np.append(processed_data['LWP_mean_zeros'], all_data[k]['LWP_mean_zeros'])
    processed_data['LWP_std_zeros'] = np.append(processed_data['LWP_std_zeros'], all_data[k]['LWP_std_zeros'])
    processed_data['LWP_all_mean'] = np.append(processed_data['LWP_all_mean'], all_data[k]['LWP_all_mean'])
    processed_data['LWP_all_std'] = np.append(processed_data['LWP_all_std'], all_data[k]['LWP_all_std'])
    processed_data['LWP_all_skewness'] = np.append(processed_data['LWP_all_skewness'], all_data[k]['LWP_all_skewness'])
    processed_data['tau_skewness'] = np.append(processed_data['tau_skewness'], all_data[k]['tau_skewness'])
    processed_data['tau_kurtosis'] = np.append(processed_data['tau_kurtosis'], all_data[k]['tau_kurtosis'])
    processed_data['CDNC_mean'] = np.append(processed_data['CDNC_mean'], all_data[k]['CDNC_mean'])
    processed_data['CDNC_std'] = np.append(processed_data['CDNC_std'], all_data[k]['CDNC_std'])
    processed_data['CDNC_median'] = np.append(processed_data['CDNC_median'], all_data[k]['CDNC_median'])
    processed_data['re_mean'] = np.append(processed_data['re_mean'], all_data[k]['re_mean'])
    processed_data['CF_mean'] = np.append(processed_data['CF_mean'], all_data[k]['CF_mean'])
    processed_data['CF_MODIS_mean'] = np.append(processed_data['CF_MODIS_mean'], all_data[k]['CF_MODIS_mean'])
    processed_data['FFT_1D_max'] = np.append(processed_data['FFT_1D_max'], all_data[k]['FFT_1D_max'])
    processed_data['reflectance_mean'] = np.append(processed_data['reflectance_mean'], all_data[k]['reflectance_mean'])
    processed_data['reflectance_std'] = np.append(processed_data['reflectance_std'], all_data[k]['reflectance_std'])
    processed_data['reflectance_skewness'] = np.append(processed_data['reflectance_skewness'], all_data[k]['reflectance_skewness'])
    processed_data['albedo_mean'] = np.append(processed_data['albedo_mean'], all_data[k]['albedo_mean'])
    processed_data['f_name'] = np.append(processed_data['f_name'], all_data[k]['f_name'])
    processed_data['lat_center'] = np.append(processed_data['lat_center'], all_data[k]['lat_center'])
    processed_data['lon_center'] = np.append(processed_data['lon_center'], all_data[k]['lon_center'])


# Plotting
# Albedo approximation vs reflectance
CF_mask = [(processed_data['CF_MODIS_mean'] > 0) *
           (processed_data['CF_MODIS_mean'] <=1)][0]
plt.figure(); plt.scatter(processed_data['albedo_mean'][CF_mask], processed_data['reflectance_mean'][CF_mask], c=processed_data['CF_MODIS_mean'][CF_mask],cmap='jet', vmax=1)
plt.colorbar()
plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), 'k')
plt.xlabel('albedo approximation'); plt.ylabel('cloud reflectance'); plt.title('Albedo approximation vs cloud reflectance')
plt.xlim(0,1); plt.ylim(0,1)

# Reflectance vs CF (in color LWP std)
CF_mask = [(processed_data['CF_MODIS_mean'] > 0.2) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], processed_data['reflectance_mean'][CF_mask], c=processed_data['LWP_std'][CF_mask], cmap='jet', vmax=100)
plt.colorbar()

CF_mask = [(processed_data['CF_MODIS_mean'] > 0.2) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], processed_data['reflectance_mean'][CF_mask]*processed_data['CF_MODIS_mean'][CF_mask],\
                          c=processed_data['LWP_std'][CF_mask], cmap='jet', vmax=100)
plt.colorbar()

# Reflectance vs CF (in color LWP mean)
CF_mask = [(processed_data['CF_MODIS_mean'] > 0.2) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], processed_data['reflectance_mean'][CF_mask]*processed_data['CF_MODIS_mean'][CF_mask],\
                          c=processed_data['LWP_mean'][CF_mask], cmap='jet', vmax=200)
plt.colorbar()


CF_mask = [(processed_data['CF_MODIS_mean'] > 0) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['LWP_kurtosis'][CF_mask], (processed_data['LWP_mean'][CF_mask]/processed_data['LWP_std'][CF_mask])**2, c=processed_data['re_mean'][CF_mask], cmap='jet', vmax=25)
plt.colorbar()
plt.xlim([-5,5]); plt.ylim(0,0.8);

CF_mask = [(processed_data['CF_MODIS_mean'] > 0.2) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], (processed_data['LWP_mean'][CF_mask]/processed_data['LWP_std'][CF_mask])**2, c=processed_data['re_mean'][CF_mask], cmap='jet', vmax=25)
plt.colorbar()
plt.xlim([-5,5]); plt.ylim(0,0.8);

for i in np.arange(0,1,0.1):
    CF_mask = [(processed_data['CF_MODIS_mean'] > i) *
               (processed_data['CF_MODIS_mean'] <= i + 0.1)][0]
    print(np.mean((processed_data['LWP_mean'][CF_mask]/processed_data['LWP_std'][CF_mask])**2))

CF_mask = [(processed_data['CF_mean'] > 0.8) *
           (processed_data['CF_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['LWP_skewness'][CF_mask], processed_data['reflectance'][CF_mask], c=processed_data['LWP_mean'][CF_mask], cmap='jet', vmax=200)
plt.colorbar()
plt.xlim([-5,5]); plt.ylim(0,0.8);



## Mask by LWP
fig, axs = plt.subplots(2, 3)
axs_flatten = axs.flatten()
#for cf in np.arange(0.6,1,0.1):
for i, lwp in enumerate(np.arange(75,225,25)):
    LWP_mask = [#(processed_data['LWP_mean'] > 100) *
               (processed_data['LWP_mean'] > lwp) *
               (processed_data['LWP_mean'] <=lwp+50)*
               (processed_data['CF_MODIS_mean'] > 0.8) *
               (processed_data['CF_MODIS_mean'] <=0.9)][0]
    im = axs_flatten[i].scatter(processed_data['albedo_mean'][LWP_mask], processed_data['LWP_std'][LWP_mask]/processed_data['LWP_mean'][LWP_mask],\
                              c=processed_data['LWP_median'][LWP_mask], cmap='jet', vmin=0, vmax=200)
    axs_flatten[i].set_title(str(lwp) +'<LWP<' + str(lwp+50))
    axs_flatten[i].set_xlim([0.2,0.7]); axs_flatten[i].set_ylim([0,2])
    regression = scipy.stats.linregress(processed_data['albedo_mean'][LWP_mask],
                                        processed_data['LWP_std'][LWP_mask] / processed_data['LWP_mean'][LWP_mask])
    print(regression)
    axs_flatten[i].plot(np.arange(0.2,0.9,0.1), np.arange(0.2,0.9,0.1)*regression[0] + regression[1], color='k')
fig.add_subplot(111, frameon=False); plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.colorbar(im, ax=axs.ravel().tolist())
#plt.xlabel("common X")
fig.text(0.45, 0.04, 'Mean cloud albedo', ha='center')
fig.text(0.06, 0.5, 'sig/mu of LWP', ha='center',rotation=90)
#fig.suptitle('LWP>'+ str(lwp) + 'LWP<'+str(lwp+50))
fig.suptitle('Mean cloud albedo vs sig/mu of LWP; in color: median LWP\n CF>0.9')


cf = 0.9
## Mask by LWP - sig/mu of albedo
fig, axs = plt.subplots(2, 3)
axs_flatten = axs.flatten()
#for i, cf in enumerate(np.arange(0.4,1,0.1)):
for i, lwp in enumerate(np.arange(0,150,25)):
    LWP_mask = [#(processed_data['LWP_mean'] > 100) *
               (processed_data['LWP_mean'] > lwp) *
               (processed_data['LWP_mean'] <=lwp+50)*
               (processed_data['CF_MODIS_mean'] > cf) *
               (processed_data['CF_MODIS_mean'] <=cf+0.1)][0]
    im = axs_flatten[i].scatter(processed_data['reflectance_mean'][LWP_mask], processed_data['reflectance_std'][LWP_mask]/processed_data['reflectance_mean'][LWP_mask],\
                              c=processed_data['CDNC_mean'][LWP_mask], cmap='jet', vmin=0, vmax=200)
    axs_flatten[i].set_title(str(lwp) +'<LWP<' + str(lwp+50))
    axs_flatten[i].set_xlim([0,0.7]); axs_flatten[i].set_ylim([0,1])
    regression = scipy.stats.linregress(processed_data['reflectance_mean'][LWP_mask],
                                        processed_data['reflectance_std'][LWP_mask]/processed_data['reflectance_mean'][LWP_mask])
    print(regression)
    axs_flatten[i].plot(np.arange(0,0.9,0.1), np.arange(0,0.9,0.1)*regression[0] + regression[1], color='k')
fig.add_subplot(111, frameon=False); plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.colorbar(im, ax=axs.ravel().tolist())
#plt.xlabel("common X")
fig.text(0.45, 0.04, 'Mean cloud reflectance', ha='center')
fig.text(0.06, 0.5, 'sig/mu of reflectance', ha='center',rotation=90)
#fig.suptitle('LWP>'+ str(lwp) + 'LWP<'+str(lwp+50))
fig.suptitle('Mean cloud reflectance vs sig/mu of reflectance; in color: mean CDNC\n' + str(cf) + ' >CF>' + str(cf+0.1))




cf = 0.8
## Mask by LWP - sig/mu of albedo
fig, axs = plt.subplots(2, 3)
axs_flatten = axs.flatten()
#for i, cf in enumerate(np.arange(0.4,1,0.1)):
for i, lwp in enumerate(np.arange(50,200,25)):
    LWP_mask = [#(processed_data['LWP_mean'] > 100) *
               (processed_data['LWP_mean'] > lwp) *
               (processed_data['LWP_mean'] <=lwp+50)*
               (processed_data['CF_MODIS_mean'] > cf) *
               (processed_data['CF_MODIS_mean'] <=cf+0.1)][0]
    im = axs_flatten[i].scatter(processed_data['albedo_mean'][LWP_mask], processed_data['LWP_std'][LWP_mask]/processed_data['LWP_mean'][LWP_mask],\
                              c=processed_data['CDNC_mean'][LWP_mask], cmap='jet', vmin=0, vmax=200)
    axs_flatten[i].set_title(str(lwp) +'<LWP<' + str(lwp+50))
    axs_flatten[i].set_xlim([0,1]); axs_flatten[i].set_ylim([0,2])
    regression = scipy.stats.linregress(processed_data['albedo_mean'][LWP_mask],
                                        processed_data['LWP_std'][LWP_mask]/processed_data['LWP_mean'][LWP_mask])
    print(regression)
    axs_flatten[i].plot(np.arange(0,1,0.1), np.arange(0,1,0.1)*regression[0] + regression[1], color='k')
fig.add_subplot(111, frameon=False); plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.colorbar(im, ax=axs.ravel().tolist())
#plt.xlabel("common X")
fig.text(0.45, 0.04, 'Mean cloud albedo', ha='center')
fig.text(0.06, 0.5, 'sig/mu of LWP', ha='center',rotation=90)
#fig.suptitle('LWP>'+ str(lwp) + 'LWP<'+str(lwp+50))
fig.suptitle('Mean cloud **albedo** vs sig/mu of LWP; in color: mean CDNC\n' + str(cf) + ' >CF>' + str(cf+0.1))


cf = 0.9
## Mask by LWP - sig/mu of albedo
fig, axs = plt.subplots(2, 3)
axs_flatten = axs.flatten()
#for i, cf in enumerate(np.arange(0.4,1,0.1)):
for i, lwp in enumerate(np.arange(0,150,25)):
    LWP_mask = [#(processed_data['LWP_mean'] > 100) *
               (processed_data['LWP_mean'] > lwp) *
               (processed_data['LWP_mean'] <=lwp+50)*
               (processed_data['CF_MODIS_mean'] > cf) *
               (processed_data['CF_MODIS_mean'] <=cf+0.1)][0]
    im = axs_flatten[i].hist(processed_data['reflectance_mean'][LWP_mask], 20)
    axs_flatten[i].set_title(str(lwp) +'<LWP<' + str(lwp+50))
    #axs_flatten[i].set_xlim([0,0.7]); axs_flatten[i].set_ylim([0,1])
    regression = scipy.stats.linregress(processed_data['reflectance_mean'][LWP_mask],
                                        processed_data['reflectance_std'][LWP_mask]/processed_data['reflectance_mean'][LWP_mask])
    print(regression)
    #axs_flatten[i].plot(np.arange(0,0.9,0.1), np.arange(0,0.9,0.1)*regression[0] + regression[1], color='k')
fig.add_subplot(111, frameon=False); plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.colorbar(im, ax=axs.ravel().tolist())
#plt.xlabel("common X")
fig.text(0.45, 0.04, 'Mean cloud reflectance', ha='center')
fig.text(0.06, 0.5, 'sig/mu of reflectance', ha='center',rotation=90)
#fig.suptitle('LWP>'+ str(lwp) + 'LWP<'+str(lwp+50))
fig.suptitle('Mean cloud reflectance vs sig/mu of reflectance; in color: mean CDNC\n' + str(cf) + ' >CF>' + str(cf+0.1))







# replicaton Wood and Hartmann 2006 Fig 7
CF_mask = [(processed_data['LWP_std'] > 0) *
           (processed_data['CF_MODIS_mean'] > 0) *
           (processed_data['CF_MODIS_mean'] <=1)][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], (processed_data['LWP_mean'][CF_mask]/processed_data['LWP_std'][CF_mask])**2,\
                          c=processed_data['reflectance_mean'][CF_mask], cmap='jet', vmax=1)
##




CF_mask = [(processed_data['CF_MODIS_mean'] > 0) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['reflectance_mean'][CF_mask],processed_data['LWP_kurtosis'][CF_mask],c=processed_data['CF_MODIS_mean'][CF_mask], cmap='jet', vmax=1)
plt.xlim(0,1);
plt.colorbar()





CF_mask = [#(processed_data['LWP_mean'] > 100) *
           (processed_data['CF_MODIS_mean'] > 0.9) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['reflectance_mean'][CF_mask], processed_data['LWP_std'][CF_mask]/processed_data['LWP_mean'][CF_mask],\
                          c=processed_data['LWP_mean'][CF_mask], cmap='jet', vmax=200)
plt.colorbar()




LWP_mask = [#(processed_data['LWP_mean'] > 100) *
           (processed_data['LWP_mean'] > 150) *
           (processed_data['LWP_mean'] <= 200)*
           (processed_data['CF_MODIS_mean'] > 0.9) *
           (processed_data['CF_MODIS_mean'] <=1)][0]
plt.figure(); plt.scatter(processed_data['reflectance_mean'][LWP_mask], processed_data['LWP_std'][LWP_mask]/processed_data['LWP_mean'][LWP_mask],\
                          c=processed_data['CF_MODIS_mean'][LWP_mask], cmap='jet', vmax=1)
plt.colorbar()
regression = scipy.stats.linregress(processed_data['albedo_mean'][LWP_mask], processed_data['LWP_std'][LWP_mask]/processed_data['LWP_mean'][LWP_mask])


CF_mask = [(processed_data['CF_MODIS_mean'] > 0.8) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['reflectance_mean'][CF_mask],processed_data['FFT_1D_max'][CF_mask],c=processed_data['LWP_mean'][CF_mask], cmap='jet', vmax=200)
plt.colorbar()

CF_mask = [(processed_data['CF_MODIS_mean'] > 0.2) *
           (processed_data['CF_MODIS_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['reflectance_mean'][CF_mask],processed_data['reflectance_skewness'][CF_mask],c=processed_data['CDNC_mean'][CF_mask], cmap='jet', vmax=200)
plt.colorbar()

CF_mask = [(processed_data['CF_MODIS_mean'] > 0.3) *
           (processed_data['CF_MODIS_mean'] <=1)][0]
plt.figure(); plt.scatter(processed_data['LWP_skewness'][CF_mask],processed_data['reflectance_mean'][CF_mask]*processed_data['CF_MODIS_mean'][CF_mask]\
                          ,c=processed_data['CF_MODIS_mean'][CF_mask], cmap='jet', vmax=1)
plt.colorbar()

CF_mask = [(processed_data['CF_MODIS_mean'] > 0.9) *
           (processed_data['CF_MODIS_mean'] <=1)][0]
plt.figure(); plt.scatter((processed_data['LWP_std'][CF_mask]),processed_data['reflectance_mean'][CF_mask]\
                          ,c=processed_data['CF_MODIS_mean'][CF_mask], cmap='jet', vmax=1)
plt.colorbar()




H, xedges, yedges = np.histogram2d(processed_data['LWP_skewness'][CF_mask],processed_data['reflectance_mean'][CF_mask], bins=(np.arange(0,5,0.1), np.arange(0,1,0.1)))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(H/sum(H), interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmax=0.2)
ax.set_aspect('auto')
plt.xlabel('delta LWP'); plt.ylabel('reflectance')


CF_mask = [(processed_data['CF_mean'] > 0) *
           (processed_data['CF_mean'] <=1 )][0]
plt.figure(); plt.scatter(processed_data['CF_MODIS_mean'][CF_mask], processed_data['reflectance'][CF_mask], c=processed_data['LWP_std'][CF_mask]/processed_data['LWP_mean'][CF_mask], cmap='jet', vmax=2)
plt.colorbar()





#### Joint Histogram approach

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