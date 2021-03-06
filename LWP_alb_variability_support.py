import numpy as np
import read_data
import copy
import scipy
from scipy import fftpack
from scipy import signal
from csat import MODIS
import datetime
import misc_codes

bins = {}
bins['lat'] = np.arange(-90, 91, 1)
bins['lon'] = np.arange(-180, 181, 1)

def grid_data(data, f_name_modis, output_all):
    # MODIS data
    data['CDNC'] = read_data.calculate_Nd_adjust(data['Cloud_Effective_Radius'], data['Cloud_Optical_Thickness'])
    data['CDNC'][data['Cloud_Optical_Thickness'] < 3] = np.nan
    data['CDNC'][data['Cloud_Effective_Radius'] < 3] = np.nan

    cld_mask_copy = np.zeros(np.shape(data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_copy[data['Cloud_Mask_1km'][:, :, 0] == 57] = 1
    cld_mask_copy[data['Cloud_Mask_1km'][:, :, 0] == 41] = 1

    mask = [  # (data['Cloud_Phase_Optical_Properties'] == 2) *
        (data['cloud_top_temperature_1km'] > 270) *
        (data['Cloud_Multi_Layer_Flag'] < 2) *
        # (np.isfinite(data['CDNC'] + data['Cloud_Water_Path'])) *
        (data['CDNC'] < 300) *
        (data['Cloud_Optical_Thickness'] > 3) *
        (data['Cloud_Effective_Radius'] > 3) *
        (cld_mask_copy == 1)][0]

    Num_temp_all = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                  bins=[bins['lon'], bins['lat']])[0].astype('int')

    CF_MODIS_temp = np.histogram2d(np.ravel(data['Longitude_1km'][mask]), np.ravel(data['Latitude_1km'][mask]), \
                                   bins=[bins['lon'], bins['lat']], weights=np.ravel(data['Cloud_Fraction_1km'][mask]))[
        0]

    Num_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']])[0].astype('int')

    CDNC_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                               bins=[bins['lon'], bins['lat']], weights=data['CDNC'][mask])[0]

    LWP_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=data['Cloud_Water_Path'][mask])[0]

    LWP_squared_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=(data['Cloud_Water_Path'][mask])**2)[0]

    LWP_cube_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=(data['Cloud_Water_Path'][mask])**3)[0]

    reff_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                               bins=[bins['lon'], bins['lat']], weights=data['Cloud_Effective_Radius'][mask])[0]

    tau_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=data['Cloud_Optical_Thickness'][mask])[0]

    # data['Atm_Corr_Refl'][:, :, 0][np.isnan(data['Atm_Corr_Refl'][:, :, 0])] = 0
    reflectance_temp = np.histogram2d(np.ravel(data['Longitude_1km'][mask]), np.ravel(data['Latitude_1km'][mask]), \
                                      bins=[bins['lon'], bins['lat']], \
                                      weights=np.ravel(data['Atm_Corr_Refl'] \
                                                           [:, :, 0][mask]) * (
                                                          np.ravel(data['Cloud_Fraction_1km'][mask]) + \
                                                          0.07 * (1 - (np.ravel(data['Cloud_Fraction_1km'][mask])))))[0]

    Lat_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=data['Latitude_1km'][mask])[0]

    Lon_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=data['Longitude_1km'][mask])[0]

    Multi_temp = copy.copy(data['Cloud_Multi_Layer_Flag'])
    Multi_temp[np.isnan(Multi_temp)] = 1  # NaN is no retrieval
    Multi_temp[Multi_temp > 1] = 0
    Multi_layer_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                      bins=[bins['lon'], bins['lat']], weights=np.ravel(Multi_temp))[0]

    Cld_phase_temp = copy.copy(data['Cloud_Phase_Optical_Properties'])
    Cld_phase_temp[Cld_phase_temp <= 2] = 1
    Cld_phase_temp[Cld_phase_temp > 2] = 0
    Cld_phase_temp[np.isnan(Cld_phase_temp)] = 1
    Cld_phase_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                    bins=[bins['lon'], bins['lat']], weights=np.ravel(Cld_phase_temp))[0]

    cld_mask_temp = np.ones(np.shape(data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_temp[data['Cloud_Mask_1km'][:, :, 0] <= 0] = 0
    cld_mask_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                   bins=[bins['lon'], bins['lat']], weights=np.ravel(cld_mask_temp))[0]

    # Masks
    Masks = [  # (LTS > 18.55)*
        ((Multi_layer_flag / Num_temp_all) > 0.95) *
        ((Cld_phase_flag / Num_temp_all) > 0.95) *
        ((CF_MODIS_temp / Num_temp_all) > 0.2) *
        (LWP_temp / Num_temp_all >= 20) *
        ((cld_mask_flag / Num_temp_all) > 0.95)][0]

    CDNC_temp[~Masks] = 0
    CDNC_temp[np.isnan(CDNC_temp)] = 0

    LWP_temp[~Masks] = 0
    LWP_temp[np.isnan(LWP_temp)] = 0

    reff_temp[~Masks] = 0
    reff_temp[np.isnan(reff_temp)] = 0

    tau_temp[~Masks] = 0
    tau_temp[np.isnan(tau_temp)] = 0

    CF_MODIS_temp[~Masks] = 0
    CF_MODIS_temp[np.isnan(CF_MODIS_temp)] = 0

    LWP_squared_temp[~Masks] = 0
    LWP_squared_temp[np.isnan(LWP_squared_temp)] = 0

    LWP_cube_temp[~Masks] = 0
    LWP_cube_temp[np.isnan(LWP_cube_temp)] = 0

    reflectance_temp[~Masks] = 0
    reflectance_temp[np.isnan(reflectance_temp)] = 0

    Num_temp[~Masks] = 0
    Num_temp[np.isnan(Num_temp)] = 0
    Num_temp_all[~Masks] = 0
    # Num_temp_all[cld_mask_mask] = 0
    Num_temp_all[np.isnan(Num_temp_all)] = 0

    CF_temp = Num_temp
    CF_temp[~Masks] = 0
    CF_temp[np.isnan(CF_temp)] = 0

    # Data into dictionary
    output_all['Num'] = np.append(output_all['Num'], Num_temp[Num_temp > 0])
    output_all['Num_all'] = np.append(output_all['Num_all'], Num_temp_all[Num_temp > 0])
    output_all['CDNC'] = np.append(output_all['CDNC'], CDNC_temp[Num_temp > 0])
    output_all['LWP'] = np.append(output_all['LWP'], LWP_temp[Num_temp > 0])
    output_all['LWP_squared'] = np.append(output_all['LWP_squared'], LWP_squared_temp[Num_temp > 0])
    output_all['LWP_cube'] = np.append(output_all['LWP_cube'], LWP_squared_temp[Num_temp > 0])
    output_all['reff'] = np.append(output_all['reff'], reff_temp[Num_temp > 0])
    output_all['tau'] = np.append(output_all['tau'], tau_temp[Num_temp > 0])
    output_all['CF'] = np.append(output_all['CF'], Num_temp[Num_temp > 0])
    output_all['CF_MODIS'] = np.append(output_all['CF_MODIS'], CF_MODIS_temp[Num_temp > 0])
    output_all['reflectance'] = np.append(output_all['reflectance'], reflectance_temp[Num_temp > 0])
    output_all['Lon'] = np.append(output_all['Lon'], Lon_temp[Num_temp > 0])
    output_all['Lat'] = np.append(output_all['Lat'], Lat_temp[Num_temp > 0])

    print('Done')

    return output_all

def creat_output_dict():
    lat = 360
    lon = 180
    output={}
    output['count_num'] = np.zeros((lat, lon))
    output['Num'] = np.zeros((lat, lon))
    output['Num_all'] = np.zeros((lat, lon))
    output['LWP'] = np.zeros((lat, lon))
    output['LWP_squared'] = np.zeros((lat, lon))
    output['LWP_cube'] = np.zeros((lat, lon))
    output['reff'] = np.zeros((lat, lon))
    output['tau'] = np.zeros((lat, lon))
    output['CF'] = np.zeros((lat, lon))
    output['CF_MODIS'] = np.zeros((lat, lon))
    output['Multi_layer_flag'] = np.zeros((lat, lon))
    output['reflectance'] = np.zeros((lat, lon))
    output['CDNC'] = np.zeros((lat, lon))
    output['Num_all_pixels'] = np.zeros((lat, lon))
    output['Num_tau_pixels'] = np.zeros((lat, lon))
    output['Lon'] = np.zeros((lat, lon))
    output['Lat'] = np.zeros((lat, lon))
    return output



def regridding_for_loop(data, modis_data):
    # Regridding data into 1x1 degree using a "for loop"
    dims_data = len(data.keys())
    product = {'LWP': np.zeros(dims_data)*np.nan, 'reff':np.zeros(dims_data)*np.nan,\
               'LWP_mean':np.zeros(dims_data)*np.nan,'LWP_std':np.zeros(dims_data)*np.nan, \
               'LWP_median': np.zeros(dims_data)*np.nan, 'LWP_skewness':np.zeros(dims_data)*np.nan, \
               'LWP_kurtosis': np.zeros(dims_data) * np.nan, \
               'LWP_all_mean': np.zeros(dims_data) * np.nan, 'LWP_all_std': np.zeros(dims_data) * np.nan, \
               'LWP_all_skewness': np.zeros(dims_data) * np.nan, \
               'LWP_mean_zeros': np.zeros(dims_data) * np.nan, 'LWP_std_zeros': np.zeros(dims_data) * np.nan, \
               'tau_skewness': np.zeros(dims_data) * np.nan, 'tau_kurtosis':np.zeros(dims_data)*np.nan,\
               'CDNC_mean':np.zeros(dims_data)*np.nan, 'CDNC_std':np.zeros(dims_data)*np.nan, 'CDNC_median':np.zeros(dims_data)*np.nan, \
               're_mean': np.zeros(dims_data) * np.nan, \
               'CF_mean': np.zeros(dims_data) * np.nan, 'CF_MODIS_mean': np.zeros(dims_data) * np.nan,'FFT_1D_max': np.zeros(dims_data) * np.nan,\
               'CF':np.zeros(dims_data)*np.nan,\
               'reflectance_mean':np.zeros(dims_data)*np.nan, 'reflectance_std':np.zeros(dims_data)*np.nan, 'reflectance_skewness':np.zeros(dims_data)*np.nan,\
               'albedo_mean':np.zeros(dims_data)*np.nan,\
               'f_name': ['']*dims_data, 'lat_center':np.zeros(dims_data)*np.nan, 'lon_center':np.zeros(dims_data)*np.nan}

    modis_data['tau_CF'] = np.zeros(dims_data)*np.nan
    modis_data['Cloud_Water_Path_zeros'] = copy.copy(modis_data['Cloud_Water_Path'])
    modis_data['Cloud_Water_Path_zeros'][np.isnan(modis_data['Cloud_Water_Path_zeros'])] = 0
    modis_data['albedo'] = ((1-0.85) * modis_data['Cloud_Optical_Thickness']) / (2 + (1-0.85)* modis_data['Cloud_Optical_Thickness'])

    # Mask based on tau and re >3
    mask_tau = [(modis_data['Cloud_Optical_Thickness']<3)*
                (modis_data['Cloud_Effective_Radius']<3)][0]
    modis_data['Cloud_Water_Path'][mask_tau] = np.nan
    mask_tau_all = [(modis_data['Cloud_Optical_Thickness_all']<3)*
                (modis_data['Cloud_Effective_Radius_all']<3)][0]
    modis_data['Cloud_Water_Path_all'][mask_tau_all] = np.nan

    # Create masks
    Multi_temp = copy.copy(modis_data['Cloud_Multi_Layer_Flag'])
    Multi_temp[np.isnan(Multi_temp)] = 1  # NaN is no retrieval
    Multi_temp[Multi_temp > 1] = 0

    cld_mask_temp = np.zeros(np.shape(modis_data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_temp[modis_data['Cloud_Mask_1km'][:, :, 0] <= 0] = 1

    cld_temperature_temp = np.zeros(np.shape(modis_data['cloud_top_temperature_1km']))
    cld_temperature_temp[modis_data['cloud_top_temperature_1km'] <= 273] = 1

    #modis_data['MODIS_CF'] = np.zeros(dims_data) * np.nan
    for ff, f in enumerate(range(dims_data)):
        ## Masking
        # Create tau CF
        tau_temp = modis_data['Cloud_Optical_Thickness'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]
        modis_data['tau_CF'][ff] = sum(sum(~np.isnan(tau_temp)))/np.size(tau_temp)
        #modis_data['MODIS_CF'][ff] = np.mean(modis_data['Cloud_Fraction_1km'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])

        # Size of the box
        if not any(data[ff][0]) or not any(data[ff][1]):
            continue

        # Size of the box
        if data[ff][0].shape[0] < 75:
            continue

        # Create masks
        # Cloud multi layer
        if np.sum(np.sum(Multi_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]])) / \
            Multi_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]].size < 0.95: # Multi-layer threshold (1 is single layer, so I want at least95% to be single
            continue

        # Cloud mask threshold - land is out
        if np.sum(np.sum(cld_mask_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]])) / \
            cld_mask_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]].size > 0.05:
            continue

        # Cloud temperature mask
        if np.sum(np.sum(cld_temperature_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]])) / \
            cld_temperature_temp[data[ff][0][0]:data[ff][0][-1],data[ff][1][0]:data[ff][1][-1]].size > 0.05:
            continue

        ## Mean and std of other variables
        product['LWP_mean'][ff] = np.nanmean(modis_data['Cloud_Water_Path'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_std'][ff] = scipy.nanstd(modis_data['Cloud_Water_Path'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_median'][ff] = np.nanmedian(modis_data['Cloud_Water_Path'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_mean_zeros'][ff] = np.nanmean(modis_data['Cloud_Water_Path_zeros'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_std_zeros'][ff] = scipy.nanstd(modis_data['Cloud_Water_Path_zeros'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_skewness'][ff] = scipy.stats.skew(np.ravel(modis_data['Cloud_Water_Path'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['LWP_all_mean'][ff] = np.nanmean(modis_data['Cloud_Water_Path_all'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['LWP_all_skewness'][ff] = scipy.stats.skew(np.ravel(modis_data['Cloud_Water_Path_all'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['LWP_all_std'][ff] = scipy.nanstd(modis_data['Cloud_Water_Path_all'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['tau_skewness'][ff] = scipy.stats.skew(np.ravel(modis_data['Cloud_Optical_Thickness'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['LWP_kurtosis'][ff] = scipy.stats.kurtosis(np.ravel(modis_data['Cloud_Water_Path'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['tau_kurtosis'][ff] = scipy.stats.kurtosis(np.ravel(modis_data['Cloud_Optical_Thickness'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['CDNC_mean'][ff] = np.nanmean(modis_data['CDNC'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['CDNC_std'][ff] = scipy.nanstd(modis_data['CDNC'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['CDNC_median'][ff] = np.nanmedian(modis_data['CDNC'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['re_mean'][ff] = np.nanmean(modis_data['Cloud_Effective_Radius'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['CF_mean'][ff] = np.nanmean(modis_data['tau_CF'][ff]) #np.nanmean(modis_data['tau_CF'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['CF_MODIS_mean'][ff] = np.nanmean(modis_data['Cloud_Fraction_1km'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['FFT_1D_max'][ff] = power_spectrum_line(modis_data['Cloud_Water_Path_zeros'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['reflectance_mean'][ff] = np.nanmean(modis_data['Atm_Corr_Refl'][:, :, 0][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['reflectance_std'][ff] = scipy.nanstd(modis_data['Atm_Corr_Refl'][:, :, 0][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['albedo_mean'][ff] = np.nanmean(modis_data['albedo'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['reflectance_skewness'][ff] = scipy.stats.skew(np.ravel(modis_data['Atm_Corr_Refl'][:, :, 0][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]]), nan_policy='omit')
        product['lat_center'][ff] = np.nanmean(modis_data['Latitude_1km'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])
        product['lon_center'][ff] = np.nanmean(modis_data['Longitude_1km'][data[ff][0][0]:data[ff][0][-1], data[ff][1][0]:data[ff][1][-1]])

    data_scene = {}
    data_scene['LWP_mean'] = product['LWP_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_std'] = product['LWP_std'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_median'] = product['LWP_median'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_mean_zeros'] = product['LWP_mean_zeros'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_std_zeros'] = product['LWP_std_zeros'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_skewness'] = product['LWP_skewness'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_kurtosis'] = product['LWP_kurtosis'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_all_mean'] = product['LWP_all_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_all_std'] = product['LWP_all_std'][~np.isnan(product['LWP_mean'])]
    data_scene['LWP_all_skewness'] = product['LWP_all_skewness'][~np.isnan(product['LWP_mean'])]
    data_scene['tau_skewness'] = product['tau_skewness'][~np.isnan(product['LWP_mean'])]
    data_scene['tau_kurtosis'] = product['tau_kurtosis'][~np.isnan(product['LWP_mean'])]
    data_scene['CDNC_mean'] = product['CDNC_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['CDNC_std'] = product['CDNC_std'][~np.isnan(product['LWP_mean'])]
    data_scene['CDNC_median'] = product['LWP_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['re_mean'] = product['re_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['CF_mean'] = product['CF_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['CF_MODIS_mean'] = product['CF_MODIS_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['FFT_1D_max'] = product['FFT_1D_max'][~np.isnan(product['LWP_mean'])]
    data_scene['reflectance_mean'] = product['reflectance_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['reflectance_std'] = product['reflectance_std'][~np.isnan(product['LWP_mean'])]
    data_scene['reflectance_skewness'] = product['reflectance_skewness'][~np.isnan(product['LWP_mean'])]
    data_scene['albedo_mean'] = product['albedo_mean'][~np.isnan(product['LWP_mean'])]
    data_scene['f_name'] = np.size(data_scene['reflectance_mean']) * modis_data['f_name']#[~np.isnan(product['f_name'])]
    data_scene['lat_center'] = product['lat_center'][~np.isnan(product['LWP_mean'])]
    data_scene['lon_center'] = product['lon_center'][~np.isnan(product['LWP_mean'])]

    return data_scene

def power_spectrum_analysis(scene):
    # reference: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    # Input: A 2D array with equal x and y dims
    dims = np.shape(scene)[0]
    fourier_image = np.fft.fftn(scene)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreq = np.fft.fftfreq(dims) * dims
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, dims+1., 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic = "mean", bins = kbins)
    Abins *= 4. * np.pi / 3. * (kbins[1:] ** 3 - kbins[:-1] ** 3)

def power_spectrum_line(scene):
    # reference: https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    # Input: A 2D array with equal x and y dims
    dims = np.shape(scene)[0]
    data_line = scene[dims//2,:]
    data_line_detrended = signal.detrend(data_line)
    f, Pxx_den = scipy.signal.welch(data_line_detrended, nperseg=256, axis=0)
    k_max = f[np.argmax(Pxx_den)]
    lambda_max = 1/k_max
    return lambda_max
'''
    fig, axarr = plt.subplots()
    axarr.loglog(f, Pxx_den)
    axarr.set_xticks([10 ** -2, 10 ** -1, 1])  # , np.exp(1)**(np.arange(-5,1,1)))
    axarr.set_xticklabels([0.01, 0.1, 1])  # , np.exp(1)**(np.arange(-5,1,1)))
    axarr.set_xlim([0.01, 1])
    axarr.set_xlabel('k [km$^{-1}$]')
'''