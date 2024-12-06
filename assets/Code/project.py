import h5py
import numpy as np
from netCDF4 import Dataset,num2date
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import imageio
import scipy.io
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams['font.family'] = 'DejaVu Sans'


#%% get_data

# input: biov_data, slope_data, predictors
# input: Features

# output: pred_bv, pred_sp

file_path_POC = './POC_UVP5_onWOAgrid.mat'
file_POC = h5py.File(file_path_POC)

POC = file_POC['poc']
lon = POC['lon']
lat = POC['lat']
xlon,xlat = np.meshgrid(lon,lat)

# lat: -90 ~ 90, lon: -180 ~ 180

biov_data = np.array(POC['biov_tot']) # shape(12,102,180,360)
slope_data = np.array(POC['slope'])


#%%
biov_data = np.log10(biov_data + 0.01)  # Adding 0.01 to avoid log(0)

maxBV = np.nanmean(biov_data) + 5 * np.nanstd(biov_data)
biov_data[biov_data > maxBV] = np.nan

slope_data = np.abs(slope_data)

#%% topo

file_path_topo = './etopo2.nc'
file_topo = Dataset(file_path_topo)

topo_val = np.array(file_topo.variables['topo'])

rows = np.arange(14, 5400, 30)
cols = np.arange(14, 10800, 30)

topo_val = -topo_val[rows, :][:, cols]
#%%

topo_val[topo_val < 0] = 0
topo_val[topo_val > 5500] = 5500

#%%
file_path_depth = './woa18_all_n00_01.nc'
file_depth = Dataset(file_path_depth)

depth_bnds = np.array(file_depth.variables['depth_bnds'])

depth_mean = np.mean(depth_bnds,axis=1)

#%%

topo_val = np.tile(topo_val, (12, 102, 1, 1))
tp_msk = np.full((12, 102, 180, 360), np.nan)

tp_msk[topo_val > 0 ] = 1

#%%

oxy = ['o2', 'aou']
nut = ['no3', 'po4']
chl = ['chl_gc', 'chl_md']
npp = ['vgpm', 'cbpm', 'cafe', 'epp']
mld = ['mld_dbm', 'mld_mm']
irn = ['LFE', 'SFE']
zeu = ['zeu_cbpm', 'zeu_vgpm', 'zeu_epp']

#%%
file_path_predictors = './predictors_3d.mat'
file_predictors = h5py.File(file_path_predictors)

#%%
predictors = file_predictors['pred_3d']

#%% predictors

# temp, salt, ddepth, si, shwv

tmp_oxy = np.random.choice(oxy)
tmp_nut = np.random.choice(nut)
tmp_chl = np.random.choice(chl)
tmp_npp = np.random.choice(npp)
tmp_mld = np.random.choice(mld)
tmp_irn = np.random.choice(irn)
tmp_zeu = np.random.choice(zeu)

#%%
predictors_ddepth = np.array(predictors['ddepth'])

predictors_ddepth = np.concatenate([predictors_ddepth[:,:,:,155:],predictors_ddepth[:,:,:,:155]],axis=-1)

#%%
predictors_temp = np.array(predictors['temp'])

predictors_temp = np.concatenate([predictors_temp[:,:,:,155:],predictors_temp[:,:,:,:155]],axis=-1)

predictors_temp_ddd = np.full((12,102,180,360),np.nan)

for month in range(12):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate depths
            for depth_idx in range(1, 101):
                predictors_temp_ddd[month, depth_idx, lat, lon] = (predictors_temp[month, depth_idx + 1, lat, lon] - predictors_temp[month, depth_idx - 1, lat, lon]) / (depth_mean[depth_idx + 1] - depth_mean[depth_idx - 1])

            # Forward difference for the first depth
            predictors_temp_ddd[month, 0, lat, lon] = (predictors_temp[month, 1, lat, lon] - predictors_temp[month, 0, lat, lon]) / (depth_mean[1] - depth_mean[0])

            # Backward difference for the last depth
            predictors_temp_ddd[month, 101, lat, lon] = (predictors_temp[month, 101, lat, lon] - predictors_temp[month, 100, lat, lon]) / (depth_mean[101] - depth_mean[100])

print(np.nanmean(predictors_temp_ddd))

predictors_temp_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_temp_ddt[month, depth_idx, lat, lon] = (predictors_temp[month + 1, depth_idx, lat, lon] - predictors_temp[month - 1, depth_idx, lat, lon]) / 2

            predictors_temp_ddt[0, depth_idx, lat, lon] = (predictors_temp[1, depth_idx, lat, lon] - predictors_temp[11, depth_idx, lat, lon]) / 2

            predictors_temp_ddt[11, depth_idx, lat, lon] = (predictors_temp[0, depth_idx, lat, lon] - predictors_temp[10, depth_idx, lat, lon]) / 2


#%%
predictors_salt = np.array(predictors['salt'])

predictors_salt = np.concatenate([predictors_salt[:,:,:,155:],predictors_salt[:,:,:,:155]],axis=-1)

predictors_salt_ddd = np.full((12,102,180,360),np.nan)

for month in range(12):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate depths
            for depth_idx in range(1, 101):
                predictors_salt_ddd[month, depth_idx, lat, lon] = (predictors_salt[month, depth_idx + 1, lat, lon] - predictors_salt[month, depth_idx - 1, lat, lon]) / (depth_mean[depth_idx + 1] - depth_mean[depth_idx - 1])

            # Forward difference for the first depth
            predictors_salt_ddd[month, 0, lat, lon] = (predictors_salt[month, 1, lat, lon] - predictors_salt[month, 0, lat, lon]) / (depth_mean[1] - depth_mean[0])

            # Backward difference for the last depth
            predictors_salt_ddd[month, 101, lat, lon] = (predictors_salt[month, 101, lat, lon] - predictors_salt[month, 100, lat, lon]) / (depth_mean[101] - depth_mean[100])


predictors_salt_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_salt_ddt[month, depth_idx, lat, lon] = (predictors_salt[month + 1, depth_idx, lat, lon] - predictors_salt[month - 1, depth_idx, lat, lon]) / 2

            predictors_salt_ddt[0, depth_idx, lat, lon] = (predictors_salt[1, depth_idx, lat, lon] - predictors_salt[11, depth_idx, lat, lon]) / 2

            predictors_salt_ddt[11, depth_idx, lat, lon] = (predictors_salt[0, depth_idx, lat, lon] - predictors_salt[10, depth_idx, lat, lon]) / 2


#%%
predictors_si = np.array(predictors['si'])

predictors_si = np.concatenate([predictors_si[:,:,:,155:],predictors_si[:,:,:,:155]],axis=-1)

predictors_si_ddd = np.full((12,102,180,360),np.nan)

for month in range(12):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate depths
            for depth_idx in range(1, 101):
                predictors_si_ddd[month, depth_idx, lat, lon] = (predictors_si[month, depth_idx + 1, lat, lon] - predictors_si[month, depth_idx - 1, lat, lon]) / (depth_mean[depth_idx + 1] - depth_mean[depth_idx - 1])

            # Forward difference for the first depth
            predictors_si_ddd[month, 0, lat, lon] = (predictors_si[month, 1, lat, lon] - predictors_si[month, 0, lat, lon]) / (depth_mean[1] - depth_mean[0])

            # Backward difference for the last depth
            predictors_si_ddd[month, 101, lat, lon] = (predictors_si[month, 101, lat, lon] - predictors_si[month, 100, lat, lon]) / (depth_mean[101] - depth_mean[100])


predictors_si_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_si_ddt[month, depth_idx, lat, lon] = (predictors_si[month + 1, depth_idx, lat, lon] - predictors_si[month - 1, depth_idx, lat, lon]) / 2

            predictors_si_ddt[0, depth_idx, lat, lon] = (predictors_si[1, depth_idx, lat, lon] - predictors_si[11, depth_idx, lat, lon]) / 2

            predictors_si_ddt[11, depth_idx, lat, lon] = (predictors_si[0, depth_idx, lat, lon] - predictors_si[10, depth_idx, lat, lon]) / 2



#%%
predictors_shwv = np.array(predictors['shwv'])

predictors_shwv = np.concatenate([predictors_shwv[:,:,:,155:],predictors_shwv[:,:,:,:155]],axis=-1)

predictors_shwv_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_shwv_ddt[month, depth_idx, lat, lon] = (predictors_shwv[month + 1, depth_idx, lat, lon] - predictors_shwv[month - 1, depth_idx, lat, lon]) / 2

            predictors_shwv_ddt[0, depth_idx, lat, lon] = (predictors_shwv[1, depth_idx, lat, lon] - predictors_shwv[11, depth_idx, lat, lon]) / 2

            predictors_shwv_ddt[11, depth_idx, lat, lon] = (predictors_shwv[0, depth_idx, lat, lon] - predictors_shwv[10, depth_idx, lat, lon]) / 2


#%%
predictors_oxy = np.array(predictors[tmp_oxy])

predictors_oxy = np.concatenate([predictors_oxy[:,:,:,155:],predictors_oxy[:,:,:,:155]],axis=-1)

predictors_oxy_ddd = np.full((12,102,180,360),np.nan)

for month in range(12):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate depths
            for depth_idx in range(1, 101):
                predictors_oxy_ddd[month, depth_idx, lat, lon] = (predictors_oxy[month, depth_idx + 1, lat, lon] - predictors_oxy[month, depth_idx - 1, lat, lon]) / (depth_mean[depth_idx + 1] - depth_mean[depth_idx - 1])

            # Forward difference for the first depth
            predictors_oxy_ddd[month, 0, lat, lon] = (predictors_oxy[month, 1, lat, lon] - predictors_oxy[month, 0, lat, lon]) / (depth_mean[1] - depth_mean[0])

            # Backward difference for the last depth
            predictors_oxy_ddd[month, 101, lat, lon] = (predictors_oxy[month, 101, lat, lon] - predictors_oxy[month, 100, lat, lon]) / (depth_mean[101] - depth_mean[100])


predictors_oxy_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_oxy_ddt[month, depth_idx, lat, lon] = (predictors_oxy[month + 1, depth_idx, lat, lon] - predictors_oxy[month - 1, depth_idx, lat, lon]) / 2

            predictors_oxy_ddt[0, depth_idx, lat, lon] = (predictors_oxy[1, depth_idx, lat, lon] - predictors_oxy[11, depth_idx, lat, lon]) / 2

            predictors_oxy_ddt[11, depth_idx, lat, lon] = (predictors_oxy[0, depth_idx, lat, lon] - predictors_oxy[10, depth_idx, lat, lon]) / 2


#%%
predictors_nut = np.array(predictors[tmp_nut])

predictors_nut = np.concatenate([predictors_nut[:,:,:,155:],predictors_nut[:,:,:,:155]],axis=-1)

predictors_nut_ddd = np.full((12,102,180,360),np.nan)

for month in range(12):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate depths
            for depth_idx in range(1, 101):
                predictors_nut_ddd[month, depth_idx, lat, lon] = (predictors_nut[month, depth_idx + 1, lat, lon] - predictors_nut[month, depth_idx - 1, lat, lon]) / (depth_mean[depth_idx + 1] - depth_mean[depth_idx - 1])

            # Forward difference for the first depth
            predictors_nut_ddd[month, 0, lat, lon] = (predictors_nut[month, 1, lat, lon] - predictors_nut[month, 0, lat, lon]) / (depth_mean[1] - depth_mean[0])

            # Backward difference for the last depth
            predictors_nut_ddd[month, 101, lat, lon] = (predictors_nut[month, 101, lat, lon] - predictors_nut[month, 100, lat, lon]) / (depth_mean[101] - depth_mean[100])


predictors_nut_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_nut_ddt[month, depth_idx, lat, lon] = (predictors_nut[month + 1, depth_idx, lat, lon] - predictors_nut[month - 1, depth_idx, lat, lon]) / 2

            predictors_nut_ddt[0, depth_idx, lat, lon] = (predictors_nut[1, depth_idx, lat, lon] - predictors_nut[11, depth_idx, lat, lon]) / 2

            predictors_nut_ddt[11, depth_idx, lat, lon] = (predictors_nut[0, depth_idx, lat, lon] - predictors_nut[10, depth_idx, lat, lon]) / 2


#%%
predictors_chl = np.array(predictors[tmp_chl])

predictors_chl = np.concatenate([predictors_chl[:,:,:,155:],predictors_chl[:,:,:,:155]],axis=-1)


predictors_chl_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_chl_ddt[month, depth_idx, lat, lon] = (predictors_chl[month + 1, depth_idx, lat, lon] - predictors_chl[month - 1, depth_idx, lat, lon]) / 2

            predictors_chl_ddt[0, depth_idx, lat, lon] = (predictors_chl[1, depth_idx, lat, lon] - predictors_chl[11, depth_idx, lat, lon]) / 2

            predictors_chl_ddt[11, depth_idx, lat, lon] = (predictors_chl[0, depth_idx, lat, lon] - predictors_chl[10, depth_idx, lat, lon]) / 2


#%%
predictors_npp = np.array(predictors[tmp_npp])

predictors_npp = np.concatenate([predictors_npp[:,:,:,155:],predictors_npp[:,:,:,:155]],axis=-1)


predictors_npp_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_npp_ddt[month, depth_idx, lat, lon] = (predictors_npp[month + 1, depth_idx, lat, lon] - predictors_npp[month - 1, depth_idx, lat, lon]) / 2

            predictors_npp_ddt[0, depth_idx, lat, lon] = (predictors_npp[1, depth_idx, lat, lon] - predictors_npp[11, depth_idx, lat, lon]) / 2

            predictors_npp_ddt[11, depth_idx, lat, lon] = (predictors_npp[0, depth_idx, lat, lon] - predictors_npp[10, depth_idx, lat, lon]) / 2


#%%
predictors_mld = np.array(predictors[tmp_mld])

predictors_mld = np.concatenate([predictors_mld[:,:,:,155:],predictors_mld[:,:,:,:155]],axis=-1)


predictors_mld_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_mld_ddt[month, depth_idx, lat, lon] = (predictors_mld[month + 1, depth_idx, lat, lon] - predictors_mld[month - 1, depth_idx, lat, lon]) / 2

            predictors_mld_ddt[0, depth_idx, lat, lon] = (predictors_mld[1, depth_idx, lat, lon] - predictors_mld[11, depth_idx, lat, lon]) / 2

            predictors_mld_ddt[11, depth_idx, lat, lon] = (predictors_mld[0, depth_idx, lat, lon] - predictors_mld[10, depth_idx, lat, lon]) / 2


#%%
predictors_irn = np.array(predictors[tmp_irn])

predictors_irn = np.concatenate([predictors_irn[:,:,:,155:],predictors_irn[:,:,:,:155]],axis=-1)


predictors_irn_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_irn_ddt[month, depth_idx, lat, lon] = (predictors_irn[month + 1, depth_idx, lat, lon] - predictors_irn[month - 1, depth_idx, lat, lon]) / 2

            predictors_irn_ddt[0, depth_idx, lat, lon] = (predictors_irn[1, depth_idx, lat, lon] - predictors_irn[11, depth_idx, lat, lon]) / 2

            predictors_irn_ddt[11, depth_idx, lat, lon] = (predictors_irn[0, depth_idx, lat, lon] - predictors_irn[10, depth_idx, lat, lon]) / 2


#%%
predictors_zeu = np.array(predictors[tmp_zeu])


predictors_zeu = np.concatenate([predictors_zeu[:,:,:,155:],predictors_zeu[:,:,:,:155]],axis=-1)


predictors_zeu_ddt = np.full((12,102,180,360),np.nan)

for depth_idx in range(102):
    for lat in range(180):
        for lon in range(360):
            # Apply central difference for intermediate months
            for month in range(1, 11):
                predictors_zeu_ddt[month, depth_idx, lat, lon] = (predictors_zeu[month + 1, depth_idx, lat, lon] - predictors_zeu[month - 1, depth_idx, lat, lon]) / 2

            predictors_zeu_ddt[0, depth_idx, lat, lon] = (predictors_zeu[1, depth_idx, lat, lon] - predictors_zeu[11, depth_idx, lat, lon]) / 2

            predictors_zeu_ddt[11, depth_idx, lat, lon] = (predictors_zeu[0, depth_idx, lat, lon] - predictors_zeu[10, depth_idx, lat, lon]) / 2


#%% RandomForestRegressor

model = RandomForestRegressor(random_state=0, oob_score=True)


pred_bv = []
pred_sp = []

preds = np.column_stack((predictors_ddepth.flatten(),
                         predictors_temp.flatten(),predictors_temp_ddd.flatten(),predictors_temp_ddt.flatten(),
                         predictors_salt.flatten(),predictors_salt_ddd.flatten(),predictors_salt_ddt.flatten(),
                         predictors_si.flatten(),predictors_si_ddd.flatten(),predictors_si_ddt.flatten(),
                         predictors_shwv.flatten(),predictors_shwv_ddt.flatten(),
                         predictors_oxy.flatten(),predictors_oxy_ddd.flatten(),predictors_oxy_ddt.flatten(),
                         predictors_nut.flatten(),predictors_nut_ddd.flatten(),predictors_nut_ddt.flatten(),
                         predictors_chl.flatten(),predictors_chl_ddt.flatten(),
                         predictors_npp.flatten(),predictors_npp_ddt.flatten(),
                         predictors_mld.flatten(),predictors_mld_ddt.flatten(),
                         predictors_irn.flatten(),predictors_irn_ddt.flatten(),
                         predictors_zeu.flatten(),predictors_zeu_ddt.flatten(),))



#%%
x = preds
X = preds
y = np.column_stack((biov_data.flatten(), slope_data.flatten()))

# Remove rows with NaN values
idrem = np.unique(np.concatenate([np.where(np.isnan(np.mean(y, axis=1)))[0], np.where(np.isnan(np.mean(x, axis=1)))[0]]))
x = np.delete(x, idrem, axis=0)
y = np.delete(y, idrem, axis=0)

model.fit(x, y)

yhat = model.predict(x)
y_oob = model.oob_prediction_


def r2rmse(y_pred, y_true):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'r2': r2, 'rmse': rmse}



#%%

mask = np.mean(X, axis=1)
mask[~np.isnan(mask)] = 1
mask[np.isnan(mask)] = 0

X[np.isnan(X)] = 0
y_recon = model.predict(X)
y_recon = y_recon * mask[:, np.newaxis]


# Calculate R2 and RMSE for both biological variable and slope predictions
pred_bv.append({
    'oobPred_bv': y_oob[:, 0],
    'inBagPred_bv': yhat[:, 0],
    'keep_data_bv': y[:, 0],
    'oobStats': r2rmse(y_oob[:, 0], y[:, 0]),
    'inBagStats': r2rmse(yhat[:, 0], y[:, 0]),
    'recon':y_recon[:, 0].reshape(12,102,180,360)
})

pred_sp.append({
    'oobPred': y_oob[:, 1],
    'inBagPred': yhat[:, 1],
    'keep_data': y[:, 1],
    'oobStats': r2rmse(y_oob[:, 1], y[:, 1]),
    'inBagStats': r2rmse(yhat[:, 1], y[:, 1]),
    'recon':y_recon[:, 1].reshape(12,102,180,360)
})


# Apply topographic mask
pred_bv[0]['recon'] *= tp_msk
pred_sp[0]['recon'] *= tp_msk

np.save('pred_bv_data.npy',pred_bv)
np.save('pred_sp_data.npy',pred_sp)

