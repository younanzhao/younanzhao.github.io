#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:14:32 2024

@author: zhaoyanchu
"""

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

pred_bv = np.load('/Users/zhaoyanchu/Documents/MATLAB/Scripts_for_Publish/pred_bv_data.npy',allow_pickle=True)
pred_sp = np.load('/Users/zhaoyanchu/Documents/MATLAB/Scripts_for_Publish/pred_sp_data.npy',allow_pickle=True)

file_path_POC = '/Users/zhaoyanchu/Documents/MATLAB/Scripts_for_Publish/POC_UVP5_onWOAgrid.mat'
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

#%%

fig1 = plt.figure(figsize=(10,7),dpi=500)
ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
m1 = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,llcrnrlon=-180.,urcrnrlon=180.,resolution='l')
m1.drawcoastlines(linewidth=0.5)
m1.drawcountries(linewidth=0.5)
m1.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],fontsize=12)
m1.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=12)
ax1.set_title('Original BioVolume Data at 100m',fontsize=16)
x,y = m1(xlon,xlat)
cs1 = m1.scatter(x,y,c=np.nanmean(biov_data,axis=0)[18,:,:],cmap='PuBuGn',vmin=-2,vmax=1,s=0.5)
#cs2 = m.contour(x,y,hgt_500_mean,levels=np.arange(588,589,1),colors='k',linewidths=1.5)
#plt.clabel(cs2,inline=True,fontsize=12)
position = fig1.add_axes([0.05,0.1,0.9,0.03])
cbar1 = fig1.colorbar(cs1,cax=position,orientation='horizontal',\
                   pad=0.08,fraction=0.03,ticks=list(np.arange(-2,1.01,0.2)),extend='both')
#cbar.set_label('m',fontdict={'family':'serif','color':'black','weight':'normal','size':12,})
plt.savefig("plot1.png", dpi=300)


#%%
fig2 = plt.figure(figsize=(10,7),dpi=500)
ax2 = fig2.add_axes([0.1,0.1,0.8,0.8])
m2 = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,llcrnrlon=-180.,urcrnrlon=180.,resolution='l')
m2.drawcoastlines(linewidth=0.5)
m2.drawcountries(linewidth=0.5)
m2.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],fontsize=12)
m2.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=12)
ax2.set_title('Original Slope Data at 100m',fontsize=16)
x,y = m2(xlon,xlat)
cs2 = m2.scatter(x,y,c=np.nanmean(slope_data,axis=0)[18,:,:],cmap='PuBuGn',vmin=3,vmax=5,s=0.5)
#plt.clabel(cs2,inline=True,fontsize=12)
position = fig2.add_axes([0.05,0.1,0.9,0.03])
cbar2 = fig2.colorbar(cs2,cax=position,orientation='horizontal',\
                    pad=0.08,fraction=0.03,ticks=list(np.arange(3,5.01,0.1)),extend='both')
#cbar.set_label('m',fontdict={'family':'serif','color':'black','weight':'normal','size':12,})

plt.savefig("plot2.png", dpi=300)

#%%

fig3 = plt.figure(figsize=(10,7),dpi=500)
ax3 = fig3.add_axes([0.1,0.1,0.8,0.8])
m3 = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,llcrnrlon=-180.,urcrnrlon=180.,resolution='l')
m3.drawcoastlines(linewidth=0.5)
m3.drawcountries(linewidth=0.5)
m3.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],fontsize=12)
m3.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=12)
ax3.set_title('Extrapolated BioVolume Data at 100m',fontsize=16)
x,y = m3(xlon,xlat)
cs3 = m3.contourf(x,y,np.nanmean(pred_bv[0]['recon'],axis=0)[18,:,:],levels=np.arange(-2,1.01,0.01),cmap='PuBuGn',extend='both')
cs5 = m3.scatter(x,y,c=np.nanmean(biov_data,axis=0)[18,:,:],cmap='PuBuGn',vmin=-2,vmax=1,s=0.5)
#cs2 = m.contour(x,y,hgt_500_mean,levels=np.arange(588,589,1),colors='k',linewidths=1.5)
#plt.clabel(cs2,inline=True,fontsize=12)
position = fig3.add_axes([0.05,0.1,0.9,0.03])
cbar3 = fig3.colorbar(cs3,cax=position,orientation='horizontal',\
                    pad=0.08,fraction=0.03,ticks=list(np.arange(-2,1.01,0.2)),extend='both')
#cbar3.set_label('m',fontdict={'family':'serif','color':'black','weight':'normal','size':12,})

plt.savefig("plot3.png", dpi=300)
#%%
fig4 = plt.figure(figsize=(10,7),dpi=500)
ax4 = fig4.add_axes([0.1,0.1,0.8,0.8])
m4 = Basemap(projection='cyl',llcrnrlat=-90.,urcrnrlat=90.,llcrnrlon=-180.,urcrnrlon=180.,resolution='l')
m4.drawcoastlines(linewidth=0.5)
m4.drawcountries(linewidth=0.5)
m4.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0],fontsize=12)
m4.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1],fontsize=12)
ax4.set_title('Extrapolated Slope Data at 100m',fontsize=16)
x,y = m4(xlon,xlat)
cs4 = m4.contourf(x,y,np.nanmean(pred_sp[0]['recon'],axis=0)[18,:,:],levels=np.arange(3,5.01,0.01),cmap='PuBuGn',extend='both')
cs6 = m4.scatter(x,y,c=np.nanmean(slope_data,axis=0)[18,:,:],cmap='PuBuGn',vmin=3,vmax=5,s=0.5)
#cs2 = m.contour(x,y,hgt_500_mean,levels=np.arange(588,589,1),colors='k',linewidths=1.5)
#plt.clabel(cs2,inline=True,fontsize=12)
position = fig4.add_axes([0.05,0.1,0.9,0.03])
cbar4 = fig4.colorbar(cs4,cax=position,orientation='horizontal',\
                    pad=0.08,fraction=0.03,ticks=list(np.arange(3,5.1,0.1)),extend='both')
#cbar.set_label('m',fontdict={'family':'serif','color':'black','weight':'normal','size':12,})

plt.savefig("plot4.png", dpi=300)


#%%

print(pred_bv[0]['oobStats'])

print(pred_bv[0]['inBagStats'])

print(pred_sp[0]['oobStats'])

print(pred_sp[0]['inBagStats'])

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KernelDensity

#%% BV

x = pred_bv[0]['keep_data_bv']
y = pred_bv[0]['inBagPred_bv']

sample_size = 10000  # consider the large number of samples, just randomly chose 10000 data points
indices = np.random.choice(len(x), size=sample_size, replace=False)
x_sampled = x[indices]
y_sampled = y[indices]

# Use gaussian_kde to estimate the density
xy = np.vstack([x_sampled, y_sampled])
kde = gaussian_kde(xy)

density = kde(xy)

# normalized density [0,1]
scaler = MinMaxScaler()
density_normalized = scaler.fit_transform(density.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 6),dpi=500)
scatter = plt.scatter(x_sampled, y_sampled, c=density_normalized, cmap='viridis', alpha=0.7,s=1)

plt.plot([-2,2],[-2,2],'k--')
plt.colorbar(scatter, label='Normalized Density')

plt.title('Performance - predicted biovolume',fontsize=16)
plt.xlabel('Observed',fontsize=16)
plt.ylabel('Predicted',fontsize=16)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.text(-1.8,1.6,'$R^2$ = 0.9905',fontsize=16)
plt.text(-1.8,1.2,'rmse = 0.0553',fontsize=16)

plt.savefig("plot5.png", dpi=300)

#%%
x = pred_bv[0]['keep_data_bv']
y = pred_bv[0]['oobPred_bv']

sample_size = 10000 
indices = np.random.choice(len(x), size=sample_size, replace=False)
x_sampled = x[indices]
y_sampled = y[indices]

xy = np.vstack([x_sampled, y_sampled])
kde = gaussian_kde(xy)

density = kde(xy)

scaler = MinMaxScaler()
density_normalized = scaler.fit_transform(density.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 6),dpi=500)
scatter = plt.scatter(x_sampled, y_sampled, c=density_normalized, cmap='viridis', alpha=0.7,s=1)

plt.plot([-2,2],[-2,2],'k--',)

plt.colorbar(scatter, label='Normalized Density')

plt.title('Out-of-bag performance - predicted biovolume',fontsize=16)
plt.xlabel('Observed',fontsize=16)
plt.ylabel('OOB',fontsize=16)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.text(-1.8,1.6,'$R^2$ = 0.9315',fontsize=16)
plt.text(-1.8,1.2,'rmse = 0.1488',fontsize=16)

plt.savefig("plot6.png", dpi=300)

#%% SP

x = pred_sp[0]['keep_data']
y = pred_sp[0]['inBagPred']

sample_size = 10000
indices = np.random.choice(len(x), size=sample_size, replace=False)
x_sampled = x[indices]
y_sampled = y[indices]

xy = np.vstack([x_sampled, y_sampled])
kde = gaussian_kde(xy)

density = kde(xy)

scaler = MinMaxScaler()
density_normalized = scaler.fit_transform(density.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 6),dpi=500)
scatter = plt.scatter(x_sampled, y_sampled, c=density_normalized, cmap='viridis', alpha=0.7,s=1)

plt.plot([2,5.5],[2,5.5],'k--')

plt.colorbar(scatter, label='Normalized Density')

plt.title('Performance - predicted slope',fontsize=16)
plt.xlabel('Observed',fontsize=16)
plt.ylabel('Predicted',fontsize=16)
plt.xlim(2,5.5)
plt.ylim(2,5.5)
plt.text(2.1,5.2,'$R^2$ = 0.9740',fontsize=16)
plt.text(2.1,4.9,'rmse = 0.0969',fontsize=16)

plt.savefig("plot7.png", dpi=300)


#%%
x = pred_sp[0]['keep_data']
y = pred_sp[0]['oobPred']

sample_size = 10000
indices = np.random.choice(len(x), size=sample_size, replace=False)
x_sampled = x[indices]
y_sampled = y[indices]

xy = np.vstack([x_sampled, y_sampled])
kde = gaussian_kde(xy)

density = kde(xy)

scaler = MinMaxScaler()
density_normalized = scaler.fit_transform(density.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 6),dpi=500)
scatter = plt.scatter(x_sampled, y_sampled, c=density_normalized, cmap='viridis', alpha=0.7,s=1)

plt.plot([2,5.5],[2,5.5],'k--')

plt.colorbar(scatter, label='Normalized Density')

plt.title('Out-of-bag performance - predicted slope',fontsize=16)
plt.xlabel('Observed',fontsize=16)
plt.ylabel('OOB',fontsize=16)
plt.xlim(2,5.5)
plt.ylim(2,5.5)
plt.text(2.1,5.2,'$R^2$ = 0.8117',fontsize=16)
plt.text(2.1,4.9,'rmse = 0.2609',fontsize=16)

plt.savefig("plot8.png", dpi=300)


