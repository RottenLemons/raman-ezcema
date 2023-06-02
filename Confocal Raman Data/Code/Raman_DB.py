# Date: 16 June 2021
# This script contains functions to combine or preprocess the Raman spectra.
##############################################################################
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.signal 
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

def AT_sort (temp):  # temp is raw data from AT(row is step, column is no of wl)
    data_shape=np.shape(temp)
    steps=int((data_shape[0])/2)
    num_spec=data_shape[1]
    
    data_671=np.zeros((steps,num_spec))
    data_785=np.zeros((steps,num_spec))
    
    for i in range(data_shape[0]-1):  # starting from 0
                   
        if (i%2)==0:
            data_671[int(i/2),:]=temp[i+1,:]
        else:
            data_785[int(i/2),:]=temp[i+1,:]
    return data_671, data_785 # first row is wl

# xx is wl ?
def smooth (data_array, N): # moving average N is order
    result=np.convolve(data_array, np.ones((N,))/N, mode='same')
    return result

def smooth1(data):
    x = savgol_filter(data, 11, 2) # window length and polynomial order
    return x

def baseline_als(intensity, lam, p, niter=10):
  L = len(intensity)
  D = scipy.sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = scipy.sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = scipy.sparse.linalg.spsolve(Z, w*intensity)
    w = p * (intensity > z) + (1-p) * (intensity < z)
  return z

def multi_fit (xx, data_array, med_N=11, loop_N=40, order=20): # fit the background
    y_raw1=scipy.signal.medfilt(data_array, kernel_size=med_N)
    y_raw2=y_raw1.copy()
    y_fit=y_raw1.copy()
    for i in range(loop_N):
        coefs=np.polynomial.polynomial.polyfit(xx, y_raw1, order)
        y_fit=np.polynomial.polynomial.polyval(xx,coefs)
        y_diff=y_raw2-y_fit
        y_diff[y_diff>0]=0  #0
        y_raw1=y_raw1+y_diff
    fdata=y_raw2 - y_fit
    return fdata, y_fit

def multi_fit_2D (data_2D, fit_range=None, med_N=11, loop_N=40): # deback of multi depths
    data_shape=np.shape(data_2D)
    if fit_range is None:
        fit_range=np.arange(1, data_shape[0])
    fit_range_shape=np.shape(fit_range)
    y=np.zeros([fit_range_shape[0]+1, data_shape[1]])
    j=1
    y[0]=data_2D[0]
    for i in fit_range:
        y[j],_=multi_fit (data_2D[0], data_2D[i], med_N, loop_N)
        j=j+1
    return y

def wl2wn (xx, central_wl): # convert wl to wn
    xx_wn = 1e7/central_wl - 1e7/xx
    return xx_wn

def select_wn_range (data, wn_low, wn_high):
    if np.shape(np.shape(data))[0] == 1:
        print ('Error! Please input a 2D array, [0] for WN, [1-N] for spectrums!')
        return
    wn_range = np.copy(data[0])
    wn_range[wn_range<wn_low]=0
    wn_range[wn_range>wn_high]=0
    selected_range=np.nonzero(wn_range)[0]
    y = data[:,selected_range[0]:selected_range[-1]]
    return y

def unmix_spec (AA, bb): # unmix 1 spectrum, AA is component list, AA[0] is wn, bb is 1D array spectrum
    if np.shape(bb)[0] != np.shape(AA)[1]:
        print('dimension is wrong')
        return
    result, rnorm=nnls(np.transpose(AA[1:]), bb)
    simu_data=np.zeros(np.shape(AA[1]))
    for i in range(np.shape(AA)[0]-1):
        simu_data=simu_data+AA[i+1]*result[i]
    return result, simu_data

def unmix_spec_2D (AA, bb): # umix spectrums at all depth
    if np.shape(np.shape(bb))[0] != 2:
        print('Please input a 2D array, [0] = WN, [1-N] = spectrums!')
        return
    result = np.zeros([np.shape(bb)[0]-1, np.shape(AA)[0]-1])
    simu_data = np.zeros(np.shape(bb))
    simu_data[0]=bb[0]
    for i in range(np.shape(bb)[0]-1):
        result[i], simu_data[i+1] = unmix_spec (AA, bb[i+1])
    return result, simu_data


def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def preprocessing1(data):
    pp_data = pd.DataFrame()
    for i in range(0,len(data)):
        bc = baseline_als(data.iloc[i,:], pow(10,4), 0.0001)
        bc = data.iloc[i,:]-bc
        x = savgol_filter(bc, 11, 2) # window length and polynomial order
        x = x.reshape([len(x),1])
        # x = scaler.fit_transform(x)
        x = pd.DataFrame(x).T
        pp_data = pd.concat([pp_data, x], axis=0, ignore_index=True)    
    return pp_data

#smoothing using savgol and background correction. Outputs the processed data and the background data
def preprocessing2(data, lam, p):
    row, col = np.shape(data)
    x=np.zeros([row-1, col])
    y=np.zeros([row-1, col])
    z=np.zeros([row-1, col])
    for i in range(row-1):
        x[i,:] = savgol_filter(data[i+1,:], 11, 2)
        y[i,:]=x[i,:]-baseline_als(x[i,:], lam, p)
        z[i,:]=baseline_als(x[i,:], lam, p)
    return y, z


    
if __name__=="__main__":
#   data from confocal Raman
    sdata=pd.read_csv('F:\\Raman\Confocal data\AD\Patient\wrist\\012_20180528_1550_1_1.csv', header=None)
    sdata=sdata.as_matrix()
    
    xx=1e7/782.8 - 1e7/sdata[0]
    length=np.shape(sdata)
    fdata=np.zeros([length[0]-1, length[1]])
    fdata_smooth=fdata.copy()
    for n in range(length[0]-1):
        fdata[n]=rm.multi_fit(xx, sdata[n+1], med_N=19, loop_N=40)
        fdata_smooth[n]=rm.smooth(fdata[n],30)
    
    #%% data from riverD
    sdata_rd=pd.read_csv('ama_wrist.txt', sep='\t',header=None)
    sdata_rd=np.transpose(sdata_rd.as_matrix())
    
    xx_rd= sdata_rd[0]
    length=np.shape(sdata_rd)
    fdata_rd=np.zeros([length[0]-1, length[1]])
    for n in range(length[0]-1):
        fdata_rd[n]=rm.multi_fit(xx_rd, sdata_rd[n+1], med_N=11, loop_N=30)
    #%%
    plt.figure()
    plt.plot(xx[25:-25],fdata_smooth[3,25:-25]*10-9,xx_rd,fdata_rd[4])