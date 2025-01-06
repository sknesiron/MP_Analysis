import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.integrate import quad
import matplotx
import os
from datetime import datetime
plt.style.use(matplotx.styles.pitaya_smoothie["light"])

def make_output_folder(
    data_in: str, folder_name: str, base_folder=r"../../../Analyses"
):
    relpath_data = os.path.relpath(data_in, os.getcwd())
    data_type = relpath_data.split("\\")[4]
    folder_out = (
        f"{base_folder}/{data_type}/{datetime.today().strftime('%Y%m%d')}_{folder_name}"
    )
    os.makedirs(folder_out, exist_ok=True)

    return folder_out
def make_histogram(data, bins=None, density=True):
    if bins is None:
        bins=int(data.size/2)
    return np.histogram(data, bins=bins,density=density)

def get_bin_centers(bin_edges):
    return (bin_edges[:-1]+bin_edges[1:])/2

def hist_scale_factor(data,bins):
    return ((data.max()-data.min())/bins)

def get_peak_estimates(data, bins=None, min_dist_Da=50, height=20, smooth=1):
    counts, bin_edges=make_histogram(data,bins,density=False)
    bin_center=get_bin_centers(bin_edges)
    min_dist=max(1,min_dist_Da/hist_scale_factor(data,counts.size))
    smoothed_counts=gaussian_filter1d(counts,smooth)
    peaks,properties=find_peaks(smoothed_counts, distance=min_dist,threshold=0,width=1,height=height)
    counts_dens, bin_edges=make_histogram(data,bins,density=True)
    
    est_A=counts[peaks]
    est_mu= bin_center[peaks]
    est_sig=bin_center[properties["right_ips"].astype("int")]-bin_center[properties["left_ips"].astype("int")]

    return np.array((est_A, est_mu, est_sig)).T,properties

def cdf(x, A,mu,sig):
    return A * sig * np.sqrt(2 * np.pi) / 2 * (1 + erf((x - mu) / (sig * np.sqrt(2))))

def gauss(x,A,mu,sig):
    return A * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

def cumulative_gaussian_mixture(x, *params):
    N = len(params) // 3
    A = params[:N]
    mu = params[N:2*N]
    sigma = params[2*N:3*N]
    F = np.zeros_like(x)
    for i in range(N):
        F+= cdf(x, A[i], mu[i], sigma[i])
    return F

# def target_function(params):
#     test=np.sum(np.square(cumulative_gaussian_mixture(data[0],*params)-data[1]))
#     print(f"{test}".ljust(20),end="\r")
#     return test

def generate_bounds(peak_estimate):
    bound_ar=np.zeros((*peak_estimate.shape,2))
    for i in range(0, peak_estimate.shape[1]):
        bound_ar[0,i,:]=np.array((0,peak_estimate[0,i]+peak_estimate[0,i]*0.2))
        bound_ar[1,i,:]=np.array((peak_estimate[1,i]-peak_estimate[2,i],peak_estimate[1,i]+peak_estimate[2,i]))
        bound_ar[2,i,:]=np.array((0,peak_estimate[2,i]))
    
    return bound_ar.reshape(-1,bound_ar.shape[2],order="F")

def multi_gaussian(x, *params):
    n_gaussians = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n_gaussians):
        A = params[3 * i]
        mu = params[3 * i + 1]
        sigma = params[3 * i + 2]
        y += gauss(x,A,mu,sigma)
    return y

def multi_gaussian_with_baseline(x, *params):
    """
    Computes a sum of Gaussian functions with a linear baseline correction.
    
    Parameters:
        x : array-like
            The independent variable.
        *params : float
            Parameters for the Gaussians and baseline:
            - For each Gaussian: amplitude (A), mean (mu), and standard deviation (sigma).
            - For baseline: slope (m) and intercept (c).
    
    Returns:
        y : array-like
            The computed values of the multi-Gaussian function with baseline correction.
    """
    n_gaussians = (len(params) - 2) // 3  # Last two parameters are for baseline
    y = np.zeros_like(x)
    
    # Add Gaussians
    for i in range(n_gaussians):
        A = params[3 * i]
        mu = params[3 * i + 1]
        sigma = params[3 * i + 2]
        y += gauss(x, A, mu, sigma)
    
    # Add baseline correction
    m = params[-2]  # Slope
    c = params[-1]  # Intercept
    y += m * x + c
    
    return y


from scipy.integrate import quad

def auc_gaussian_fwhm(amplitude, mean, sigma):
    # FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    # Integration limits (mean ± FWHM/2)
    x_min = mean - fwhm / 2
    x_max = mean + fwhm / 2
    # Define the Gaussian function
    def gaussian(x):
        return amplitude * np.exp(-((x - mean)**2) / (2 * sigma**2))
    # Integrate the Gaussian over the FWHM
    auc, _ = quad(gaussian, x_min, x_max)
    return auc