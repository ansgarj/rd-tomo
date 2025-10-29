# Imports
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
from multiprocessing import Pool
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import gamma
from scipy.ndimage import uniform_filter

# Multilook
def multilook(I: np.ndarray, ds: int, npar: int = os.cpu_count()):
    N, H, W= I.shape
    out_H = (H + ds - 1) // ds
    out_W = (W + ds - 1) // ds
    out = np.empty((N, out_H, out_W), dtype=I.dtype)

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=npar) as ex:
            futures = [ex.submit(_multilook_slice, I[n, ...], ds) for n in range(N)]
            for n, f in enumerate(tqdm(futures, desc="Multilooking: ", total = N, unit='slices',leave=False)):
                out[n, ...] = f.result()

    return out

def _multilook_slice(I2d, ds):
    # I2d is linear intensity, 2-D; implement your block-mean here (vectorized).
    H, W = I2d.shape
    out_H = (H + ds - 1) // ds
    out_W = (W + ds - 1) // ds
    h_full, w_full = H // ds, W // ds
    out = np.empty((out_H, out_W), dtype=I2d.dtype)

    if h_full > 0 and w_full > 0:
        core = I2d[:h_full*ds, :w_full*ds].reshape(h_full, ds, w_full, ds)
        out[:h_full, :w_full] = core.mean(axis=(1, 3))
    if w_full*ds < W:
        right = I2d[:h_full*ds, w_full*ds:].reshape(h_full, ds, W - w_full*ds)
        out[:h_full, w_full] = right.mean(axis=(1, 2))
    if h_full*ds < H:
        bottom = I2d[h_full*ds:, :w_full*ds].reshape(H - h_full*ds, w_full, ds)
        out[h_full, :w_full] = bottom.mean(axis=(0, 2))
    if h_full*ds < H and w_full*ds < W:
        out[h_full, w_full] = I2d[h_full*ds:, w_full*ds:].mean()
    return out

# Filter 
def filter(I: np.ndarray, sigma_xi: float = 0.9, size: int = 9, point_percentile: float = 98.0,
           point_threshold: int = 9, nlooks: int = 1, npar: int = os.cpu_count()):
    sigma_range = _estimate_sigma_range(nlooks, sigma_xi)
    point_mask, mean_estimate = _point_target_estimator(I, percentile=point_percentile, min_voxels=point_threshold,
                                                       nlooks=nlooks)

    # Prepare per-slice arguments
    args_list = [(I[z, ...], point_mask[z, ...], mean_estimate[z, ...], sigma_range, size, nlooks) for z in range(I.shape[0])]

    # Filter each slice in parallel
    with Pool(processes=npar) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(_filter_wrapper, args_list),
                        total=len(args_list), desc="Filtering: ", unit="slices", leave=False):
            results.append(result)

    out = np.stack(results, axis=0)
    return out

def _estimate_sigma_range(nlooks, sigma_xi):
    """
    Estimate the sigma range for multiplicative noise using gamma distribution.
    """
    shape = nlooks
    scale = 1.0 / nlooks
    lower_bound = gamma.ppf((1 - sigma_xi) / 2, a=shape, scale=scale)
    upper_bound = gamma.ppf(1 - (1 - sigma_xi) / 2, a=shape, scale=scale)
    return lower_bound, upper_bound

def _point_target_estimator(volume, percentile, min_voxels, nlooks):
    """
    Identify point targets in a 3D tomogram using a percentile threshold and 3x3x3 neighborhood.
    Also compute MMSE-based mean estimate for non-point voxels assuming multiplicative noise.

    Parameters:
        volume (ndarray): 3D numpy array representing the tomogram (intensity image).
        percentile (float): Percentile threshold to identify bright voxels.
        min_voxels (int): Minimum number of bright voxels in a 3x3x3 neighborhood to classify as point target.
        n_looks (int): Number of looks (used for MMSE estimation).

    Returns:
        point_mask (ndarray): Boolean 3D array where True indicates point target voxels.
        mean_estimate (ndarray): 3D array of estimated means for non-point voxels.
    """
    # Compute threshold from percentile
    threshold = np.percentile(volume, percentile)

    # Pad the volume symmetrically
    padded = np.pad(volume, 1, mode='symmetric')

    # Create binary mask of high-intensity voxels
    high_intensity_mask = (padded >= threshold).astype(np.uint8)

    # Count number of high-intensity voxels in 3x3x3 neighborhood
    neighborhood_sum = uniform_filter(high_intensity_mask.astype(np.float32), size=3, mode='nearest') * 27
    neighborhood_sum = neighborhood_sum[1:-1, 1:-1, 1:-1]

    # Initial point target mask
    point_mask = (volume >= threshold) & (neighborhood_sum >= min_voxels)

    # Initialize mean estimate volume
    mean_estimate = np.zeros_like(volume, dtype=np.float32)

    # MMSE estimation for non-point voxels
    noise_var = 1.0 / nlooks
    padded_volume = np.pad(volume, 1, mode='symmetric')

    # Compute local mean and squared mean
    local_mean = uniform_filter(volume, size=3, mode='reflect')
    local_sq_mean = uniform_filter(volume**2, size=3, mode='reflect')

    # Compute local variance
    local_var = local_sq_mean - local_mean**2

    b = (local_var - noise_var * local_mean**2) / ((1 + noise_var) * local_var)
    b = np.clip(b, 0, 1)  # Ensure b ≥ 0

    # Apply formula
    mean_estimate = b * volume + (1 - b) * local_mean

    # Restore original values where point_mask is True
    mean_estimate[point_mask] = volume[point_mask]

    return point_mask, mean_estimate

def _filter_slice(image, point_mask, mean_estimate, size, sigma_range, nlooks):
    """
    Apply an approximate version of the improved Lee filter to non-point pixels within the sigma range.
    """

    pad_size = size // 2
    if size == 2 * pad_size:
        size += 1

    height, width = image.shape
    padded_image = np.pad(image, pad_size, mode='symmetric')
    windows = sliding_window_view(padded_image, (size, size))  # shape: (H, W, size, size)
    windows = windows.reshape(height, width, -1)  # flatten each window

    # Compute bounds
    bounds_low = mean_estimate * sigma_range[0]
    bounds_high = mean_estimate * sigma_range[1]

    # Mask out-of-bound values
    mask = (windows > bounds_low[..., None]) & (windows < bounds_high[..., None])
    valid_windows = np.where(mask, windows, np.nan)

    # Compute local mean and variance ignoring NaNs
    local_mean = np.nanmean(valid_windows, axis=-1)
    local_var = np.nanvar(valid_windows, axis=-1)

    # Compute b
    noise_var = 1.0 / nlooks
    b = (local_var - noise_var * local_mean**2) / ((1 + noise_var) * local_var)
    b = np.clip(b, 0, 1)

    # Final output
    output = b * image + (1 - b) * local_mean

    # Restore original values for masked points
    output[point_mask] = image[point_mask]


    return output

def _filter_wrapper(args):
        I_slice, point_mask, mean_estimate, sigma_range, size, nlooks = args
        return _filter_slice(I_slice, point_mask, mean_estimate,
                            sigma_range=sigma_range, size=size, nlooks=nlooks)
    
# Circularize
def circularize(I: np.ndarray, rescale: bool = False) -> np.ndarray:
    # Validate input dimensions
    if I.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D or 3D array.")
    if not (I.shape[1] == I.shape[2] or abs(I.shape[1] - I.shape[2]) == 1):
        raise ValueError("Input must be square or nearly square in horizontal dimensions.")

    # Get image dimensions
    sz = I.shape[1:3]
    N = 1 if I.ndim == 2 else I.shape[0]

    # Nominal radius
    r0 = max(sz) / 2

    # Find center
    xcenter = (sz[0] + 1) / 2
    ycenter = (sz[1] + 1) / 2

    # Coordinate vectors
    x = np.arange(1, sz[0] + 1) - xcenter
    y = np.arange(1, sz[1] + 1) - ycenter

    # Create coordinate grid
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Calculate radius grid
    R = np.sqrt(X**2 + Y**2)

    # Create mask
    mask = R <= r0

    # Apply mask to each layer
    if I.ndim == 2:
        J = np.where(mask, I, np.nan + np.nan*1j if np.iscomplexobj(I) else np.nan)
    else:
        J = np.empty_like(I, dtype=complex if np.iscomplexobj(I) else float)
        for i in range(N):
            J[i ,:, :] = np.where(mask, I[i, :, :], np.nan + np.nan*1j if np.iscomplexobj(I) else np.nan)

    # Rescale if requested
    if rescale:
        max_val = np.nanmax(np.abs(J))
        if max_val != 0:
            J = J / max_val

    return J
