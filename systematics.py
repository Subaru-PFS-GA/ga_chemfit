import numpy as np
import scipy as scp
import hashlib
import chemfit
import zipfile
import re, io
from astropy.io import fits

suite = np.array([[-1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00, -1.500e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00, -3.000e+00,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01, -5.000e-01,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00, -1.000e+00,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01, -2.000e-01,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00,
        -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00, -2.000e+00],
       [ 6.307e+03,  4.330e+03,  5.705e+03,  4.873e+03,  5.696e+03,
         6.696e+03,  5.240e+03,  4.600e+03,  6.209e+03,  5.421e+03,
         6.307e+03,  4.330e+03,  5.705e+03,  4.873e+03,  5.696e+03,
         6.696e+03,  5.240e+03,  4.600e+03,  6.209e+03,  5.421e+03,
         6.307e+03,  4.330e+03,  5.705e+03,  4.873e+03,  5.696e+03,
         6.696e+03,  5.240e+03,  4.600e+03,  6.209e+03,  5.421e+03,
         6.881e+03,  4.643e+03,  5.570e+03,  6.417e+03,  5.096e+03,
         6.975e+03,  5.986e+03,  4.865e+03,  5.329e+03,  6.697e+03,
         6.881e+03,  4.643e+03,  5.570e+03,  6.417e+03,  5.096e+03,
         6.975e+03,  5.986e+03,  4.865e+03,  5.329e+03,  6.697e+03,
         6.881e+03,  4.643e+03,  5.570e+03,  6.417e+03,  5.096e+03,
         6.975e+03,  5.986e+03,  4.865e+03,  5.329e+03,  6.697e+03,
         5.656e+03,  3.831e+03,  4.972e+03,  5.129e+03,  4.316e+03,
         6.108e+03,  4.929e+03,  5.589e+03,  4.072e+03,  4.569e+03,
         5.656e+03,  3.831e+03,  4.972e+03,  5.129e+03,  4.316e+03,
         6.108e+03,  4.929e+03,  5.589e+03,  4.072e+03,  4.569e+03,
         5.656e+03,  3.831e+03,  4.972e+03,  5.129e+03,  4.316e+03,
         6.108e+03,  4.929e+03,  5.589e+03,  4.072e+03,  4.569e+03,
         6.074e+03,  4.128e+03,  5.301e+03,  5.421e+03,  4.596e+03,
         6.437e+03,  5.143e+03,  5.910e+03,  4.364e+03,  4.847e+03,
         6.074e+03,  4.128e+03,  5.301e+03,  5.421e+03,  4.596e+03,
         6.437e+03,  5.143e+03,  5.910e+03,  4.364e+03,  4.847e+03,
         6.074e+03,  4.128e+03,  5.301e+03,  5.421e+03,  4.596e+03,
         6.437e+03,  5.143e+03,  5.910e+03,  4.364e+03,  4.847e+03,
         5.435e+03,  3.687e+03,  4.782e+03,  4.964e+03,  4.156e+03,
         5.911e+03,  4.786e+03,  5.414e+03,  4.395e+03,  3.917e+03,
         5.435e+03,  3.687e+03,  4.782e+03,  4.964e+03,  4.156e+03,
         5.911e+03,  4.786e+03,  5.414e+03,  4.395e+03,  3.917e+03,
         5.435e+03,  3.687e+03,  4.782e+03,  4.964e+03,  4.156e+03,
         5.911e+03,  4.786e+03,  5.414e+03,  4.395e+03,  3.917e+03,
         6.423e+03,  4.554e+03,  5.969e+03,  5.095e+03,  6.577e+03,
         5.710e+03,  6.882e+03,  4.821e+03,  6.258e+03,  5.364e+03,
         6.423e+03,  4.554e+03,  5.969e+03,  5.095e+03,  6.577e+03,
         5.710e+03,  6.882e+03,  4.821e+03,  6.258e+03,  5.364e+03,
         6.423e+03,  4.554e+03,  5.969e+03,  5.095e+03,  6.577e+03,
         5.710e+03,  6.882e+03,  4.821e+03,  6.258e+03,  5.364e+03],
       [ 4.500e+00,  1.000e+00,  2.530e+00,  2.040e+00,  3.560e+00,
         4.130e+00,  2.830e+00,  1.510e+00,  3.790e+00,  3.250e+00,
         4.500e+00,  1.000e+00,  2.530e+00,  2.040e+00,  3.560e+00,
         4.130e+00,  2.830e+00,  1.510e+00,  3.790e+00,  3.250e+00,
         4.500e+00,  1.000e+00,  2.530e+00,  2.040e+00,  3.560e+00,
         4.130e+00,  2.830e+00,  1.510e+00,  3.790e+00,  3.250e+00,
         4.500e+00,  9.100e-01,  2.910e+00,  3.500e+00,  1.890e+00,
         3.810e+00,  3.230e+00,  1.400e+00,  2.410e+00,  3.670e+00,
         4.500e+00,  9.100e-01,  2.910e+00,  3.500e+00,  1.890e+00,
         3.810e+00,  3.230e+00,  1.400e+00,  2.410e+00,  3.670e+00,
         4.500e+00,  9.100e-01,  2.910e+00,  3.500e+00,  1.890e+00,
         3.810e+00,  3.230e+00,  1.400e+00,  2.410e+00,  3.670e+00,
         4.500e+00,  8.600e-01,  2.390e+00,  3.550e+00,  1.690e+00,
         4.140e+00,  2.970e+00,  3.840e+00,  1.270e+00,  2.160e+00,
         4.500e+00,  8.600e-01,  2.390e+00,  3.550e+00,  1.690e+00,
         4.140e+00,  2.970e+00,  3.840e+00,  1.270e+00,  2.160e+00,
         4.500e+00,  8.600e-01,  2.390e+00,  3.550e+00,  1.690e+00,
         4.140e+00,  2.970e+00,  3.840e+00,  1.270e+00,  2.160e+00,
         4.500e+00,  9.600e-01,  2.430e+00,  3.560e+00,  1.820e+00,
         4.130e+00,  3.010e+00,  3.800e+00,  1.380e+00,  2.330e+00,
         4.500e+00,  9.600e-01,  2.430e+00,  3.560e+00,  1.820e+00,
         4.130e+00,  3.010e+00,  3.800e+00,  1.380e+00,  2.330e+00,
         4.500e+00,  9.600e-01,  2.430e+00,  3.560e+00,  1.820e+00,
         4.130e+00,  3.010e+00,  3.800e+00,  1.380e+00,  2.330e+00,
         4.500e+00,  8.600e-01,  2.380e+00,  3.530e+00,  1.670e+00,
         4.160e+00,  2.960e+00,  3.870e+00,  2.100e+00,  1.250e+00,
         4.500e+00,  8.600e-01,  2.380e+00,  3.530e+00,  1.670e+00,
         4.160e+00,  2.960e+00,  3.870e+00,  2.100e+00,  1.250e+00,
         4.500e+00,  8.600e-01,  2.380e+00,  3.530e+00,  1.670e+00,
         4.160e+00,  2.960e+00,  3.870e+00,  2.100e+00,  1.250e+00,
         4.500e+00,  1.160e+00,  2.580e+00,  2.220e+00,  2.810e+00,
         3.440e+00,  4.140e+00,  1.670e+00,  3.720e+00,  2.830e+00,
         4.500e+00,  1.160e+00,  2.580e+00,  2.220e+00,  2.810e+00,
         3.440e+00,  4.140e+00,  1.670e+00,  3.720e+00,  2.830e+00,
         4.500e+00,  1.160e+00,  2.580e+00,  2.220e+00,  2.810e+00,
         3.440e+00,  4.140e+00,  1.670e+00,  3.720e+00,  2.830e+00],
       [-3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
        -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01, -3.000e-01,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,
         3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01,  3.000e-01]])

def approximate_convolution_weights(bins, x, sigma, clip = 5.0, mode = 'window', max_size = 10e6, flush_cache = False):
    global _approximate_convolution_weights_cache
    try:
        _approximate_convolution_weights_cache
    except:
        _approximate_convolution_weights_cache = {}
    if not np.all(bins[1:] > bins[:-1]):
        raise ValueError('Bin wavelengths must be strictly ascending')
    if not np.all(x[1:] > x[:-1]):
        raise ValueError('x must be strictly ascending')
    try:
        sigma[0]
    except:
        sigma = np.full(len(bins), sigma)
        mode = 'window'
    if not flush_cache:
        hash_string = ''.join(list(map(lambda arg: hashlib.sha256(bytes(arg)).hexdigest(), [bins, x, sigma]))) + str(clip) + mode
        if hash_string in _approximate_convolution_weights_cache:
            return _approximate_convolution_weights_cache[hash_string]
    if mode == 'window':
        if len(bins) != len(sigma):
            raise ValueError('In "window" mode, must provide sigma for each bin')
    elif mode == 'dispersion':
        if len(x) != len(sigma):
            raise ValueError('In "dispersion" mode, must provide sigma for each x')
    clip_start = np.zeros(len(bins), dtype = int)
    clip_end = np.zeros(len(bins), dtype = int)
    for i in range(len(bins)):
        if mode == 'dispersion':
            clip_indices = np.where(np.abs(x - bins[i]) < (clip * sigma))[0]
            if len(clip_indices) == 0:
                clip_start[i] = 0; clip_end[i] = 0
            else:
                clip_start[i] = np.min(clip_indices); clip_end[i] = np.max(clip_indices) + 1
        else:
            clip_start[i] = np.searchsorted(x, bins[i] - clip * sigma[i])
            clip_end[i] = np.searchsorted(x, bins[i] + clip * sigma[i])
    row_lengths = clip_end - clip_start
    nonzero_count = np.sum(row_lengths)
    if nonzero_count > max_size:
        raise ValueError('Maximum memory usage exceeded. Requested: {}, allowed: {}. See the optional `max_size` argument'.format(nonzero_count, max_size))
    row_end = np.cumsum(row_lengths)
    row_start = row_end - row_lengths
    row_indices = np.zeros(nonzero_count, dtype = int)
    col_indices = np.zeros(nonzero_count, dtype = int)
    for i in range(len(bins)):
        row_indices[row_start[i] : row_end[i]] = np.arange(clip_start[i], clip_end[i])
        col_indices[row_start[i] : row_end[i]] = i
    if mode == 'dispersion':
        sigma_all = sigma[row_indices]
    else:
        sigma_all = sigma[col_indices]
    x_edges = chemfit.get_bin_edges(x)
    C = scp.stats.norm.pdf(x[row_indices] - bins[col_indices], scale = sigma_all) * (x_edges[1:][row_indices] - x_edges[:-1][row_indices])
    C = scp.sparse.coo_matrix((C, (col_indices, row_indices)), shape = [len(bins), len(x)])
    _approximate_convolution_weights_cache[hash_string] = C
    return C

def gen_test(params, arms, exact_resample = True, add_continuum = True, SNR = 100, clear_nans = False):
    global __gen_test_continua_cache

    # Get stellar parameters
    if type(params) is not dict:
        params = {'teff': params[1], 'logg': params[2], 'zscale': params[0], 'alpha': params[3]} # If parameters are unlabelled, assume a row from `suite` was passed

    # Load lines
    interpolator = chemfit.ModelGridInterpolator(resample = False)
    wl, flux = interpolator(params)

    # Load continuum
    if add_continuum:
        try:
            points, continua = __gen_test_continua_cache
        except:
            grid = zipfile.ZipFile('/media/roman/TeraTwo/chemgrid.zip')
            pattern = '(M_over_H_([0-9.-]+)/alpha_over_M_([0-9.-]+)/O_over_M_0.0/Mg_over_M_0.0/Si_over_M_0.0/Fe_over_M_0.0/teff_([0-9.-]+)_logg_([0-9.-]+).fits)'
            points = re.findall(pattern, '\n'.join(grid.namelist()))
            continua = []
            for point in points:
                f = grid.open(point[0]); h = fits.open(io.BytesIO(f.read()))
                continuum = list(filter(lambda h: 'TABLE' in h.header and h.header['TABLE'] == 'Emergent spectrum', h))[0].data
                h.close(); f.close()
                continua += [np.interp(wl, continuum['Wavelength'], continuum['Continuum flux density'])]
            points = np.array(points).T[1:].astype(float).T
            __gen_test_continua_cache = points, continua
        flux *= scp.interpolate.griddata(points, continua, np.array([[params['zscale'], params['alpha'], params['teff'], params['logg']]]), rescale = True, method = 'nearest')[0]

    # Resample
    if not exact_resample:
        backup_resample = chemfit.convolution_weights
        chemfit.convolution_weights = approximate_convolution_weights
    wl, flux = chemfit.simulate_observation(wl, flux, detector_wl = arms, combine = False)
    if not exact_resample:
        chemfit.convolution_weights = backup_resample

    # Clear nans
    if clear_nans:
        for arm in wl:
            wl[arm] = wl[arm][~np.isnan(flux[arm])]
            flux[arm] = flux[arm][~np.isnan(flux[arm])]

    # Add ivar
    ivar = {}
    for arm in wl:
        ivar[arm] = flux[arm] / SNR

    return wl, flux, ivar

