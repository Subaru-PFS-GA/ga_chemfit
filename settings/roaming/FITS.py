############################################################
#                                                          #
#                BasicATLAS/Palantir PRESET                #
#                                                          #
#   This preset allows chemfit to load synthetic spectra   #
#   from FITS files produced by BasicATLAS and/or palantir #
#   PHOENIX setups                                         #
#                                                          #
############################################################

import os, pickle
import glob
from astropy.io import fits
import scipy as scp

settings = {
    ### Model grid settings ###
    'griddir': original_settings['griddir'],    # Model directory must be specified in local settings

    ### Which parameters to fit? ###
    'fit_dof': ['zscale', 'alpha', 'teff', 'logg', 'redshift', 'quickblur'],

    ### Virtual grid dimensions ###
    'virtual_dof': {'redshift': [-200, 200], 'quickblur': [0, 0.5]},
}

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions
    
    This version of the function loads models from FITS files in the Palantir format.
    `settings['griddir']` must be pointed to the directories that contains the
    FITS files either in the top directory or subdirectories. Recursive search for
    *.fits files will be carried out at first call

    The function implements file caching
    
    Parameters
    ----------
    flush_cache : bool, optional
        If True, discard cache and read the grid afresh
    
    Returns
    -------
    dict
        Dictionary of lists, keyed be grid axis names. The lists contain unique
        values along the corresponding axis that are available in the grid
    """
    global __model_parameters

    # Apply caching
    if os.path.isfile(cache := (settings['griddir'] + '/cache.pkl')) and (not flush_cache):
        f = open(cache, 'rb')
        grid, __model_parameters = pickle.load(f)
        f.close()
        return grid

    grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': [], 'carbon': [], 'sodium': [], 'vanadium': []}
    __model_parameters = {}

    # Extract the parameters of all available models
    models = glob.glob(settings['griddir'] + '/*.fits') + glob.glob(settings['griddir'] + '/**/*.fits', recursive = True)
    for model in models:
        h = fits.open(model)
        try:
            index
        except:
            index = [i for i in range(1, len(h)) if h[i].header['TABLE'] == 'Chemical composition'][0]
        grid['teff'] += [h[0].header['TEFF']]; grid['logg'] += [h[0].header['LOGG']]; grid['zscale'] += [np.round(h[0].header['ZSCALE'], 2)]
        grid['alpha'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'O')[0][0]], 2)]
        grid['carbon'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'C')[0][0]], 2)]
        grid['sodium'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'Na')[0][0]], 2)]
        grid['vanadium'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'V')[0][0]], 2)]
        model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}c{:.3f}na{:.3f}v{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['alpha'][-1], grid['carbon'][-1], grid['sodium'][-1], grid['vanadium'][-1]).replace('-0.000', '0.000')
        __model_parameters[model_id] = model
        h.close()

    grid = {axis: np.unique(grid[axis]) for axis in grid}
    f = open(cache, 'wb')
    pickle.dump((grid, __model_parameters), f)
    f.close()
    return grid

def read_grid_model(params, grid):
    """Load a specific model spectrum from the model grid
    
    This version of the function loads models from FITS files in the Palantir format

    In order to handle redshift, the function will trim the wavelength range of the model spectrum
    on both sides to make sure that the resulting wavelength range remains within the model coverage
    at all redshifts between the bounds in `settings['virtual_dof']['redshift']`. The trimmed parts
    of the spectrum are provided as additional model data in `meta` and may be used by the
    preprocessor to apply the redshift correction
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters. A value must be provided for each grid
        axis, keyed by the axis name
    grid   : dict
        Model grid dimensions, previously obtained with `read_grid_dimensions()`
    
    Returns
    -------
    wl : array_like
        Grid of model wavelengths in A
    flux : array_like
        Corresponding flux densities
    meta : dict
        Dictionary with two keys. 'left' stores the part of the spectrum trimmed on the blue end, and
        'right' stores the part of the spectrum trimmed on the red end. Each value is a list with two
        elements: wavelengths and fluxes
    """
    global __model_parameters

    try:
        __model_parameters
    except:
        read_grid_dimensions()

    def read_FITS_model(params):
        model_id = 't{teff:.3f}l{logg:.3f}z{zscale:.3f}a{alpha:.3f}c{carbon:.3f}na{sodium:.3f}v{vanadium:.3f}'.format(**params).replace('-0.000', '0.000')
        if model_id not in __model_parameters:
            raise FileNotFoundError('Cannot locate model {}'.format(params))
        h = fits.open(__model_parameters[model_id])
        wl = h[1].data['Wavelength']
        flux = h[1].data['Flux density']
        h.close()
        return wl, flux

    try:
        wl, flux = read_FITS_model(params)
    except FileNotFoundError:
        grid = read_grid_dimensions()

        # Find nearest neighbours along each axis in each of the two directions
        neighbors = {}; neighbor_distances = {}; neighbor_positions = {}
        for axis in params:
            neighbors[axis] = [None, None]
            neighbor_distances[axis] = [np.nan, np.nan]
            neighbor_positions[axis] = [np.nan, np.nan]
            for direction in [-1, 1]:
                pos = (init_pos := np.where(np.sort(grid[axis]) == params[axis])[0][0]) + direction
                while pos >= 0 and pos < len(grid[axis]):
                    neighbor_params = copy.copy(params)
                    neighbor_params[axis] = np.sort(grid[axis])[pos]
                    try:
                        neighbors[axis][int(direction > 0)] = read_FITS_model(neighbor_params)
                        neighbor_positions[axis][int(direction > 0)] = pos
                        neighbor_distances[axis][int(direction > 0)] = pos - init_pos
                        break
                    except FileNotFoundError:
                        pos += direction
                        continue

        # If possible, interpolate over the shortest interval
        intervals = {axis: neighbor_positions[axis][1] - neighbor_positions[axis][0] for axis in params}
        if not np.all(np.isnan(list(intervals.values()))):
            shortest = min(intervals, key = lambda k: intervals[k] if not np.isnan(intervals[k]) else np.inf)
            x = [np.sort(grid[shortest])[neighbor_positions[shortest][0]], np.sort(grid[shortest])[neighbor_positions[shortest][1]]]
            warn('Cannot load model {}. Will interpolate from {}={},{}'.format(params, shortest, *x))
            wl, flux = tuple(scp.interpolate.interp1d(x, [neighbors[shortest][0][i], neighbors[shortest][1][i]], axis = 0)(params[shortest]) for i in range(len(neighbors[shortest][0])))
        else:
            # Otherwise, extrapolate from the nearest neighbor
            intervals = {axis: np.nanmin(neighbor_distances[axis]) for axis in params if (not np.all(np.isnan(neighbor_distances[axis])))}
            if len(intervals) == 0: # None of the axes have viable models to interpolate or extrapolate
                raise ValueError('Unable to safely load {}: no models on the same gridlines found'.format(params))
            shortest = min(intervals, key = intervals.get)
            direction = np.where(~np.isnan(neighbor_distances[shortest]))[0][0]
            warn('Cannot load model {}. Will load {}={} instead'.format(params, shortest, grid[shortest][neighbor_positions[shortest][direction]]))
            wl, flux = neighbors[shortest][direction]

    # Trim the spectrum on both sides to make sure we can do redshift corrections
    wl_range = [np.min(wl * (1 + settings['virtual_dof']['redshift'][1] * 1e3 / scp.constants.c)), np.max(wl * (1 + settings['virtual_dof']['redshift'][0] * 1e3 / scp.constants.c))]
    mask_left = wl < wl_range[0]; mask_right = wl > wl_range[1]; mask_in = (~mask_left) & (~mask_right)
    meta = {'left': [wl[mask_left], flux[mask_left]], 'right': [wl[mask_right], flux[mask_right]]}
    return wl[mask_in], flux[mask_in], meta

def preprocess_grid_model(wl, flux, params, meta):
    """Apply redshift correction and quickblur to a loaded model

    quickblur is a virtual degree of freedom that applies a quick Gaussian blur
    to the model spectrum with a given kernel FWHM. This parameter is helpful
    when the exact line spread function of the observed spectrum is not known
    
    Parameters
    ----------
    wl : array_like
        Grid of model wavelengths in A (trimmed to accommodate all redshifts)
    flux : array_like
        Corresponding flux densities
    params : dict
        Parameters of the model, including desired redshift and quickblur FWHM
    meta : dict
        Trimmed parts of the spectrum, as detailed in `read_grid_model()`
    
    Returns
    -------
    array_like
        Redshifted flux with quickblur applied
    """
    # Restore the full (untrimmed) spectrum
    wl_full = np.concatenate([meta['left'][0], wl, meta['right'][0]])
    flux_full = np.concatenate([meta['left'][1], flux, meta['right'][1]])

    # Apply the redshift
    wl_redshifted = wl_full * (1 + params['redshift'] * 1e3 / scp.constants.c)

    # Apply quickblur
    if params['quickblur'] != 0:
        sigma = params['quickblur'] / (2 * np.sqrt(2 * np.log(2)))
        intervals = np.linspace(wl_redshifted[0], wl_redshifted[-1], 100)
        for interval_left, interval_right in zip(intervals[:-1], intervals[1:]):
            interval = (wl_redshifted >= interval_left) & (wl_redshifted <= interval_right)
            flux_full[interval] = scp.ndimage.gaussian_filter1d(flux_full[interval], sigma / (interval_right - interval_left) * len(wl_redshifted[interval]))

    # Re-interpolate back into the original wavelength grid
    flux = np.interp(wl, wl_redshifted, flux_full)
    return flux

