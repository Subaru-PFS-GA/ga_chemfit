############################################################
#                                                          #
#                        ElfOwl PRESET                     #
#                                                          #
#   This preset allows chemfit to load synthetic spectra   #
#   from ElfOwl models from Mukherjee+2024                 #
#                                                          #
############################################################

import os, pickle
import glob
import xarray
import json
import scipy as scp

settings = {
    ### Model grid settings ###
    'griddir': original_settings['griddir'],    # Model directory must be specified in local settings

    ### Which parameters to fit? ###
    'fit_dof': ['zscale', 'teff', 'logg', 'co', 'kzz'],

    ### Virtual grid dimensions ###
    'virtual_dof': {'redshift': [-200, 200]},
}

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions
    
    This version of the function loads models from NetCDF (*.nc) files used by
    the Sonora ElfOwl model atmospheres.
    `settings['griddir']` must be pointed to the directories that contains the
    *.nc files in the top directory

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

    grid = {'teff': [], 'logg': [], 'zscale': [], 'co': [], 'kzz': []}
    __model_parameters = {}

    # Extract the parameters of all available models
    models = glob.glob(settings['griddir'] + '/*.nc')
    for model in models:
        ds = xarray.load_dataset(model)
        meta = json.loads(ds.planet_params)
        grid['teff'] += [meta['teff']['value']]; assert meta['teff']['unit'] == 'K'
        grid['logg'] += [np.round(np.round(np.log10(meta['logg']['value'] * 100) / 0.25) * 0.25, 3)]; assert meta['logg']['unit'] == 'm / s2'
        grid['zscale'] += [meta['mh']]
        grid['co'] += [meta['cto']]
        grid['kzz'] += [meta['logkzz']]
        assert meta['PH3'] == 'chemeq'
        model_id = 't{:.3f}l{:.3f}z{:.3f}c{:.3f}kzz{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['co'][-1], grid['kzz'][-1]).replace('-0.000', '0.000')
        __model_parameters[model_id] = model

    grid = {axis: np.unique(grid[axis]) for axis in grid}
    f = open(cache, 'wb')
    pickle.dump((grid, __model_parameters), f)
    f.close()
    return grid

def read_grid_model(params, grid):
    """Load a specific model spectrum from the model grid
    
    This version of the function loads models from NetCDF files in the ElfOwl format

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

    def read_ElfOwl_model(params):
        model_id = 't{teff:.3f}l{logg:.3f}z{zscale:.3f}c{co:.3f}kzz{kzz:.3f}'.format(**params).replace('-0.000', '0.000')
        if model_id not in __model_parameters:
            raise FileNotFoundError('Cannot locate model {}'.format(params))
        ds = xarray.load_dataset(__model_parameters[model_id])
        return np.array(ds['wavelength'])[::-1] * 1e4, np.array(ds['flux'])[::-1] / 1e8 / np.pi

    try:
        wl, flux = read_ElfOwl_model(params)
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
                        neighbors[axis][int(direction > 0)] = read_ElfOwl_model(neighbor_params)
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
    """Apply redshift correction to a loaded model
    
    Parameters
    ----------
    wl : array_like
        Grid of model wavelengths in A (trimmed to accommodate all redshifts)
    flux : array_like
        Corresponding flux densities
    params : dict
        Parameters of the model, including desired redshift
    meta : dict
        Trimmed parts of the spectrum, as detailed in `read_grid_model()`
    
    Returns
    -------
    array_like
        Redshifted flux
    """
    # Restore the full (untrimmed) spectrum
    wl_full = np.concatenate([meta['left'][0], wl, meta['right'][0]])
    flux_full = np.concatenate([meta['left'][1], flux, meta['right'][1]])

    # Apply the redshift
    wl_redshifted = wl_full * (1 + params['redshift'] * 1e3 / scp.constants.c)

    # Re-interpolate back into the original wavelength grid
    flux = np.interp(wl, wl_redshifted, flux_full)
    return flux

