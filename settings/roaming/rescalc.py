############################################################
#                                                          #
#                     rescalc PRESET                       #
#                                                          #
#   This preset allows chemfit to generate synthetic       #
#   spectra including the effect of individual element     #
#   variations using element response functions computed   #
#   with the rescalc utility                               #
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

    ### Default initial guesses ###
    'default_initial': {'redshift': 0.0, 'quickblur': 0.0}
}

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions
    
    This version of the function loads models from pickle files generated with rescalc.
    `settings['griddir']` must be pointed to the directories that contains the
    pickle files either in the top directory or subdirectories. Recursive search for
    *.pkl files will be carried out at first call

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
    global __available_elements

    # Apply caching
    if os.path.isfile(cache := (settings['griddir'] + '/cache.pkl')) and (not flush_cache):
        f = open(cache, 'rb')
        grid, __model_parameters, __available_elements = pickle.load(f)
        f.close()
        return grid

    grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': []}
    __model_parameters = {}
    __available_elements = {}

    # Extract the parameters of all available models
    models = glob.glob(settings['griddir'] + '/*.pkl') + glob.glob(settings['griddir'] + '/**/*.pkl', recursive = True)
    models = [model for model in models if os.path.basename(model) != 'cache.pkl']
    for i, model in enumerate(models):
        f = open(model, 'rb')
        model = pickle.load(f)
        f.close()
        grid['teff'] += [np.round(model['meta']['teff'], 3)]
        grid['logg'] += [np.round(model['meta']['logg'], 3)]
        grid['zscale'] += [np.round(model['meta']['zscale'], 3)]
        if 'Mg' in model['meta']['abun']:
            grid['alpha'] += [np.round(model['meta']['abun']['Mg'], 3)]
        else:
            grid['alpha'] += [0.0]
        model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['alpha'][-1])
        __model_parameters[model_id] = models[i]

        for element in model['response']:
            element_range = [np.min(model['response'][element]['abun']), np.max(model['response'][element]['abun'])]
            if element not in __available_elements:
                __available_elements[element] = element_range
            else:
                assert __available_elements[element] == element_range

    grid = {axis: np.unique(grid[axis]) for axis in grid}
    f = open(cache, 'wb')
    pickle.dump((grid, __model_parameters, __available_elements), f)
    f.close()
    return grid

# Add available element responses to settings
if settings['griddir'] is None:
    raise ValueError('settings[\'griddir\'] must be defined in settings/local/rescalc.py')
read_grid_dimensions()
for element in __available_elements:
    settings['fit_dof'] += ['response_{}'.format(element)]
    settings['virtual_dof']['response_{}'.format(element)] = __available_elements[element]
    settings['default_initial']['response_{}'.format(element)] = np.mean(__available_elements[element])


def read_grid_model(params, grid):
    """Load a specific model spectrum from the model grid with element response functions
    
    This version of the function loads models produced by the rescalc utility

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
        Dictionary with trimmed parts of the spectrum for redshift calculations, continuum component
        of the model flux and element response functions
    """
    global __model_parameters

    try:
        __model_parameters
    except:
        read_grid_dimensions()

    model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(params['teff'], params['logg'], params['zscale'], params['alpha'])
    if model_id not in __model_parameters:
        raise FileNotFoundError('Cannot locate model {}'.format(params))

    f = open(__model_parameters[model_id], 'rb')
    model = pickle.load(f)
    f.close()
    wl, flux = model['null']['wl'], model['null']['line']

    # Prepare continuum flux
    cont = np.zeros(len(model['null']['flux']))
    sat = model['null']['line'] == 0
    cont[~sat] = model['null']['flux'][~sat] / model['null']['line'][~sat]
    cont[sat] = np.interp(wl[sat], wl[~sat], cont[~sat])

    # Trim the spectrum on both sides to make sure we can do redshift corrections
    wl_range = [np.min(wl * (1 + settings['virtual_dof']['redshift'][1] * 1e3 / scp.constants.c)), np.max(wl * (1 + settings['virtual_dof']['redshift'][0] * 1e3 / scp.constants.c))]
    mask_left = wl < wl_range[0]; mask_right = wl > wl_range[1]; mask_in = (~mask_left) & (~mask_right)
    meta = {'left': [wl[mask_left], flux[mask_left]], 'right': [wl[mask_right], flux[mask_right]], 'response': model['response'], 'cont': cont}

    return wl[mask_in], flux[mask_in], meta

def preprocess_grid_model(wl, flux, params, meta):
    """Apply redshift correction, quickblur and element responses to a loaded model

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
        Parameters of the model, including desired redshift, quickblur FWHM and
        element abundances
    meta : dict
        Trimmed parts of the spectrum, continuum spectrum and element response functions
    
    Returns
    -------
    array_like
        Redshifted flux with quickblur and element responses applied
    """
    # Restore the full (untrimmed) spectrum
    wl_full = np.concatenate([meta['left'][0], wl, meta['right'][0]])
    flux_full = np.concatenate([meta['left'][1], flux, meta['right'][1]])

    # Apply response functions
    for param in params:
        if param[:9] == 'response_':
            element = param[9:]
            flux_full[meta['response'][element]['mask']] += scp.interpolate.interp1d(meta['response'][element]['abun'], meta['response'][element]['spectra'])(params[param]) - meta['response'][element]['spectra'][:,0]

    # Add continuum
    flux_full *= meta['cont']

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

