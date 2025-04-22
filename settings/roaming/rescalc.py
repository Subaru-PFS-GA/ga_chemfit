############################################################
#                                                          #
#                     RESCALC PRESET                       #
#                                                          #
#   This preset allows chemfit to generate synthetic       #
#   spectra including the effect of individual element     #
#   variations using element response functions computed   #
#   with the RESCALC utility                               #
#                                                          #
############################################################

import os, pickle
import glob
from astropy.io import fits
import scipy as scp
import zipfile


# This is the "defective" mask that removes parts of the spectrum where the models do not match the spectra of 
# Arcturus and the Sun well
defective_mask = [[3800, 4000], [4006, 4012], [4065, 4075], [4093, 4110], [4140, 4165], [4170, 4180], [4205, 4220],
                 [4285, 4300], [4335, 4345], [4375, 4387], [4700, 4715], [4775, 4790], [4855, 4865], [5055, 5065],
                 [5145, 5160], [5203, 5213], [5885, 5900], [6355, 6365], [6555, 6570], [7175, 7195], [7890, 7900],
                 [8320, 8330], [8490, 8505], [8530, 8555], [8650, 8672]]

settings = {
    ### Model grid settings ###
    'griddir': original_settings['griddir'],    # Model directory must be specified in local settings

    ### Reduce size of model cache since the models now have response functions ###
    'max_model_cache': 32,

    ### Which parameters to fit? The abundances of individual elements do not need to be listed here ###
    ### as they will be added automatically based on the available models ###
    'fit_dof': ['zscale', 'alpha', 'teff', 'logg', 'carbon', 'redshift'],

    ### Virtual grid bounds. As before, abundances of individual elements are added automatically ###
    'virtual_dof': {'redshift': [-300, 300], 'redshift_correction': [-100, 100]},

    ### Default initial guesses. As before, abundances of individual elements are added automatically ###
    'default_initial': {'redshift': 0.0, 'redshift_correction': 0.0},

    ### Fitting masks ###
    'masks': apply_standard_mask(defective_mask, original_settings['masks']),
}

if 'model_index_file' in original_settings:
    settings['model_index_file'] = original_settings['model_index_file']

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions
    
    This version of the function loads models from pickle files generated with rescalc.
    `settings['griddir']` must be pointed to the directories that contains the
    pickle files either in the top directory or subdirectories. Recursive search for
    *.pkl files will be carried out at first call

    The function implements file caching. If an index file is provided in the settings
    (`settings['model_index_file']`), it will be used as the cache file, and it will
    be required to exist, such that the models are never scanned
    
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

    if 'model_index_file' not in settings:
        cache = settings['griddir'] + '/cache.pkl'
    else:
        cache = settings['model_index_file']

    # Apply caching
    if ('model_index_file' in settings) or (os.path.isfile(cache) and (not flush_cache)):
        f = open(cache, 'rb')
        grid, __model_parameters, __available_elements = pickle.load(f)
        f.close()
        return grid

    grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': [], 'carbon': []}
    __model_parameters = {}
    __available_elements = {}

    # Extract the parameters of all available models
    models = np.unique(glob.glob(settings['griddir'] + '/*.pkl') + glob.glob(settings['griddir'] + '/**/*.pkl', recursive = True))
    for i, model in enumerate(models):
        f = open(model, 'rb')
        model = pickle.load(f)
        f.close()
        grid['teff'] += [np.round(model['meta']['teff'] + 0.0, 3)]
        grid['logg'] += [np.round(model['meta']['logg'] + 0.0, 3)]
        grid['zscale'] += [np.round(model['meta']['zscale'] + 0.0, 3)]
        if 'Mg' in model['meta']['abun']:
            grid['alpha'] += [np.round(model['meta']['abun']['Mg'] + 0.0, 3)]
        else:
            grid['alpha'] += [0.0]
        if 'C' in model['meta']['abun']:
            grid['carbon'] += [np.round(model['meta']['abun']['C'] + 0.0, 3)]
        else:
            grid['carbon'] += [0.0]
        model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}c{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['alpha'][-1], grid['carbon'][-1])
        if model_id in __model_parameters:
            raise ValueError('Two models with identical parameters ({}) found: {} and {}'.format(model_id, __model_parameters[model_id], models[i]))
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

    model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}c{:.3f}'.format(params['teff'] + 0.0, params['logg'] + 0.0, params['zscale'] + 0.0, params['alpha'] + 0.0, params['carbon'] + 0.0)
    if model_id not in __model_parameters:
        raise FileNotFoundError('Cannot locate model {}'.format(params))

    if __model_parameters[model_id][0] != '/':
        model_path = settings['griddir'] + '/' + __model_parameters[model_id]
    else:
        model_path = __model_parameters[model_id]
    if (index := model_path.find('.zip/')) == -1:
        f = open(model_path, 'rb')
        model = pickle.load(f)
        f.close()
    else:
        z = zipfile.ZipFile(model_path[:index + 4], 'r')
        f = z.open(model_path[index + 5:])
        model = pickle.load(f)
        f.close()

    wl = np.exp(np.arange(np.ceil(np.log(model['meta']['wl_start']) / (lgr := np.log(1.0 + 1.0 / model['meta']['res']))), np.floor(np.log(model['meta']['wl_end']) / lgr) + 1) * lgr) * 10
    if 'binning' in model['meta'] and model['meta']['binning'] != 1:
        wl = wl[:(len(wl) // model['meta']['binning']) * model['meta']['binning']].reshape(-1, model['meta']['binning']).mean(axis = 1)
    cont, flux = model['null']['cont'], model['null']['line']
    wl = wl; cont = cont; flux = flux
    assert len(wl) == len(flux)

    # Convert response functions into SciPy interpolation objects
    for element in model['response']:
        assert model['response'][element]['abun'][0] == 0
        model['response'][element]['spectra'] = scp.interpolate.interp1d(model['response'][element]['abun'], model['response'][element]['spectra'] - model['response'][element]['spectra'][:,0][:,np.newaxis])

    # Trim the spectrum on both sides to make sure we can do redshift corrections
    shift_wl = lambda wl, redshift, redshift_correction: wl * (1 + (redshift + (wl - 10000) / 5000 * redshift_correction) * 1e3 / scp.constants.c)
    wl_range_min = []; wl_range_max = []
    for i in range(2):
        for j in range(2):
            wl_range_min += [shift_wl(np.min(wl), settings['virtual_dof']['redshift'][i], settings['virtual_dof']['redshift_correction'][j])]
            wl_range_max += [shift_wl(np.max(wl), settings['virtual_dof']['redshift'][i], settings['virtual_dof']['redshift_correction'][j])]
    wl_range = [np.max(wl_range_min), np.min(wl_range_max)]
    mask_left = wl < wl_range[0]; mask_right = wl > wl_range[1]; mask_in = (~mask_left) & (~mask_right)
    meta = {'left': [wl[mask_left], flux[mask_left]], 'right': [wl[mask_right], flux[mask_right]], 'response': model['response'], 'cont': cont}

    return wl[mask_in], flux[mask_in], meta

def preprocess_grid_model(wl, flux, params, meta):
    """Apply redshift correction and element responses to a loaded model

    Parameters
    ----------
    wl : array_like
        Grid of model wavelengths in A (trimmed to accommodate all redshifts)
    flux : array_like
        Corresponding flux densities
    params : dict
        Parameters of the model, including desired redshift and element abundances
    meta : dict
        Trimmed parts of the spectrum, continuum spectrum and element response functions
    
    Returns
    -------
    array_like
        Redshifted flux with element responses applied
    """
    # Restore the full (untrimmed) spectrum
    wl_full = np.concatenate([meta['left'][0], wl, meta['right'][0]])
    flux_full = np.concatenate([meta['left'][1], flux, meta['right'][1]])

    # Apply response functions
    for param in params:
        if param[:9] == 'response_':
            element = param[9:]
            flux_full[meta['response'][element]['mask']] += meta['response'][element]['spectra'](params[param])

    # Add continuum
    flux_full *= meta['cont']

    # Apply the redshift
    shift_wl = lambda wl, redshift, redshift_correction: wl * (1 + (redshift + (wl - 10000) / 5000 * redshift_correction) * 1e3 / scp.constants.c)
    wl_redshifted = shift_wl(wl_full, params['redshift'], params['redshift_correction'])

    # Re-interpolate back into the original wavelength grid
    flux = np.interp(wl, wl_redshifted, flux_full)
    return flux

