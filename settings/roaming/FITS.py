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

settings = {
    ### Model grid settings ###
    'griddir': None,    # Model directory must be specified in local settings

    ### Which parameters to fit? ###
    'fit_dof': ['zscale', 'alpha', 'teff', 'logg'],
}

def read_grid_dimensions(flush_cache = False):
    """Determine the available dimensions in the model grid and the grid points
    available in those dimensions

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

    grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': []}
    __model_parameters = {}

    # Extract the parameters of all available models
    models = glob.glob(settings['griddir'] + '/*.fits')
    for model in models:
        h = fits.open(model)
        try:
            index
        except:
            index = [i for i in range(1, len(h)) if h[i].header['TABLE'] == 'Chemical composition'][0]
        grid['teff'] += [h[0].header['TEFF']]; grid['logg'] += [h[0].header['LOGG']]; grid['zscale'] += [np.round(h[0].header['ZSCALE'], 2)]
        grid['alpha'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'Ti')[0][0]], 2)]
        model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['alpha'][-1])
        __model_parameters[model_id] = model
        h.close()

    grid = {axis: np.unique(grid[axis]) for axis in grid}
    f = open(cache, 'wb')
    pickle.dump((grid, __model_parameters), f)
    f.close()
    return grid

def read_grid_model(params):
    """Load a specific model spectrum from the model grid
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters. A value must be provided for each grid
        axis, keyed by the axis name
    
    Returns
    -------
    wl : array_like
        Grid of model wavelengths in A
    flux : array_like
        Corresponding flux densities
    """
    global __model_parameters

    try:
        __model_parameters
    except:
        read_grid_dimensions()

    model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(params['teff'], params['logg'], params['zscale'], params['alpha'])
    if model_id not in __model_parameters:
        raise FileNotFoundError('Cannot locate model {}'.format(params))

    h = fits.open(__model_parameters[model_id])
    wl = h[1].data['Wavelength']
    flux = h[1].data['Total flux density']
    h.close()

    from PyAstronomy import pyasl
    return pyasl.vactoair2(wl), flux
