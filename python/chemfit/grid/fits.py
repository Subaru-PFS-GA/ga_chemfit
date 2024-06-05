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
import numpy as np

from .grid import Grid

class FITS(Grid):
    def __init__(self):
        super().__init__()

    def get_default_settings(self, original_settings = {}):
        return {
            ### Model grid settings ###
            'griddir': os.path.expandvars('${CHEMFIT_GRIDDIR}'),    # Model directory must be specified in local settings

            ### Which parameters to fit? ###
            'fit_dof': ['zscale', 'alpha', 'teff', 'logg'],
        }

    def _read_grid_dimensions_impl(self):
        """Determine the available dimensions in the model grid and the grid points
        available in those dimensions

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

        grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': []}
        model_parameters = {}

        # Extract the parameters of all available models
        models = glob.glob(self.settings['griddir'] + '/*.fits')
        for model in models:
            h = fits.open(model)
            try:
                index
            except:
                index = [i for i in range(1, len(h)) if h[i].header['TABLE'] == 'Chemical composition'][0]
            grid['teff'] += [h[0].header['TEFF']]; grid['logg'] += [h[0].header['LOGG']]; grid['zscale'] += [np.round(h[0].header['ZSCALE'], 2)]
            grid['alpha'] += [np.round(h[index].data['Relative abundance'][np.where(h[index].data['Element'] == 'Ti')[0][0]], 2)]
            model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(grid['teff'][-1], grid['logg'][-1], grid['zscale'][-1], grid['alpha'][-1])
            model_parameters[model_id] = model
            h.close()

        grid = {axis: np.unique(grid[axis]) for axis in grid}
        
        return grid, model_parameters

    def _read_grid_model_impl(self, params):
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

        model_id = 't{:.3f}l{:.3f}z{:.3f}a{:.3f}'.format(params['teff'], params['logg'], params['zscale'], params['alpha'])
        if model_id not in self._model_parameters:
            raise FileNotFoundError('Cannot locate model {}'.format(params))

        h = fits.open(self._model_parameters[model_id])
        wl = h[1].data['Wavelength']
        flux = h[1].data['Total flux density'] / h[1].data['Continuum flux density']
        h.close()

        return wl, flux
