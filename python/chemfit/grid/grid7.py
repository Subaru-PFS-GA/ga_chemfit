############################################################
#                                                          #
#                   Grid7/GridIE PRESET                    #
#                                                          #
#   This preset allows chemfit to use the Grid7/GridIE     #
#   suite of synthetic spectra                             #
#                                                          #
############################################################

import os, pickle, gzip
import numpy as np
import scipy as scp
import re

from .grid import Grid

class Grid7(Grid):
    def __init__(self):
        super().__init__()

    def get_default_settings(self, original_settings = {}):
        return {
            ### Model grid settings ###
            'griddir': None,    # Model directory must be specified in local settings

            ### Which parameters to fit? ###
            'fit_dof': ['zscale', 'alpha', 'teff', 'logg'],
        }

    def _read_grid_dimensions_impl(self):
        """Determine the available dimensions in the model grid and the grid points
        available in those dimensions
        
        This version of the function interfaces with Grid7/GridIE.
        `settings['griddir']` must be pointed to the directories that contains the
        "grid7/bin" and "gridie/bin" subdirectories

        Returns
        -------
        dict
            Dictionary of lists, keyed be grid axis names. The lists contain unique
            values along the corresponding axis that are available in the grid
        """

        # Grid7 only has four dimensions: Teff, log(g), [M/H] and [alpha/M]
        grid = {'teff': [], 'logg': [], 'zscale': [], 'alpha': []}

        # Recursively collect and parse the filenames of all *.bin.gz models
        for root, subdirs, files in os.walk(self.settings['griddir']):
            for file in files:
                if file[-7:].lower() == '.bin.gz':
                    breakdown = list(re.findall('t([0-9]{4})g([_-][0-9]{2})f([_-][0-9]{2})a([_-][0-9]{2})\.', file.replace('a_00a', 'a'))[0])
                    breakdown[0] = int(breakdown[0])
                    for i in range(1, 4):
                        breakdown[i] = np.round(float(breakdown[i].replace('_', '')) / 10.0, 1)
                    grid['teff'] += [breakdown[0]]
                    grid['logg'] += [breakdown[1]]
                    grid['zscale'] += [breakdown[2]]
                    grid['alpha'] += [breakdown[3]]

        grid = {axis: np.unique(grid[axis]) for axis in grid}
        
        return grid, None

    def _read_grid_model_impl(self, params):
        """Load a specific model spectrum from the model grid
        
        This version of the function interfaces with Grid7/GridIE.
        `settings['griddir']` must be pointed to the directories that contains the
        "grid7/bin" and "gridie/bin" subdirectories
        
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
        wl = np.array([], dtype = float)
        flux = np.array([], dtype = float)

        # Load the blue synthetic spectrum, then the red one
        for blue_or_red in [True, False]:

            # Determine the path to the file containing the spectrum
            subgrid_dir = ['grid7', 'gridie'][int(blue_or_red)]
            params_formatted = {}
            for param in ['logg', 'zscale', 'alpha']:
                params_formatted[param] = ['_', '-'][int(params[param] < 0)] + '{:02d}'.format(int(np.round(np.abs(params[param]) * 10)))
            params_formatted['teff'] = int(params['teff'])
            filename = 't{teff}g{logg}f{zscale}a{alpha}.bin.gz'.format(**params_formatted)
            cards = [self.settings['griddir'], subgrid_dir, 'bin', params_formatted['teff'], params_formatted['logg'], filename]
            if not os.path.isfile(path := (template := '{}/{}/{}/t{}/g{}/{}').format(*cards)):
                cards[-1] = cards[-1].replace('a', 'a_00a')
                if not os.path.isfile(path := template.format(*cards)):
                    raise FileNotFoundError('Cannot locate {}'.format(path))

            # Load flux from the binary file
            with gzip.open(path, 'rb') as f:
                file_flux = 1.0 - np.frombuffer(f.read(), dtype = np.float32)

            # Generate the corresponding wavelengths based on some hard-coded parameters
            step = 0.14
            if blue_or_red:
                start = 4100
                stop = 6300
            else:
                start = 6300
                stop = 9100
            file_wl = np.arange(start, stop + step, step)
            file_wl = file_wl[file_wl <= stop + step * 0.1]
            if len(file_wl) != len(file_flux):
                raise ValueError('Unexpected number of points in {}'.format(path))

            wl = np.concatenate([wl, file_wl])
            flux = np.concatenate([flux, file_flux])

        if not np.all(wl[1:] > wl[:-1]):
            raise ValueError('Model wavelengths out of order')

        # Some of GridIE models have negative fluxes and spurious spikes in the flux which may even be infinite
        if np.max(flux) > 100 or np.min(flux) < 0:
            flux[(flux > 100) | (flux < 0.0)] = 1.0
            warn('Unphysical flux values in model {} replaced with unity'.format(params))

        # Since Grid7/GridIE models do not have continua, we attach Planck's law blackbody continua to them as a temporary measure
        bb = 2 * scp.constants.h * scp.constants.c ** 2.0 / (wl * 1e-10) ** 5 * (np.exp(scp.constants.h * scp.constants.c / ((wl * 1e-10) * scp.constants.k * params['teff'])) - 1) ** -1

        return wl, flux * bb
