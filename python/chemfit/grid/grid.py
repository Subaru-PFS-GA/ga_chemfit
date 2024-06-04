import os
import pickle

class Grid():
    def __init__(self):
        self._model_parameters = None

    def get_default_settings(self, original_settings = {}):
        raise NotImplementedError()

    def read_grid_model(self, params):
        """Load a specific model spectrum from the model grid
        
        This function definition is a template. The actual implementation is deferred to the
        settings preset files
        
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
            Corresponding flux densities in wavelength space in arbitrary units
        """
        
        if self._model_parameters is None:
            self.read_grid_dimensions()

        return self._read_grid_model_impl(params)
    
    def _read_grid_model_impl(self, params):
        """Load a specific model spectrum from the model grid
        
        This function definition is a template. The actual implementation is deferred to the
        settings preset files
        
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
            Corresponding flux densities in wavelength space in arbitrary units
        """

        raise NotImplementedError()

    def read_grid_dimensions(self, flush_cache = False):
        """Determine the available dimensions in the model grid and the grid points
        available in those dimensions
        
        This function definition is a template. The actual implementation is deferred to the
        settings preset files
        
        Parameters
        ----------
        flush_cache : bool, optional
            If True, discard cache and scan the grid afresh
        
        Returns
        -------
        dict
            Dictionary of lists, keyed be grid axis names. The lists contain unique
            values along the corresponding axis that are available in the grid
        """
        
        # Apply caching
        cache = (self.settings['griddir'] + '/cache.pkl')
        if os.path.isfile(cache) and (not flush_cache):
            with open(cache, 'rb') as f:
                grid, self._model_parameters = pickle.load(f)
        else:
            grid, self._model_parameters =  self._read_grid_dimensions_impl()

            with open(cache, 'wb') as f:
                pickle.dump((grid, self._model_parameters), f)

        return grid
        
    def _read_grid_dimensions_impl(self):
        """Determine the available dimensions in the model grid and the grid points
        available in those dimensions
        
        This function definition is a template. The actual implementation is deferred to the
        settings preset files
                
        Returns
        -------
        dict
            Dictionary of lists, keyed be grid axis names. The lists contain unique
            values along the corresponding axis that are available in the grid
        """

        raise NotImplementedError()