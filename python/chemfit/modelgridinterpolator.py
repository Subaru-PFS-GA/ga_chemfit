class ModelGridInterpolator():
    """Handler class for interpolating the model grid to arbitrary stellar parameters
    
    The class provides the `interpolate()` method to carry out the interpolation. The
    interpolation is linear with dynamic fetching of models from disk and caching for
    loaded models
    
    Attributes
    ----------
    statistics : array_like
        Statistical data to track the interpolator's performance. Includes the following
        keys:
            'num_models_used': Total number of models read from disk *if* no caching was
                            used
            'num_models_loaded': Actual number of models read from disk
            'num_interpolations': Total number of interpolator calls
            'num_interpolators_built': Total number of interpolator objects constructed
                                    (scipy.interpolate.RegularGridInterpolator)
    """
    def __init__(self, settings, resample = True, detector_wl = None, synphot_bands = [], mag_system = None, reddening = None, max_models = 10000):
        """
        Parameters
        ----------
        resample : bool, optional
            If True, resample the models to the detector bins before interpolation. The
            resampling is carried out with `simulate_observation()`. Otherwise,
            interpolate without resampling over the original wavelength grid of the models
        detector_wl : None or list or dict, optional
            Reference wavelengths of the detector bins in each arm of the spectrograph. The
            argument is only relevant if `resample` is set to True. If `detector_wl` is
            a dictionary, it must be keyed by the identifiers of the arms that
            are used in this observation. The values are 1D arrays of reference wavelengths of
            detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
            identifiers and the "typical" wavelength sampling for each arm (as defined in
            `settings['arms']`) will be assumed. Alternatively, set to `None` to use all
            arms defined in `settings['arms']` with "typical" wavelength sampling
        synphot_bands : list, optional
            If synthetic photometry for the interpolated models is required, provide the desired
            colors as elements of this list. Each color must be a 2-element tuple with the
            filenames of the bluer and redder filters in the `settings['filter_dir']` directory
        mag_system : str, optional
            Magnitude system to use for synthetic photometry. Must be supported by `synphot()`
        reddening : float, optional
            Optical reddening parameter (E(B-V)) to use for synthetic photometry
        max_models : int, optional
            Maximum number of models to keep in the loaded models cache. If exceeded, the
            models loaded earliest will be removed from the cache. Higher numbers lead to
            less frequent disk access (and hence faster performance) but higher memory usage
        """

        mag_system = mag_system if mag_system is not None else settings['default_mag_system']
        reddening = reddening if reddening is not None else settings['default_reddening']

        # Populate detector_wl if not given
        if resample:
            if detector_wl is None:
                detector_wl = {arm: settings['arms'][arm]['wl'] for arm in settings['arms']}
            elif type(detector_wl) is list:
                detector_wl = {arm: settings['arms'][arm]['wl'] for arm in detector_wl}
            setattr(self, '_detector_wl', detector_wl)
        setattr(self, '_resample', resample)

        # Load grid dimensions
        setattr(self, '_grid', read_grid_dimensions())

        # Loaded models
        setattr(self, '_loaded', {})
        setattr(self, '_loaded_ordered', [])
        setattr(self, '_max_models', max_models)

        # Models embedded into the current interpolator
        setattr(self, '_interpolator_models', set())

        # Synthetic photometry parameters
        setattr(self, '_synphot_bands', synphot_bands)
        setattr(self, '_mag_system', mag_system)
        setattr(self, '_reddening', reddening)

        # Holders of statistical information
        setattr(self, 'statistics', {'num_models_used': 0, 'num_models_loaded': 0, 'num_interpolations': 0, 'num_interpolators_built': 0})

    def _build_interpolator(self, x):
        # Make sure x has the right dimensions
        if set(x.keys()) != set(self._grid.keys()):
            raise ValueError('Model grid dimensions and requested interpolation target dimensions do not match')

        # Which models are required to interpolate to x?
        subgrid = {}
        for key in self._grid.keys():
            if (x[key] > np.max(self._grid[key])) or (x[key] < np.min(self._grid[key])):
                raise ValueError('Model grid dimensions exceeded along {} axis'.format(key))
            if x[key] in self._grid[key]:
                subgrid[key] = np.array([x[key]])
            else:
                subgrid[key] = np.array([np.max(self._grid[key][self._grid[key] < x[key]]), np.min(self._grid[key][self._grid[key] > x[key]])])
        required_models = set(['|'.join(np.array(model).astype(str)) for model in itertools.product(*[subgrid[key] for key in sorted(list(subgrid.keys()))])])
        self.statistics['num_models_used'] += len(required_models)

        # If the current interpolator is already based on these models, just return it
        if required_models == self._interpolator_models:
            return self._interpolator

        # Determine which of the required models have not been loaded yet
        new_models = [model for model in required_models if model not in self._loaded]

        # If the total number of models exceeds max_models, delete the ones loaded earlier
        if len(new_models) + len(self._loaded) > self._max_models:
            to_delete = len(new_models) + len(self._loaded) - self._max_models
            remaining = []
            for model in self._loaded_ordered:
                if model in required_models or to_delete == 0:
                    remaining += [model]
                else:
                    del self._loaded[model]
                    to_delete -= 1
            self._loaded_ordered = remaining

        # Load the new models
        for model in new_models:
            params = np.array(model.split('|')).astype(float)
            keys = sorted(list(x.keys()))
            params = {keys[i]: params[i] for i in range(len(keys))}
            wl, flux = safe_read_grid_model(params, self._grid)

            # Carry out synthetic photometry
            colors = np.zeros(len(self._synphot_bands))
            if len(self._synphot_bands) != 0:
                if 'teff' not in params:
                    raise ValueError('The model grid must have "teff" as one of the axes to compute synthetic photometry')
                phot = synphot(wl, flux, params['teff'], np.unique(self._synphot_bands), mag_system = self._mag_system, reddening = self._reddening)
                for i, color in enumerate(self._synphot_bands):
                    if np.isnan(phot[color[0]]) or np.isnan(phot[color[1]]):
                        raise ValueError('Could not calculate synthetic color {} for model {}'.format(color, params))
                    # Note that the order of bands is reversed in the color calculation, since synphot() returns bolometric corrections, not magnitudes
                    colors[i] = phot[color[1]] - phot[color[0]]

            if self._resample:
                wl, flux = simulate_observation(wl, flux, self._detector_wl)
            try:
                self._wl
            except:
                setattr(self, '_wl', wl)
            self._loaded[model] = np.concatenate([colors, flux]) # Prepend photometry to fluxes, so they are interpolated at the same time
            self._loaded_ordered += [model]
        self.statistics['num_models_loaded'] += len(new_models)

        # Build the interpolator
        subgrid_ordered = [subgrid[key] for key in sorted(list(subgrid.keys()))]
        meshgrid = np.meshgrid(*subgrid_ordered, indexing = 'ij')
        spectra = np.vectorize(lambda *x: self._loaded['|'.join(np.array(x).astype(str))], signature = ','.join(['()'] * len(self._grid)) + '->(n)')(*meshgrid)
        setattr(self, '_interpolator', scp.interpolate.RegularGridInterpolator(subgrid_ordered, spectra))
        self._interpolator_models = required_models
        self.statistics['num_interpolators_built'] += 1
        return self._interpolator

    def interpolate(self, params):
        """Carry out model grid interpolation
        
        Parameters
        ----------
        params : dict
            Dictionary of target stellar parameters. A value must be provided for each grid
            axis, keyed by the axis name
        
        Returns
        -------
        wl : array_like
            Grid of model wavelengths in A
        flux : array_like
            Interpolated flux densities in wavelength space in arbitrary units corresponding to the
            wavelengths in `wl`
        phot : array_like
            If requested (i.e. if `synphot_bands` is not empty), synthetic colors for the bands
            listed in `synphot_bands` with the desired reddening in the desired magnitude system
        """
        self.statistics['num_interpolations'] += 1
        interpolator = self._build_interpolator(params)
        result = interpolator([params[key] for key in sorted(list(params.keys()))])[0]
        if len(self._synphot_bands) != 0:
            return self._wl, result[len(self._synphot_bands):], result[:len(self._synphot_bands)]
        else:
            return self._wl, result

    def __call__(self, params):
        return self.interpolate(params)