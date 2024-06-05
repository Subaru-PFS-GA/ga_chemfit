import numpy as np
import scipy as scp
import pickle
import gzip
import os
import re
import hashlib
import itertools
import warnings
import copy
import inspect
import importlib.util

from .modelgridinterpolator import ModelGridInterpolator

# scipy.optimize.curve_fit() throws an exception when the optimizer exhausts the maximum allowed number of function
# evaluations. Since we are demanding fairly narrow tolerance, reaching the maximum number of evaluations is not
# necessarily fatal (it merely means that the actual tolerance is somewhat less than our default standard). So we would
# like to deescalate the error to a warning and proceed. Unfortunately, I am unable to find a more elegant way of doing
# it except by overriding the optimizer function that curve_fit() calls
original_least_squares = scp.optimize._minpack_py.least_squares
def least_squares_wrapper(*args, **kwargs):
    res = original_least_squares(*args, **kwargs)
    if (not res.success) and (res.status == 0):
        warn('Optimizer failed to reach desired convergence after the maximum number ({}) of function evaluations'.format(res.nfev))
        res.success = True
    return res
scp.optimize._minpack_py.least_squares = least_squares_wrapper

class Chemfit():
    def __init__(self, script_dir = None):
        self.script_dir = script_dir if script_dir is not None else os.getcwd()
        self.settings = {}
        self.grid = None

        self.warnings_stack = []
        self.warnings_messages = {}

    def get_default_settings(self, original_settings = {}):
        return {
            ### Synthetic photometry ###
            'filter_dir': self.script_dir + '/bands/',     # Path to the transmission profile directory
            'default_mag_system': 'VEGAMAG',               # Default magnitude system
            'default_reddening': 0.0,                      # Default E(B-V)

            ### Spectrograph settings ###
            'arms': {                                      # Parameters of individual spectrograph arms
                'default_arm': {
                    'FWHM': 2.07,                              # FWHM of the line spread function in the arm in A
                    'wl': np.linspace(3800, 6500, 4096),       # Default bin wavelengths in the arm in A
                    'priority': 1,                             # Arm priority if overlapping with other arms in the wavelength space
                },
            },

            ### Fitting masks ###
            'masks': {
                'all': {
                    'all': [[4500., 5164.322], [5170.322, 5892.924], [5898.924, 8488.023], [8508.023, 8525.091], [8561.091, 8645.141], [8679.141, 9100.]],
                },
                'continuum': [[6864, 6935], [7591, 7694], [8938, 9100]],
            },

            ### Optimization parameters ###
            'return_diagnostics': True, # Return best-fit model, continuum correction and fitting masks in the chemfit.chemfit() output
            'gradient_descent': {
                'curve_fit': {
                    'absolute_sigma': False,
                    'ftol': 1e-10,
                    'gtol': 1e-10,
                    'xtol': 1e-10,
                },
            },
            'mcmc': {
                'nwalkers': 32,
                'nsteps': 5000,
                'discard': 300,
                'initial': 'gradient_descent',
                'progress': True,
            },
            'cont_pix': 165,
            'spline_order': 3,

            ### Warnings ###
            'throw_python_warnings': True,
        }

    def warn(self, message):
        """Issue a warning. Wrapper for `warnings.warn()`
        
        Add the warning message to the stack. If required, also throw the normal Python warning

        Instead of storing warning messages in the stack in full, we associate unique numerical
        identifiers with each distinct message and store them in a dictionary. The stack then
        only contains the identifiers to save memory
        
        Parameters
        ----------
        message : str
            Warning message
        """
        if message not in self.warnings_messages:
            warning_id = len(self.warnings_messages)
            self.warnings_messages[message] = warning_id
        else:
            warning_id = self.warnings_messages[message]

        self.warnings_stack += [warning_id]
        if self.settings['throw_python_warnings']:
            warnings.warn(message)

    def initialize(self, grid, instrument, *presets):
        """Load the settings presets

        Instrument specifications, fitting parameters, model grid handling and other required data and procedures
        are stored in a collection of Python scripts referred to as presets. The scripts may be found in the
        "settings" directory. Each script is expected to define a `settings` dictionary, the entries of which will
        then be used to update the global `settings` dictionary. Optionally, the scripts may also define the
        `read_grid_model()` and `read_grid_dimensions()` functions, following the templates in this file. These
        functions will be used to load model spectra during the fitting process

        Note that the "default" preset will be loaded automatically

        Parameters
        ----------
        presets : tuple
            Settings presets to load. For each preset, the function will attempt to load both
            "settings/roaming/<preset>.py" and "settings/local/<preset>.py" in that order, if available. It is
            recommended to store global parameters in the former file (versioned) and machine-specific parameters in
            the latter file (not versioned, i.e. ignored)
        """

        # Set the grid
        self.grid = grid
        self.instrument = instrument

        # Reset settings and get defaults
        settings = {}

        modules = [ self, grid, instrument ]

        # Get the default settings from each module
        for module in [ self, grid, instrument ]:
            if module is not None:
                s = module.get_default_settings(original_settings = copy.deepcopy(settings))
                settings.update(s)

        settings = self._initialize_from_presets(settings, *presets)

        # Update each module with the new settings
        for module in [ self, grid, instrument ]:
            if module is not None:
                module.settings = settings


    def _initialize_from_presets(self, settings, *presets):
        # Environment variables provided to the preset scripts
        env = {'script_dir': self.script_dir,
               'np': np,
               'original_settings': copy.deepcopy(settings),
               'copy': copy,
               'warn': lambda message: self.warn(message)}

        index = 0 # Load counter to ensure unique module names for all loaded files
        for preset in list(presets):
            scripts = [
                os.path.join(self.script_dir, 'settings', 'roaming', f'{preset}.py'),
                os.path.join(self.script_dir, 'settings', 'local', f'{preset}.py'),
            ]
            scripts = [script for script in scripts if os.path.isfile(script)]
            if len(scripts) == 0:
                raise ValueError('Settings preset {} not found'.format(preset))
            for script in scripts:
                # Load and run the file
                index += 1
                spec = importlib.util.spec_from_file_location('settings_{}'.format(index), script)
                module = importlib.util.module_from_spec(spec)
                module.__dict__.update(env)
                spec.loader.exec_module(module)

                # Update the global settings variable and make it available to the module in case it defines model read
                # functions and they need it
                settings.update(module.settings)
                env['original_settings'] = copy.deepcopy(settings)
                module.settings = settings

        return settings

    def safe_read_grid_model(self, params):
        """Wrapper for `read_grid_model()` that deescalates missing model errors to warnings, and
        replaces the output with an interpolated result from adjacent models

        When a requested model should be present in the model grid, but cannot be loaded from disk,
        `read_grid_model()` throws `FileNotFoundError`. This function, instead, resolves the error
        using the following algorithm:
            - Find nearest models to the requested model along all grid lines in both directions
            - If possible, take the axis with the smallest interval between the neighbors and
            interpolate the result to the requested point
            - If no grid line has neighbors on both sides of the requested point, carry out
            nearest neighbor extrapolation along the axis with the closest neighbor
            - If extrapolation is impossible either, raise `ValueError()`

        Note: this function is not designed for accuracy or efficiency. Any "proper" handling of
        missing models should be carried out in the main interpolator. This function merely provides
        a safe way to read models from disk following the "just warn don't crash" philosophy

        Parameters
        ----------
        params : dict
            Same as in `read_grid_model()`

        Returns
        -------
        Same as in `read_grid_model()`
        """
        try:
            return self.grid.read_grid_model(params)
        except FileNotFoundError:
            # Find nearest neighbours along each axis in each of the two directions
            neighbors = {}; neighbor_distances = {}; neighbor_positions = {}
            for axis in params:
                neighbors[axis] = [None, None]
                neighbor_distances[axis] = [np.nan, np.nan]
                neighbor_positions[axis] = [np.nan, np.nan]
                for direction in [-1, 1]:
                    pos = (init_pos := np.where(np.sort(self.grid[axis]) == params[axis])[0][0]) + direction
                    while pos >= 0 and pos < len(self.grid[axis]):
                        neighbor_params = copy.copy(params)
                        neighbor_params[axis] = np.sort(self.grid[axis])[pos]
                        try:
                            neighbors[axis][int(direction > 0)] = self.read_grid_model(neighbor_params)
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
                x = [np.sort(self.grid[shortest])[neighbor_positions[shortest][0]], np.sort(self.grid[shortest])[neighbor_positions[shortest][1]]]
                self.warn('Cannot load model {}. Will interpolate from {}={},{}'.format(params, shortest, *x))
                return tuple(scp.interpolate.interp1d(x, [neighbors[shortest][0][i], neighbors[shortest][1][i]], axis = 0)(params[shortest]) for i in range(len(neighbors[shortest][0])))
            # Otherwise, extrapolate from the nearest neighbor
            intervals = {axis: np.nanmin(neighbor_distances[axis]) for axis in params if (not np.all(np.isnan(neighbor_distances[axis])))}
            if len(intervals) == 0: # None of the axes have viable models to interpolate or extrapolate
                raise ValueError('Unable to safely load {}: no models on the same gridlines found'.format(params))
            shortest = min(intervals, key = intervals.get)
            direction = np.where(~np.isnan(neighbor_distances[shortest]))[0][0]
            self.warn('Cannot load model {}. Will load {}={} instead'.format(params, shortest, self.grid[shortest][neighbor_positions[shortest][direction]]))
            return neighbors[shortest][direction]

    def convolution_integral(self, sigma, segment_left, segment_right, bin_left, bin_right):
        """Calculate weights of the convolution for an arbitrary flux density spectrum
        and a Gaussian kernel with fixed or slowly-varying standard deviation
        
        A source with a given (true) flux density spectrum is observed with a detector
        of given resolution that records the average flux density in a set of wavelength
        bins. This function estimates the linear coefficients `C1` and `C2` such that

        ``C1 * A + C2 * B``

        gives the contribution of the spectrum segment between the wavelengths
        `segment_left` and `segment_right`, to the detector bin with edge wavelengths
        `bin_left` and `bin_right`. It is assumed that the resolution of the detector
        is constant for each detector bin / spectrum segment pair, and that the flux
        density within the segment varies linearly with wavelength:

        ``flux density = A + B * wavelength``

        This function is primarily designed to handle model spectra, for which the flux
        density has been calculated at given wavelength points, and can be linearly
        interpolated between those points. The weight coefficients may then be evaluated
        for each segment between adjacent wavelength points, and the contributions
        of each segment can be added together to obtain the total flux density in the
        detector bin
        
        Parameters
        ----------
        sigma : array_like
            Detector resolution for the flux arriving from the spectrum segment of
            interest at the detector bin of interest. Adopted as the standard deviation
            of the Gaussian convolution kernel
        segment_left : array_like
            Lower wavelength bound of the spectrum segment of interest
        segment_right : array_like
            Upper wavelength bound of the spectrum segment of interest
        bin_left : array_like
            Lower wavelength bound of the detector bin of interest
        bin_right : array_like
            Upper wavelength bound of the detector bin of interest
        
        Returns
        -------
        C1 : array_like
            Weight coefficient of the vertical offset in the flux density spectrum segment
        C2 : array_like
            Weight coefficient of the linear slope in the flux density spectrum segment
        """
        sqrt_sigma = np.sqrt(2) * sigma
        sqpi = np.sqrt(np.pi)
        sq2pi = np.sqrt(2) * sqpi

        blsl = (segment_left - bin_left) / sqrt_sigma; blsl_sigma = segment_left ** 2 - bin_left ** 2 + sigma ** 2
        blsr = (segment_left - bin_right) / sqrt_sigma; blsr_sigma = segment_left ** 2 - bin_right ** 2 + sigma ** 2
        brsl = (segment_right - bin_left) / sqrt_sigma; brsl_sigma = segment_right ** 2 - bin_left ** 2 + sigma ** 2
        brsr = (segment_right - bin_right) / sqrt_sigma; brsr_sigma = segment_right ** 2 - bin_right ** 2 + sigma ** 2

        erfblsl = scp.special.erf(blsl); expblsl = np.exp(-blsl ** 2)
        erfblsr = scp.special.erf(blsr); expblsr = np.exp(-blsr ** 2)
        erfbrsl = scp.special.erf(brsl); expbrsl = np.exp(-brsl ** 2)
        erfbrsr = scp.special.erf(brsr); expbrsr = np.exp(-brsr ** 2)

        x = lambda x, erfxsl, erfxsr, expxsl, expxsr: (expxsl - expxsr) * sqrt_sigma / sqpi + (x - bin_left) * erfxsl + (bin_right - x) * erfxsr
        x1 = x(segment_left, erfblsl, erfblsr, expblsl, expblsr)
        x2 = x(segment_right, erfbrsl, erfbrsr, expbrsl, expbrsr)

        x = lambda erfxsl, erfxsr, expxsl, expxsr: (expxsl - expxsr) * sigma / sq2pi + 0.5 * (erfxsl - erfxsr)
        x3 = x(blsl_sigma * erfblsl, blsr_sigma * erfblsr, expblsl * (segment_left + bin_left), expblsr * (segment_left + bin_right))
        x4 = x(brsl_sigma * erfbrsl, brsr_sigma * erfbrsr, expbrsl * (segment_right + bin_left), expbrsr * (segment_right + bin_right))

        C1 = 0.5 * (x2 - x1) / (bin_right - bin_left)
        C2 = 0.5 * (sigma ** 2 * (erfblsl - erfblsr - erfbrsl + erfbrsr) - x3 + x4) / (bin_right - bin_left)
        return C1, C2

    def get_bin_edges(self, bins):
        """Convert reference wavelengths of detector bins to edge wavelengths
        
        `bins` is the array of reference wavelengths of length M. The M-1 inner bin edges are
        taken as midpoints between adjacent values of this array. The 2 outer bin edges are
        estimated by assuming that the first and the last bins are symmetric with respect to
        their reference wavelengths. The input must be strictly ascending
        
        Parameters
        ----------
        bins : array of length M
            Reference wavelengths of all detector bins
        
        Returns
        -------
        array of length M+1
            Bin edge wavelengths
        """
        if not np.all(bins[1:] > bins[:-1]):
            raise ValueError('Bin wavelengths must be strictly ascending')

        return np.concatenate([[bins[0] - (bins[1] - bins[0]) / 2], (bins[1:] + bins[:-1]) / 2, [bins[-1] + (bins[-1] - bins[-2]) / 2]])

    def convolution_weights(self, bins, x, sigma, clip = 5.0, mode = 'window', max_size = 15e6, flush_cache = False):
        """Calculate the complete convolution matrix for an arbitrary flux density spectrum
        and an arbitrary set of detector bins
        
        The final observed spectrum is then given by `C * flux`, where `flux` is the (true) flux
        density spectrum sampled at wavelengths `x`, and `C` is the convolution matrix calculated
        by this function. Here `*` represents matrix multiplication (dot product)
        
        It is assumed that the flux density between the sampling wavelengths is given by
        linear interpolation, while beyond the range of `x` it is zero

        All of `bins`, `x` and `sigma` must be given in the same units, e.g. A

        Since the number of detector bins and flux wavelengths may be very large, the output is
        returned as a sparse matrix in the COOrdinate format. The function also checks that the
        memory usage does not exceed the limit provided in the optional argument `max_size`

        The function implements memory caching
        
        Parameters
        ----------
        bins : array of length M
            Reference wavelengths of all detector bins. See `get_bin_edges()`
        x : array of length N
            Array of wavelengths where the flux density spectrum is sampled. The segments are
            defined as the N-1 intervals between adjacent values of this array. Must be
            strictly ascending
        sigma : float or array of length M or array of length N
            Resolution of the detector, defined as the standard deviation of the Gaussian
            convolution kernel. If scalar, constant resolution is adopted for all bins and all
            segments. If `mode` is 'window', the resolution is defined at each detector bin, and
            the length of this array must be M. If `mode` is 'dispersion', the resolution is
            defined at each wavelength in the spectrum and the length of this array must be N
        clip : float, optional
            Assume that the weights for the detector bin / spectrum segment pair are zero if
            the wavelength ranges of the bin and the segment are separated by this many values of
            `sigma` for this pair. The argument removes negligibly small weights from the result,
            allowing the final sparse matrix to take up less memory
        mode : str, optional
            If 'window', the resolution determines the width of the window, over which each
            detector bin is sampling the spectrum flux density. If 'dispersion', the resolution
            determines the dispersion range, into which each segment of the spectrum
            spreads out before it is binned by the detector. There is no distinction between
            these two modes if the resolution is constant. This argument is therefore ignored
            when `sigma` is scalar
        max_size : int, optional
            Maximum number of non-zero elements in the convolution matrix. An exception is thrown
            if the predicted number of non-zero elements exceeds this argument. The number of
            non-zero elements directly correlates with memory usage
        flush_cache : bool, optional
            If True, discard cache and calculate the convolution matrix afresh
        
        Returns
        -------
        C : scipy.sparse._coo.coo_matrix
            Convolution weight matrix, such that the dot product of this matrix and the flux vector
            gives the vector of average flux densities in the detector bins
        """
        global _convolution_weights_cache
        try:
            _convolution_weights_cache
        except:
            _convolution_weights_cache = {}

        # Since input data can be very large, we will not attempt sorting it here. Leave it up to the user
        if not np.all(bins[1:] > bins[:-1]):
            raise ValueError('Bin wavelengths must be strictly ascending')
        if not np.all(x[1:] > x[:-1]):
            raise ValueError('x must be strictly ascending')

        # If `sigma` is scalar, we can choose any mode and populate the entire array with constant values
        try:
            sigma[0]
        except:
            sigma = np.full(len(bins), sigma)
            mode = 'window'

        # Check for cached output
        if not flush_cache:
            hash_string = ''.join(list(map(lambda arg: hashlib.sha256(bytes(arg)).hexdigest(), [bins, x, sigma]))) + str(clip) + mode
            if hash_string in _convolution_weights_cache:
                return _convolution_weights_cache[hash_string]

        # Check dimensions of `sigma` depending on the mode
        if mode == 'window':
            if len(bins) != len(sigma):
                raise ValueError('In "window" mode, must provide sigma for each bin')
        elif mode == 'dispersion':
            if len(x) != len(sigma):
                raise ValueError('In "dispersion" mode, must provide sigma for each x')
            # For dispersion, we want `sigma` at spectrum segments, not wavelength points
            sigma = (sigma[1:] + sigma[:-1]) / 2

        # Calculate bin edges
        bin_edges = self.get_bin_edges(bins)

        # Estimate the range of wavelengths that fall within the clipping range (clip * sigma) of each bin
        clip_start = np.zeros(len(bins), dtype = int)
        clip_end = np.zeros(len(bins), dtype = int)
        for i in range(len(bins)):
            if mode == 'dispersion':
                clip_indices = np.where(((x[:-1] - bin_edges[i + 1]) < (clip * sigma)) & ((bin_edges[i] - x[1:]) < (clip * sigma)))[0]
                if len(clip_indices) == 0:
                    clip_start[i] = 0; clip_end[i] = 0
                else:
                    clip_start[i] = np.min(clip_indices); clip_end[i] = np.max(clip_indices) + 1
            else:
                # If sigma is constant for the bin, we can find the clipping range more efficiently with np.searchsorted()
                clip_start[i] = np.searchsorted(x[1:], bin_edges[i] - clip * sigma[i])
                clip_end[i] = np.searchsorted(x[:-1], bin_edges[i + 1] + clip * sigma[i])

        # Get the row and column indices of each non-zero element of the (to be calculated) convolution matrix. Columns correspond to
        # detector bins, and rows correspond to spectrum segments
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

        # Obtain the values of every argument required by convolution_integral() for each non-zero element of the convolution matrix
        segment_left = x[:-1][row_indices]
        segment_right = x[1:][row_indices]
        bin_left = bin_edges[col_indices]
        bin_right = bin_edges[col_indices + 1]
        if mode == 'dispersion':
            sigma_all = sigma[row_indices]
        else:
            sigma_all = sigma[col_indices]

        # Run the convolution integral calculator
        C1, C2 = self.convolution_integral(sigma_all, segment_left, segment_right, bin_left, bin_right)

        # The way convolution_integral() works, we now have "observed = C1 * offset + C2 * slope". Instead, we want to convert these
        # two matrices into a single matrix C that gives "observed = C * flux". We can do this with SciPy's sparse matrix operations.
        # First, pad both C1 and C2 with empty rows and get their C1[:-1] and C1[1:] slices (low and high as they are called here)
        C1_low  = scp.sparse.coo_matrix((C1, (col_indices, row_indices + 1)), shape = [len(bins), len(x)])
        C2_low  = scp.sparse.coo_matrix((C2, (col_indices, row_indices + 1)), shape = [len(bins), len(x)])
        C1_high = scp.sparse.coo_matrix((C1, (col_indices, row_indices)), shape = [len(bins), len(x)])
        C2_high = scp.sparse.coo_matrix((C2, (col_indices, row_indices)), shape = [len(bins), len(x)])
        # Now do the same with the wavelength vector and do the conversion
        padded_x = np.insert(x, [0, len(x)], [0, 0])
        C = C1_high + (C2_low - C1_low.multiply(padded_x[:-2])).multiply(1 / (padded_x[1:-1] - padded_x[:-2])) - (C2_high - C1_high.multiply(padded_x[1:-1])).multiply(1 / (padded_x[2:] - padded_x[1:-1]))

        _convolution_weights_cache[hash_string] = C
        return C

    def combine_arms(self, wl = None, flux = None):
        """Combine wavelengths and fluxes recorded by individual spectrograph arms into
        a single spectrum
        
        `wl` and `flux` are dictionaries, keyed by spectrograph arm identifiers (must be
        listed in `settings['arms']`). The function combines the spectra in each arm into a
        single spectrum. If the wavelength ranges overlap between two arms, the overlap
        region is removed from the lower priority arm (the priorities are set in
        `settings['arms']` as well)
        
        Parameters
        ----------
        wl : None or list or dict, optional
            If `wl` is a dictionary, it must be keyed by the identifiers of the arms that
            need to be combined. The values are 1D arrays of reference wavelengths of
            detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
            identifiers and the "typical" wavelength sampling for each arm (as defined in
            `settings['arms']`) will be assumed. Alternatively, set to `None` to use all
            arms defined in `settings['arms']` with "typical" wavelength sampling
        flux : None or dict, optional
            Dictionary of fluxes corresponding to the wavelength bins in `wl`. Must have
            the same keys and array lengths as `wl`. Alternatively, set to `None` if only
            wavelengths are required in the output
        
        Returns
        -------
        wl : array_like
            Combined and sorted array of reference wavelengths for all arms with overlapping
            wavelength bins removed according to arm priorities
        flux : array_like
            Corresponding array of fluxes. Only returned if the optional argument `flux` is
            not `None`
        """
        # Populate wl if not given
        if wl is None:
            wl = {arm: self.settings['arms'][arm]['wl'] for arm in self.settings['arms']}
        elif type(wl) is list:
            wl = {arm: self.settings['arms'][arm]['wl'] for arm in wl}
        else:
            wl = copy.deepcopy(wl) # We will be transforming wavelengths in place, so get a copy
        flux = copy.deepcopy(flux) # Same for flux

        # If flux is given, make sure its arms and dimensionality match wl
        if flux is not None:
            if set(wl.keys()) != set(flux.keys()):
                raise ValueError('The spectrograph arms in the `wl` and `flux` dictionaries do not match')
            if not np.all([len(wl[key]) == len(flux[key]) for key in wl]):
                raise ValueError('The dimensions of `wl` and `flux` do not match')

        # Resolve overlaps
        for arm_1, arm_2 in itertools.combinations(wl.keys(), 2):
            # We need at least two reference wavelengths to define wavelength bins
            if (len(wl[arm_1]) < 2) or (len(wl[arm_2]) < 2):
                continue
            # The overlap is evaluated for wavelength bin edges, not reference wavelengths themselves
            bin_edges_1 = self.get_bin_edges(wl[arm_1])
            bin_edges_2 = self.get_bin_edges(wl[arm_2])
            # Default priorities to zero if not set
            if 'priority' not in self.settings['arms'][arm_1]:
                priority_1 = 0
            else:
                priority_1 = self.settings['arms'][arm_1]['priority']
            if 'priority' not in self.settings['arms'][arm_2]:
                priority_2 = 0
            else:
                priority_2 = self.settings['arms'][arm_2]['priority']
            # If no overlap, do nothing
            if (np.min(bin_edges_1) > np.max(bin_edges_2)) or (np.min(bin_edges_2) > np.max(bin_edges_1)):
                continue
            # Compute the overlap region and check that priorities are different (else we don't know which arm to keep)
            overlap = [max(np.min(bin_edges_1), np.min(bin_edges_2)), min(np.max(bin_edges_1), np.max(bin_edges_2))]
            if priority_1 == priority_2:
                raise ValueError('Spectrograph arms {} and {} overlap in the region ({}:{}), but have equal priorities ({})'.format(arm_1, arm_2, *overlap, priority_2))
            if priority_1 > priority_2:
                high = bin_edges_1; low = bin_edges_2
                arm_high = arm_1; arm_low = arm_2
            else:
                high = bin_edges_2; low = bin_edges_1
                arm_high = arm_2; arm_low = arm_1
            # A higher priority arm cannot be internal to a lower priority arm, as that would slice the lower priority arm in two
            if (np.min(high) > np.min(low)) and (np.max(high) < np.max(low)):
                raise ValueError('Spectrograph arm {} (priority {}) is internal to {} (priority {}). This results in wavelength range discontinuity'.format(arm_high, max(priority_1, priority_2), arm_low, min(priority_1, priority_2)))
            # Mask out the overlap region in the lower priority arm
            mask = (low <= np.min(high)) | (low >= np.max(high))
            # Since "<=" and ">=" are used above, we may get "orphaned" bin edges, which need to be removed as well
            if (np.max(high) == np.max(low)):
                mask[low == np.max(high)] = False
            if (np.min(high) == np.min(low)):
                mask[low == np.min(high)] = False
            # Convert the bin edge mask into reference wavelength mask (both left and right edges must be defined for the bin to survive)
            mask = mask[:-1] & mask[1:]
            wl[arm_low] = wl[arm_low][mask]
            if flux is not None:
                flux[arm_low] = flux[arm_low][mask]

        # Remove empty arms
        for arm in list(wl.keys()):
            if len(wl[arm]) < 2:
                del wl[arm]
                if flux is not None:
                    del flux[arm]

        # Combine the arms into a single spectrum
        keys = list(wl.keys())
        wl = np.concatenate([wl[key] for key in keys])
        if flux is not None:
            flux = np.concatenate([flux[key] for key in keys])
        sort = np.argsort(wl)
        wl = wl[sort]
        if flux is not None:
            flux = flux[sort]

        if flux is not None:
            return wl, flux
        else:
            return wl
        
    def create_interpolator(self, resample = True, detector_wl = None, synphot_bands = [], mag_system = None, reddening = None, max_models = 10000):
        return ModelGridInterpolator(self,
                                     resample=resample,
                                     detector_wl=detector_wl,
                                     synphot_bands=synphot_bands,
                                     mag_system=mag_system,
                                     reddening=reddening,
                                     max_models=max_models)

    def simulate_observation(self, wl, flux, detector_wl = None, mask_unmodelled = True, clip = 5, combine = True):
        """Simulate observation of the model spectrum by a spectrograph
        
        If the model does not fully cover the range of a spectrograph arm, the output flux density
        in the affected detector wavelength bins will be set to `np.nan`. This behavior can be
        disabled by setting `mask_unmodelled` to False, in which case the edge effects of the
        convolution will be left in the output spectrum
        
        Parameters
        ----------
        wl : array_like
            Model wavelengths
        flux : array_like
            Model flux densities corresponding to `wl`. It is assumed that the flux density between
            the values of `wl` can be obtained by linear interpolation, and the flux density beyond
            the range of `wl` is zero
        detector_wl : None or list or dict, optional
            Reference wavelengths of the detector bins in each arm of the spectrograph. If
            `detector_wl` is a dictionary, it must be keyed by the identifiers of the arms that
            are used in this observation. The values are 1D arrays of reference wavelengths of
            detector bins (see `get_bin_edges()`). Alternatively, provide a list of arm
            identifiers and the "typical" wavelength sampling for each arm (as defined in
            `settings['arms']`) will be assumed. Alternatively, set to `None` to use all
            arms defined in `settings['arms']` with "typical" wavelength sampling
        mask_unmodelled : bool, optional
            If True, set to `np.nan` the flux density in all bins that are affected by the finite
            wavelength range of the model spectrum. Otherwise, the bins near the edges of the
            model will suffer from convolution edge effects, and the bins far beyond the wavelength
            range of the model will receive zero flux
        clip : float, optional
            Sigma-clipping parameter for the convolution calculator (see `convolution_weights()`).
            This parameter also determines the range within which the lack of model coverage at a
            particular wavelength can affect the detector bins
        combine : bool, optional
            If True, return a single wavelength and a single flux array that represents the combined
            spectrum across all arms of the spectrograph (see `combine_arms()`). Otherwise, return
            the spectra in individual arms
        
        Returns
        -------
        wl : dict or array_like
            Wavelengths of the observed spectrum with the spectrograph (keyed by spectrograph arm
            if `combine` is True)
        flux : dict or array_like
            Corresponding flux densities (keyed by spectrograph arm if `combine` is True)
        """
        # Populate detector_wl if not given
        if detector_wl is None:
            detector_wl = {arm: self.settings['arms'][arm]['wl'] for arm in self.settings['arms']}
        elif type(detector_wl) is list:
            detector_wl = {arm: self.settings['arms'][arm]['wl'] for arm in detector_wl}

        # Resample the spectrum onto the detector bins of each arm
        detector_flux = {}
        for arm in detector_wl:
            if 'sigma' not in self.settings['arms'][arm]:
                sigma = self.settings['arms'][arm]['FWHM'] / (2 * np.sqrt(2 * np.log(2)))
            else:
                if 'FWHM' in self.settings['arms'][arm]:
                    raise ValueError('Both FWHM and sigma provided for arm {}'.format(arm))
                sigma = self.settings['arms'][arm]['sigma']
            C = self.convolution_weights(detector_wl[arm], wl, sigma, clip = clip)
            detector_flux[arm] = C * flux
            # Remove wavelengths that exceed the modelled range
            if mask_unmodelled:
                message = 'In spectrograph arm {} the model does not cover the full wavelength range of the detector. Affected bins were set to np.nan'.format(arm)
                first = C.getcol(0).nonzero()[0]; last = C.getcol(-1).nonzero()[0]
                if len(first) == 0 and len(last) == 0:
                    # If neither the first nor the last spectrum segment contribute to detected flux, the model either
                    # covers the entire range of the detector or completely misses it
                    if (np.min(wl) < np.min(detector_wl[arm])) and (np.max(wl) > np.max(detector_wl[arm])):
                        continue
                    else:
                        detector_flux[arm] = np.full(len(detector_flux[arm]), np.nan)
                        self.warn(message)
                        continue
                self.warn(message)
                mask = np.full(len(detector_flux[arm]), True)
                if len(first) != 0:
                    mask[detector_wl[arm] <= detector_wl[arm][np.max(first)]] = False
                if len(last) != 0:
                    mask[detector_wl[arm] >= detector_wl[arm][np.min(last)]] = False
                detector_flux[arm][~mask] = np.nan


        # Combine the results into a single spectrum
        if combine:
            return self.combine_arms(detector_wl, detector_flux)
        else:
            return detector_wl, detector_flux

    def ranges_to_mask(self, arr, ranges, in_range_value = True, strict = False):
        """Convert a list of value ranges into a boolean mask, such that all values in `arr` that
        fall in any of the ranges correspond to `in_range_value`, and the rest correspond to
        `not in_range_value`
        
        Parameters
        ----------
        arr : array_like
            Array of values
        ranges : list of tuple
            List of two-element tuples, corresponding to the lower and upper bounds of each range
        in_range_value : bool, optional
            If True, in-range elements are set to True, and the rest are set to False in the mask.
            Otherwise, in-range values are set to False, and the rest are set to True
        strict : bool, optional
            If True, use strict comparison (`lower_bound < value < upper_bound`). Otherwise, use
            non-strict comparison (`lower_bound <= value <= upper_bound`)
        
        Returns
        -------
        array_like
            Boolean mask array of the same shape as `arr`
        """
        mask = np.full(np.shape(arr), not in_range_value)
        for window in ranges:
            if strict:
                mask[(arr > window[0]) & (arr < window[1])] = in_range_value
            else:
                mask[(arr >= window[0]) & (arr <= window[1])] = in_range_value
        return mask

    def estimate_continuum(self, wl, flux, ivar, npix = 100, k = 3, masks = None):
        """Estimate continuum correction in the spectrum using a spline fit
        
        The function carries out a weighted spline fit to a spectrum given by wavelengths in `wl`,
        flux densities in `flux` and using the weights in `ivar` (usually inverted variances)

        The wavelength regions given in `settings['masks']['continuum']` are excluded from the fit.
        If one of the excluded regions overlaps with the edge of the spectrum, the spline fit near
        the edge may be poorly conditioned (the spline will be extrapolated in that region, leading
        to potentially very large edge effects). If that part of the continuum is then used in
        stellar parameter determination, extremely poor convergence is likely. As such, when the
        spectral masks used by the fitter are passed in the optional `masks` parameter, it will be
        updated to mask out the affected region of the spectrum
        
        Parameters
        ----------
        wl : array_like
            Spectrum wavelengths
        flux : array_like
            Spectrum flux densities
        ivar : array_like
            Spectrum weights (inverted variances)
        npix : int, optional
            Desired interval between spline knots in pixels. The actual interval will be adjusted
            to keep the number of pixels in each spline segment identical
        k : int, optional
            Spline degree. Defaults to cubic
        masks : dict, optional
            Dictionary of boolean masks, keyed by stellar parameters. If given, this argument will be
            modified to exclude the regions of the spectrum potentially affected by spline extrapolation
            from the main fitter
        
        Returns
        -------
        array_like
            Estimated continuum correction multiplier at each wavelength in `wl`
        """
        mask = (ivar > 0) & (~np.isnan(ivar)) & (~np.isnan(flux))
        for bad_continuum_range in self.settings['masks']['continuum']:
            bad_continuum_range_mask = self.ranges_to_mask(wl, [bad_continuum_range], False)
            # Check for potential edge effects and remove the affected region from the fit
            if masks is not None:
                if (not bad_continuum_range_mask[mask][-1]) or (not bad_continuum_range_mask[mask][0]):
                    self.warn('Region {} excluded from continuum estimation overflows the spectral range. To avoid edge effects, this region will be ignored by the fitter'.format(bad_continuum_range))
                    for arm in masks:
                        masks[arm] &= self.ranges_to_mask(wl, [bad_continuum_range], False)
            mask &= bad_continuum_range_mask

        # Fit the spline
        t = wl[mask][np.round(np.linspace(0, len(wl[mask]), int(len(wl[mask]) / npix))).astype(int)[1:-1]]
        spline = scp.interpolate.splrep(wl[mask], flux[mask], w = ivar[mask], t = t, k = k)
        return scp.interpolate.splev(wl, spline)
    
    def get_pack_params_fun(self, dof):
        def pack_params(params):
            return np.array([ params[p] for p in dof ])
        
        def unpack_params(x):
            return { p: x[i] for i, p in enumerate(dof) }
        
        def pack_bounds(grid):
            return [ [ np.min(grid[axis]), np.max(grid[axis]) ] for axis in dof ]
        
        return pack_params, unpack_params, pack_bounds

    def get_log_likelihood_fun(self,  wl, flux, ivar, priors, dof, mask, interpolator, phot, diagnostic):

        pack_params, unpack_params, pack_bounds = self.get_pack_params_fun(dof)
        bounds = np.array(pack_bounds(interpolator._grid)).T

        # Evaluate the log of (likelihood * priors) given a set of model parameters
        def log_likelihood(x):

            # Likelihood is negative infinity if any of the parameters are outside the bounds
            if np.any(x < bounds[0]) or np.any(bounds[1] < x):
                return -np.inf

            params = unpack_params(x)

            if len(interpolator._synphot_bands) != 0:
                model_wl, model_flux, model_phot = interpolator(params)
            else:
                model_wl, model_flux = interpolator(params)

            cont = self.estimate_continuum(wl, flux / model_flux, ivar * model_flux ** 2,
                                           npix = self.settings['cont_pix'],
                                           k = self.settings['spline_order'])
            
            diagnostic['model_wl'] = model_wl
            diagnostic['model_flux'] = model_flux
            diagnostic['model_cont'] = cont

            log_l = -0.5 * np.sum((flux[mask] - (cont * model_flux)[mask]) ** 2 * ivar[mask])

            # Add priors
            # TODO

            # Add photometric colors
            # TODO

            return log_l
        
        return log_likelihood
    
    def get_fitting_mask(self, wl, flux, ivar, initial, dof, masks, interpolator):
        # Construct  and apply the fitting mask by superimposing the masks of individual parameters and removing bad pixels
        mask = np.full(len(wl), False)
        for param in dof:
            mask |= masks[param]
        mask &= (ivar > 0) & (~np.isnan(ivar)) & (~np.isnan(flux))
        mask &= ~np.isnan(interpolator(initial)[1])

        return mask

    def fit_model(self, wl, flux, ivar, initial, priors, dof, masks, interpolator, phot, method):
        """Fit the model to the spectrum
        
        Helper function to `chemfit()`. It sets up a model callback for `scp.optimize.curve_fit()` with
        the appropriate signature, defines the initial guesses and bounds for all free parameters, applies
        parameter masks and initiates the optimization routine

        The results of the fit and the associated errors are placed in the `initial` and `errors` arguments
        
        Parameters
        ----------
        wl : array_like
            Spectrum wavelengths
        flux : array_like
            Spectrum flux densities
        ivar : array_like
            Spectrum weights (inverted variances)
        initial : dict
            Initial guesses for the stellar parameters, keyed by parameter. The updated parameters after the
            optimization are stored in this dictionary as well
        priors : dict of 2-element tuples
            Prior estimates of the stellar parameters, keyed by parameter. Each element is a tuple with the
            first element storing the best estimate and the second element storing its uncertainty. All tuples
            of length other than 2 are ignored
        dof : list
            List of parameters to be optimized. The rest are treated as fixed to their initial values
        masks : dict
            Dictionary of boolean masks, keyed by stellar parameters. The masks determine which wavelengths are
            included in the fit for each parameter. If multiple parameters are fit simultaneously, the masks are
            logically added (or)
        interpolator : ModelGridInterpolator
            Model grid interpolator object that will be used to construct models during optimization
        phot : dict
            Photometric colors of the star. Each color is keyed by `BAND1#BAND2`, where `BAND1` and `BAND2` are
            the transmission profile filenames of the filters, as required by `synphot()`. Each element is a
            2-element tuple, where the first element is the measured color, and the second element is the
            uncertainty in the measurement. The dictionary may also include optional elements `reddening`
            (E(B-V), single numerical value,), and `mag_system` (one of the magnitude systems supported by
            `synphot()`, single string)
        method : str
            Method to determine the best-fit stellar parameters. Must correspond to a callable function of form
            `fit_{method}(f, x, y, p0, sigma, bounds)`, where the arguments have the same meaning as those
            used by `scipy.optimize.curve_fit()`. The callable must return 3 values: best-fit parameter values
            in the same order as accepted by `f()`, errors in the best-fit parameter values, and a dictionary of
            additional data from the fit (e.g. covariance matrices, MCMC chains etc)
        Returns
        -------
        dict
            Additional data from the fit as returned by the fitting method callable function. If
            `settings['return_diagnostics']`, will also return detailed diagnostic data, including the observed
            spectrum, fitting masks and the best-fit model
        """

        # Create a dictionary to store diagnostic data in
        # TODO: this should be a trace object
        global _fitter_diagnostic_storage
        _fitter_diagnostic_storage = {}
        diagnostic = _fitter_diagnostic_storage

        # Construct  and apply the fitting mask by superimposing the masks of individual parameters and removing bad pixels
        mask = self.get_fitting_mask(wl, flux, ivar, initial, dof, masks, interpolator)

        # Define p0 and bounds. The bounds are set to the model grid range stored in the interpolator object
        pack_params, unpack_params, pack_bounds = self.get_pack_params_fun(dof)
        p0 = pack_params(initial)
        bounds = pack_bounds(interpolator._grid)

        llh = self.get_log_likelihood_fun(wl, flux, ivar,
                                          priors,
                                          dof,
                                          mask,
                                          interpolator,
                                          phot,
                                          diagnostic)
        
        # Run the optimizer and save the results in "initial" and "errors"
        # fit = mcmc_fit(f, x, y, p0 = p0, bounds = bounds, sigma = sigma)
        if method == 'mcmc':
            best, error, extra = self._fit_mcmc(llh, p0, bounds)
        elif method == 'gradient_descent':
            best, error, extra = self._fit_gradient_descent(lambda x: -llh(x), p0, bounds)
        else:
            raise NotImplementedError()
        
        # Extract results
        best = unpack_params(best)
        errors = unpack_params(error)

        # Provide diagnostic data if requested
        if self.settings['return_diagnostics']:
            extra['observed'] = {'wl': wl, 'flux': flux, 'ivar': ivar}
            extra['mask'] = mask
            extra['fit'] = {
                # TODO: evaluate best model
                # 'x': x, 'y': y, 'sigma': sigma
                # 'f': f(x, *fit[0])
                'p0': p0,
                'bounds': bounds,
                'dof': dof
            }
            
            extra['model'] = {
                'wl': diagnostic['model_wl'],
                'flux': diagnostic['model_flux'],
                'cont': diagnostic['model_cont']
            }

        return best, errors, extra

    # TODO: DELETE
    def fit_model_aside(self, wl, flux, ivar, initial, priors, dof, errors, masks, interpolator, phot, method):
        """Fit the model to the spectrum
        
        Helper function to `chemfit()`. It sets up a model callback for `scp.optimize.curve_fit()` with
        the appropriate signature, defines the initial guesses and bounds for all free parameters, applies
        parameter masks and initiates the optimization routine

        The results of the fit and the associated errors are placed in the `initial` and `errors` arguments
        
        Parameters
        ----------
        wl : array_like
            Spectrum wavelengths
        flux : array_like
            Spectrum flux densities
        ivar : array_like
            Spectrum weights (inverted variances)
        initial : dict
            Initial guesses for the stellar parameters, keyed by parameter. The updated parameters after the
            optimization are stored in this dictionary as well
        priors : dict of 2-element tuples
            Prior estimates of the stellar parameters, keyed by parameter. Each element is a tuple with the
            first element storing the best estimate and the second element storing its uncertainty. All tuples
            of length other than 2 are ignored
        dof : list
            List of parameters to be optimized. The rest are treated as fixed to their initial values
        errors : dict
            Estimate errors in the best-fit parameter values are placed in this dictionary
        masks : dict
            Dictionary of boolean masks, keyed by stellar parameters. The masks determine which wavelengths are
            included in the fit for each parameter. If multiple parameters are fit simultaneously, the masks are
            logically added (or)
        interpolator : ModelGridInterpolator
            Model grid interpolator object that will be used to construct models during optimization
        phot : dict
            Photometric colors of the star. Each color is keyed by `BAND1#BAND2`, where `BAND1` and `BAND2` are
            the transmission profile filenames of the filters, as required by `synphot()`. Each element is a
            2-element tuple, where the first element is the measured color, and the second element is the
            uncertainty in the measurement. The dictionary may also include optional elements `reddening`
            (E(B-V), single numerical value,), and `mag_system` (one of the magnitude systems supported by
            `synphot()`, single string)
        method : str
            Method to determine the best-fit stellar parameters. Must correspond to a callable function of form
            `fit_{method}(f, x, y, p0, sigma, bounds)`, where the arguments have the same meaning as those
            used by `scipy.optimize.curve_fit()`. The callable must return 3 values: best-fit parameter values
            in the same order as accepted by `f()`, errors in the best-fit parameter values, and a dictionary of
            additional data from the fit (e.g. covariance matrices, MCMC chains etc)
        Returns
        -------
        dict
            Additional data from the fit as returned by the fitting method callable function. If
            `settings['return_diagnostics']`, will also return detailed diagnostic data, including the observed
            spectrum, fitting masks and the best-fit model
        """
        # Create a dictionary to store diagnostic data in
        global _fitter_diagnostic_storage
        _fitter_diagnostic_storage = {}
        diagnostic = _fitter_diagnostic_storage

        # This function will be passed to curve_fit() as the model callback. The <signature> comment is a placeholder
        # to be replaced with the interpretation of the function signature later
        def f(x, params, mask, data_wl = wl, data_flux = flux, data_ivar = ivar, priors = priors, interpolator = interpolator, phot = phot, diagnostic = _fitter_diagnostic_storage):
            # <signature>

            # Load the requested model
            if len(interpolator._synphot_bands) != 0:
                model_wl, model_flux, model_phot = interpolator(params)
            else:
                model_wl, model_flux = interpolator(params)
            cont = self.estimate_continuum(data_wl, data_flux / model_flux, data_ivar * model_flux ** 2, npix = self.settings['cont_pix'], k = self.settings['spline_order'])
            diagnostic['model_wl'] = model_wl; diagnostic['model_flux'] = model_flux; diagnostic['model_cont'] = cont
            model_wl = model_wl[mask]; model_flux = (cont * model_flux)[mask]

            # Add priors
            index = 1
            for param in sorted(list(params.keys())):
                if len(np.atleast_1d(priors[param])) == 2:
                    model_wl = np.concatenate([np.array([-index]), model_wl])
                    model_flux = np.concatenate([np.array([params[param]]), model_flux])
                    index += 1

            # Add photometric colors
            for i, color in enumerate(interpolator._synphot_bands):
                model_wl = np.concatenate([np.array([-index * 100]), model_wl])
                model_flux = np.concatenate([np.array([model_phot[i]]), model_flux])
                index += 1

            return model_flux

        # Define p0 and bounds. The bounds are set to the model grid range stored in the interpolator object
        p0 = [initial[param] for param in dof]
        bounds = np.array([[np.min(interpolator._grid[axis]), np.max(interpolator._grid[axis])] for axis in dof]).T

        # Construct  and apply the fitting mask by superimposing the masks of individual parameters and removing bad pixels
        mask = np.full(len(wl), False)
        for param in dof:
            mask |= masks[param]
        mask &= (ivar > 0) & (~np.isnan(ivar)) & (~np.isnan(flux))
        mask &= ~np.isnan(interpolator(initial)[1])
        x = wl[mask]; y = flux[mask]; sigma = ivar[mask] ** -0.5

        # Since we do not a priori know the number of parameters being fit, we need to dynamically update the signature of the
        # model callback, f(). Unfortunately, there appears to be no better way to do that than by retrieving the source code
        # of the function with inspect, updating it, and reevaluating with exec()
        f = inspect.getsource(f).split('\n')
        f[0] = f[0].replace('params', ', '.join(dof))
        f[0] = f[0].replace('mask', 'mask = mask')
        f[1] = f[1].replace('# <signature>', 'params = {' + ', '.join(['\'{}\': {}'.format(param, [param, initial[param]][param not in dof]) for param in initial]) + '}')
        scope = {'priors': priors, 'phot': phot, 'interpolator': interpolator, 'mask': mask, 'np': np, 'wl': wl, 'flux': flux, 'ivar': ivar, 'estimate_continuum': estimate_continuum, 'settings': settings, 'synphot': synphot, '_fitter_diagnostic_storage': _fitter_diagnostic_storage}
        exec('\n'.join(f)[f[0].find('def'):], scope)
        f = scope['f']

        # Add priors. Each prior is just an extra pixel in the spectrum
        index = 1
        for param in sorted(list(priors.keys())):
            if len(np.atleast_1d(priors[param])) == 2:
                x = np.concatenate([np.array([-index]), x])
                y = np.concatenate([np.array([priors[param][0]]), y])
                sigma = np.concatenate([np.array([priors[param][1]]), sigma])
                index += 1

        # Add photometric colors
        for color in interpolator._synphot_bands:
            color = '#'.join(color)
            x = np.concatenate([np.array([-index * 100]), x])
            y = np.concatenate([np.array([phot[color][0]]), y])
            sigma = np.concatenate([np.array([phot[color][1]]), sigma])
            index += 1

        # Run the optimizer and save the results in "initial" and "errors"
        # fit = mcmc_fit(f, x, y, p0 = p0, bounds = bounds, sigma = sigma)
        fit = globals()['fit_{}'.format(method)](f, x, y, p0, sigma, bounds)
        for i, param in enumerate(dof):
            initial[param] = fit[0][i]
            errors[param] = fit[1][i]

        # Provide diagnostic data if requested
        if self.settings['return_diagnostics']:
            fit[2]['observed'] = {'wl': wl, 'flux': flux, 'ivar': ivar}
            fit[2]['mask'] = mask
            fit[2]['fit'] = {'x': x, 'y': y, 'sigma': sigma, 'f': f(x, *fit[0]), 'p0': p0, 'bounds': bounds, 'dof': dof}
            fit[2]['model'] = {'wl': diagnostic['model_wl'], 'flux': diagnostic['model_flux'], 'cont': diagnostic['model_cont']}
            fit[2]['cost'] = (fit[2]['fit']['f'] - fit[2]['fit']['y']) ** 2.0 / fit[2]['fit']['sigma'] ** 2.0

        return fit[2]
    
    def _fit_gradient_descent(self, f, p0, bounds):

        # TODO: the default method of minimize returns the inverse Hessian
        #       but this is not the case with methods such as Nelder-Mead
        #       use some differentiation lib here to get the Hessian in those
        #       cases

        # TODO: Add optimizer options from settings

        res = scp.optimize.minimize(f, p0, bounds = bounds)
        best = res.x
        cov = res.hess_inv.todense()
        error = np.sqrt(np.diag(cov))
        cost = res.fun

        return best, error, { 'cov': cov, 'cost': cost }

    # TODO: DELETE
    #       This uses curve_fit which is too high-level to finely control the optimization
    def fit_gradient_descent_aside(self, f, x, y, p0, sigma, bounds):
        """Fit a 2D data series to a model using the Trust Region Reflective gradient descent algorithm
        
        The fit is carried out using `scipy.optimize.curve_fit()`. The covariance matrix is returned as
        additional data
        
        Parameters
        ----------
        See the parameters of `scipy.optimize.curve_fit()`
        
        Returns
        -------
        best : array_like
            Best-fit model parameters
        errors : array_like
            Errors in the best-fit model parameters
        extra : dict
            Dictionary with a single key, 'cov', that contains the covariance matrix of the fit
        """
        fit = scp.optimize.curve_fit(f, x, y, p0 = p0, sigma = sigma, bounds = bounds, **self.settings['gradient_descent']['curve_fit'])
        best = fit[0]
        errors = np.sqrt(np.diagonal(fit[1]))
        return best, errors, {'cov': fit[1]}
    
    def _init_mcmc_gradient_descent(self, best, errors, bounds):
        initial = []
        for i in range(len(best)):
            # Gaussian initial positions based on gradient descent
            v = scp.stats.truncnorm.rvs(loc = best[i], scale = errors[i],
                                        a = (bounds[i][0] - best[i]) / errors[i],
                                        b = (bounds[i][1] - best[i]) / errors[i],
                                        size = self.settings['mcmc']['nwalkers'])
            initial.append(v)
        return np.vstack(initial).T
    
    def _init_mcmc_uniform(self, p0, bounds):
        initial = []
        for i in range(len(p0)):
            # Uniformly random initial walker positions):
            v = np.random.uniform(bounds[0][i], bounds[1][i], self.settings['mcmc']['nwalkers'])
            initial.append(v)
        return np.vstack(initial).T
    
    def _mcmc_convergence(self, chain, c = 5):
        """Calculate the convergence parameters of an MCMC chain
        
        This function takes the chain output by emcee, and computes the autocorrelation length and the
        Geweke drift for each dimension of the parameter space
        
        Parameters
        ----------
        chain : array_like
            MCMC chain as returned by `emcee.EnsembleSampler().get_chain()`
        c : number, optional
            Step size for the autocorrelation window search (see `emcee.autocorr.integrated_time()`)
        
        Returns
        -------
        autocorr : array_like
            2D array with autocorrelation lengths for each parameter (first dimension) and each walker
            (second dimension), expressed as the number of autocorrelation lengths contained within the
            provided chain. Larger values indicate better convergence (emcee documentaion recommends
            requiring the minimum value of this array to exceed 50)
        geweke : array_like
            Array of Geweke drifts (z-scores) for each parameter. Larger values indicate poor convergence
        """

        import emcee

        nsteps, nwalkers, ndim = np.shape(chain)
        autocorr = np.zeros([ndim, nwalkers])
        geweke = np.zeros(ndim)
        for i in range(ndim):
            for w in range(nwalkers):
                f = emcee.autocorr.function_1d(chain[:, w, i])
                taus = 2.0 * np.cumsum(f) - 1.0
                windows = emcee.autocorr.auto_window(taus, c)
                tau = taus[windows]
                autocorr[i,w] = nsteps / tau

            flatchain = chain.reshape(chain.shape[0] * chain.shape[1], -1)
            a = flatchain[:len(flatchain) // 4, i]
            b = flatchain[-len(flatchain) // 4:, i]
            geweke[i] = (np.mean(a) - np.mean(b)) / (np.var(a) + np.var(b)) ** 0.5

        return autocorr, geweke
    
    def _fit_mcmc(self, log_likelihood, p0, bounds):
        try:
            import emcee
        except:
            raise ImportError('emcee not installed')
        
        # Choose initial walker positions
        if self.settings['mcmc']['initial'] == 'gradient_descent':
            best, errors, extra_gd = self._fit_gradient_descent(lambda x: -log_likelihood(x), p0, bounds)
            initial = self._init_mcmc_gradient_descent(best, errors, bounds)
        elif self.settings['mcmc']['initial'] == 'uniform':
            initial = self._init_mcmc_gradient_uniform(p0, bounds)
        
        # Run the MCMC sampler
        sampler = emcee.EnsembleSampler(self.settings['mcmc']['nwalkers'],
                                        np.shape(initial)[1],
                                        log_likelihood,
                                        args = [])
        
        sampler.run_mcmc(initial, self.settings['mcmc']['nsteps'], progress = self.settings['mcmc']['progress'])
        chain = sampler.get_chain(flat = False)
        
        autocorr, geweke = self._mcmc_convergence(chain)
        flatchain = chain[self.settings['mcmc']['discard']:,:,:].reshape((chain.shape[0] - self.settings['mcmc']['discard']) * chain.shape[1], -1)
        extra = {'chain': chain, 'initial': initial, 'autocorr': autocorr, 'geweke': geweke}
        
        if self.settings['mcmc']['initial'] == 'gradient_descent':
            extra['gradient_descent'] = extra_gd
            extra['gradient_descent']['fit'] = best
            extra['gradient_descent']['errors'] = errors
        
        # TODO: Fix these statistics. Either calculate median and quantiles or
        #       calculate variance around median
        return np.median(flatchain, axis = 0), np.std(flatchain, axis = 0), extra

    def fit_mcmc_aside(self, f, x, y, p0, sigma, bounds):
        """Fit a 2D data series to a model using Markov Chain Monte Carlo (MCMC) sampling
        
        The fit is carried out using `emcee.EnsembleSampler()`. The MCMC chains for individual walkers are
        returned as additional data. The initial positions of the walkers are drawn from a random uniform
        distribution, within the prescribed bounds. The best-fit parameters and their errors are calculated
        as the median and the standard deviation of the chains after the removal of the burn-in steps
        
        Parameters
        ----------
        See the parameters of `scipy.optimize.curve_fit()`
        
        Returns
        -------
        best : array_like
            Best-fit model parameters
        errors : array_like
            Errors in the best-fit model parameters
        extra : dict
            Dictionary with a single key, 'chain', that contains the full MCMC chains for each walker
        """
        try:
            import emcee
        except:
            raise ImportError('emcee not installed')

        def mcmc_convergence(chain, c = 5):
            """Calculate the convergence parameters of an MCMC chain
            
            This function takes the chain output by emcee, and computes the autocorrelation length and the
            Geweke drift for each dimension of the parameter space
            
            Parameters
            ----------
            chain : array_like
                MCMC chain as returned by `emcee.EnsembleSampler().get_chain()`
            c : number, optional
                Step size for the autocorrelation window search (see `emcee.autocorr.integrated_time()`)
            
            Returns
            -------
            autocorr : array_like
                2D array with autocorrelation lengths for each parameter (first dimension) and each walker
                (second dimension), expressed as the number of autocorrelation lengths contained within the
                provided chain. Larger values indicate better convergence (emcee documentaion recommends
                requiring the minimum value of this array to exceed 50)
            geweke : array_like
                Array of Geweke drifts (z-scores) for each parameter. Larger values indicate poor convergence
            """
            nsteps, nwalkers, ndim = np.shape(chain)
            autocorr = np.zeros([ndim, nwalkers])
            geweke = np.zeros(ndim)
            for i in range(ndim):
                for w in range(nwalkers):
                    f = emcee.autocorr.function_1d(chain[:, w, i])
                    taus = 2.0 * np.cumsum(f) - 1.0
                    windows = emcee.autocorr.auto_window(taus, c)
                    tau = taus[windows]
                    autocorr[i,w] = nsteps / tau

                flatchain = chain.reshape(chain.shape[0] * chain.shape[1], -1)
                a = flatchain[:len(flatchain) // 4, i]
                b = flatchain[-len(flatchain) // 4:, i]
                geweke[i] = (np.mean(a) - np.mean(b)) / (np.var(a) + np.var(b)) ** 0.5

            return autocorr, geweke

        # Choose initial walker positions
        if self.settings['mcmc']['initial'] == 'gradient_descent':
            best, errors, extra_gd = self.fit_gradient_descent(f, x, y, p0, sigma, bounds)
        initial = []
        for i in range(len(p0)):
            # Uniformly random initial walker positions
            if self.settings['mcmc']['initial'] == 'uniform':
                initial += [np.random.uniform(bounds[0][i], bounds[1][i], self.settings['mcmc']['nwalkers'])]
            # Gaussian initial positions based on gradient descent
            elif self.settings['mcmc']['initial'] == 'gradient_descent':
                initial += [scp.stats.truncnorm.rvs(loc = best[i], scale = errors[i], a = (bounds[0][i] - best[i]) / errors[i], b = (bounds[1][i] - best[i]) / errors[i], size = self.settings['mcmc']['nwalkers'])]
            else:
                raise ValueError('Unrecognized initial walker distribution {}'.format(self.settings['mcmc']['initial']))

        initial = np.vstack(initial).T

        def log_likelihood(p0, x, y, sigma, bounds):
            for i in range(len(p0)):
                if p0[i] <= bounds[0][i] or p0[i] >= bounds[1][i]:
                    return -np.inf
            model = f(x, *p0)
            return np.sum(scp.stats.norm.logpdf(y, model, sigma))

        # Run the MCMC sampler
        sampler = emcee.EnsembleSampler(self.settings['mcmc']['nwalkers'], np.shape(initial)[1], log_likelihood, args = [x, y, sigma, bounds])
        sampler.run_mcmc(initial, self.settings['mcmc']['nsteps'], progress = self.settings['mcmc']['progress'])
        chain = sampler.get_chain(flat = False)
        autocorr, geweke = mcmc_convergence(chain)
        flatchain = chain[self.settings['mcmc']['discard']:,:,:].reshape((chain.shape[0] - self.settings['mcmc']['discard']) * chain.shape[1], -1)
        extra = {'chain': chain, 'initial': initial, 'autocorr': autocorr, 'geweke': geweke}
        if self.settings['mcmc']['initial'] == 'gradient_descent':
            extra['gradient_descent'] = extra_gd
            extra['gradient_descent']['fit'] = best
            extra['gradient_descent']['errors'] = errors
        return np.median(flatchain, axis = 0), np.std(flatchain, axis = 0), extra

    def chemfit(self, wl, flux, ivar, initial, phot = {}, method = 'gradient_descent'):
        """Determine the stellar parameters of a star given its spectrum
        
        Parameters
        ----------
        wl : dict
            Spectrum wavelengths keyed by spectrograph arm
        flux : dict
            Spectrum flux densities keyed by spectrograph arm
        ivar : dict
            Spectrum weights (inverted variances) keyed by spectrograph arm
        initial : dict
            Initial guesses for the stellar parameters, keyed by parameter. Each parameter supported
            by the model grid must be listed. The value of each element is either a float or a
            2-element tuple. In the former case, the value is treated as the initial guess to the
            fitter. Otherwise, the first element is treated as an initial guess and the second value
            is treated as the prior uncertainty in the parameter
        phot : dict, optional
            Photometric colors of the star (if available). The colors and the spectrum will be fit to
            the models simultaneously to attain stricter constraints on the stellar parameters. Each
            color is keyed by `BAND1#BAND2`, where `BAND1` and `BAND2` are the transmission profile
            filenames of the filters, as required by `synphot()`. Each element is a 2-element tuple,
            where the first element is the measured color, and the second element is the uncertainty
            in the measurement. The dictionary may also include optional elements `reddening` (E(B-V),
            single numerical value), and `mag_system` (one of the magnitude systems supported by
            `synphot()`, single string)
        method : str
            Method to use for model fitting. Currently supported methods are 'gradient_descent' that
            employs the Trust Region Reflective algorithm implemented in `scipy`, and 'mcmc' that uses
            the MCMC sampler implemented in `emcee`
        
        Returns
        -------
        dict
            Fitting results. Dictionary with the following keys:
                'fit': Best-fit stellar parameters
                'errors': Standard errors in the best-fit parameters from the diagonal of covariance matrix
                'interpolator_statistics': Interpolator statistics (see `ModelGridInterpolator().statistics`)
                'warnings': Warnings issued during the fitting process
                'extra': Additional fit data and diagnostic data depending on the chosen fitting method
        """
        # Remember the length of pre-existing warnings stack
        warnings_stack_length = len(self.warnings_stack)

        # Combine arms
        wl_combined, flux_combined = self.combine_arms(wl, flux)
        ivar_combined = self.combine_arms(wl, ivar)[1]

        # Build the interpolator and resampler
        synphot_bands = [color.split('#') for color in phot if len(color.split('#')) == 2]
        if 'reddening' in phot:
            reddening = phot['reddening']
        else:
            reddening = self.settings['default_reddening']
        if 'mag_system' in phot:
            mag_system = phot['mag_system']
        else:
            mag_system = self.settings['default_mag_system']
        interpolator = self.create_interpolator(detector_wl = wl, synphot_bands = synphot_bands, reddening = reddening, mag_system = mag_system)

        # Get the fitting masks for each parameter
        masks = {}
        for param in initial:
            mask = {}
            ranges_specific = []
            ranges_general = []
            for arm in wl:
                if (arm in self.settings['masks']) and (param in self.settings['masks'][arm]):
                    ranges_specific += list(self.settings['masks'][arm][param])
                if ('all' in self.settings['masks']) and (param in self.settings['masks']['all']):
                    ranges_specific += list(self.settings['masks']['all'][param])
                if (arm in self.settings['masks']) and ('all' in self.settings['masks'][arm]):
                    ranges_general += list(self.settings['masks'][arm]['all'])
                if ('all' in self.settings['masks']) and ('all' in self.settings['masks']['all']):
                    ranges_general += list(self.settings['masks']['all']['all'])
            masks[param] = self.ranges_to_mask(wl_combined, ranges_general)
            if len(ranges_specific) != 0:
                masks[param] &= self.ranges_to_mask(wl_combined, ranges_specific)
        # Run the continuum fitter once to give it a chance to update the fitting masks if it needs to
        cont = self.estimate_continuum(wl_combined, flux_combined, ivar_combined, npix = self.settings['cont_pix'], k = self.settings['spline_order'], masks = masks)

        # Run the main fitter
        fit, errors, extra = self.fit_model(wl_combined, flux_combined, ivar_combined, initial, None, np.atleast_1d(self.settings['fit_dof']), masks, interpolator, phot, method = method)

        # Get the texts of unique issued warnings
        warnings = np.unique(self.warnings_stack[warnings_stack_length:])
        inv_warnings_messages = {self.warnings_messages[key]: key for key in self.warnings_messages}
        warnings = [inv_warnings_messages[warning_id] for warning_id in warnings]

        return {'fit': fit, 'errors': errors, 'extra': extra, 'interpolator_statistics': interpolator.statistics, 'warnings': warnings}

    def synphot(self, wl, flux, teff, bands, mag_system = None, reddening = None):

        mag_system = mag_system if mag_system is not None else self.settings['default_mag_system']
        reddening = reddening if reddening is not None else self.settings['default_reddening']

        # Placeholder function to compute synthetic photometry (bolometric corrections). For now, this relies on the BasicATLAS routine
        import atlas
        band_filenames = list(map(lambda fn: self.settings['filter_dir'] + '/' + fn, bands))
        phot = atlas.synphot(None, mag_system, reddening, band_filenames, spectrum = {'wl': wl, 'flux': flux, 'teff': teff}, silent = True)
        return {os.path.basename(key): phot[key] for key in phot}

