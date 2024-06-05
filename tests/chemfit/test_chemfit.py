import os
from unittest import TestCase
import numpy as np

from chemfit import Chemfit
from chemfit.grid import Grid7
from chemfit.instrument import PFS

class TestChemfit(TestCase):
    def test_initialize(self):       
        cf = Chemfit()
        cf.initialize(Grid7(), PFS())

        self.assertEqual(13, len(cf.settings))

    def test_initialize_from_presets(self):
        cf = Chemfit(script_dir = os.path.dirname(os.path.realpath(__file__)))
        cf.initialize(Grid7(), PFS(), 'test')

        self.assertEqual(13, len(cf.settings))

    def test_safe_read_grid_model(self):
        cf = Chemfit()
        cf.initialize(Grid7(), PFS())

        wl, fl = cf.safe_read_grid_model({'teff': 5000, 'logg': 4.5, 'zscale': 0.0, 'alpha': 0.0})

        self.assertEqual(35716, wl.size)
        self.assertEqual(35716, fl.size)

    def test_create_interpolator(self):
        cf = Chemfit()
        cf.initialize(Grid7(), PFS())

        params = {'teff': 4000, 'logg': 1.5, 'zscale': -1.0, 'alpha': 0.3}

        # Interpolate the model grid
        interpolator = cf.create_interpolator(resample = False)
        model_wl, model_flux = interpolator(params)

    def test_simulate_observation(self):
        cf = Chemfit()
        cf.initialize(Grid7(), PFS())

        params = {'teff': 4000, 'logg': 1.5, 'zscale': -1.0, 'alpha': 0.3}

        # Interpolate the model grid
        interpolator = cf.create_interpolator(resample = False)
        model_wl, model_flux = interpolator(params)

        # Generate simulated observation
        obs_wl, obs_fl = cf.simulate_observation(model_wl, model_flux, detector_wl = ['blue', 'red_mr'], combine = False)

        self.assertEqual(4096, obs_wl['blue'].size)
        self.assertEqual(4096, obs_fl['blue'].size)

    def chemfit_helper(self):
        cf = Chemfit()
        cf.initialize(Grid7(), PFS())

        params = {'teff': 4000, 'logg': 1.5, 'zscale': -1.0, 'alpha': 0.3}
        initial = {'teff': 5000, 'logg': 2.0, 'zscale': 0.0, 'alpha': 0.0}

        # Interpolate the model grid
        interpolator = cf.create_interpolator(resample = False)
        model_wl, model_flux = interpolator(params)

        # Generate simulated observation
        obs_wl, obs_fl = cf.simulate_observation(model_wl, model_flux, detector_wl = ['blue', 'red_mr'], combine = False)

        # Generate noise with SNR=15
        SNR = 15
        obs_ivar = {}
        for arm in obs_fl:
            sigma = obs_fl[arm] / SNR
            obs_ivar[arm] = sigma ** -2.0
            obs_fl[arm] = np.random.normal(obs_fl[arm], sigma)

        return cf, obs_wl, obs_fl, obs_ivar, initial

    def test_chemfit_mle(self):
        cf, obs_wl, obs_fl, obs_ivar, initial = self.chemfit_helper()
        res = cf.chemfit(obs_wl, obs_fl, obs_ivar, initial, method = 'gradient_descent')

    def test_chemfit_mcmc(self):
        cf, obs_wl, obs_fl, obs_ivar, initial = self.chemfit_helper()
        cf.settings['mcmc']['nsteps'] = 10
        cf.settings['mcmc']['discard'] = 5
        res = cf.chemfit(obs_wl, obs_fl, obs_ivar, initial, method = 'mcmc')
