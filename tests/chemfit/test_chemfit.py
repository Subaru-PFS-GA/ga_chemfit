import os
from unittest import TestCase

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
        cf.settings['griddir'] = '/datascope/subaru/data/pfsspec/models/stellar/grid/roman'

        wl, fl = cf.safe_read_grid_model({'teff': 5000, 'logg': 4.5, 'zscale': 0.0, 'alpha': 0.0})

        self.assertEqual(35716, wl.size)
        self.assertEqual(35716, fl.size)