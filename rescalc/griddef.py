import numpy as np
import itertools

zscale = [-5.0, -4.8, -4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
alpha = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
logg = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
teff = [3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5800, 6000, 6200, 6400, 6600, 6800, 7000, 7200, 7400, 7600, 7800, 8000]

import pickle
import scipy as scp
f = open('/expanse/lustre/projects/csd835/rgerasim/pfsgrid/Cgrid.pkl', 'rb')
Cgrid = pickle.load(f)
f.close()
interpolator = scp.interpolate.RegularGridInterpolator(*Cgrid)

populations = {}

for zscale_value in zscale:
    for logg_value in logg:
        carbon = [interpolator([zscale_value, logg_value, i])[0] for i in np.arange(-2, 3)]
        carbon = np.round(np.array(carbon) + 0.001, 1)
        for carbon_value in carbon:
            for alpha_value in alpha:
                machine_name = 'z{}_a{}_c{}'.format(zscale_value, alpha_value, carbon_value)
                abun = {element: alpha_value for element in ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']}
                abun['C'] = carbon_value
                if machine_name not in populations:
                    populations[machine_name] = {'Y': 0.245, 'zscale': zscale_value, 'abun': abun, 'restrict_logg': [logg_value]}
                else:
                    populations[machine_name]['restrict_logg'] += [logg_value]



