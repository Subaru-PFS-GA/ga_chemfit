import numpy as np
import itertools

zscale = np.round(np.arange(-1.5, -0.5 + 0.001, 0.1), 2)
alpha = np.round(np.arange(-0.8, 0.2 + 0.001, 0.1), 2)
logg = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
teff = np.array([3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500])
carbon = np.round(np.arange(-0.5, 0.5 + 0.001, 0.1), 2)

populations = {}
for population in itertools.product(*[zscale, alpha, carbon]):
    machine_name = 'z{}_a{}_c{}'.format(*population)
    abun = {element: population[1] for element in ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']}
    abun['C'] = population[2]
    populations[machine_name] = {'Y': 0.25, 'zscale': population[0], 'abun': abun}


