import numpy as np
import itertools

zscale = np.round(np.arange(-7, 4) * 0.1, 1)
alpha = np.round(np.arange(-2, 4) * 0.1, 1)
teff = np.concatenate([np.arange(4500, 6001, 100), np.array([6200, 6400, 6600, 6800, 7000])])
logg = np.round(np.arange(30, 51, 5) * 0.1, 1)

populations = {}
for population in itertools.product(*[zscale, alpha]):
    machine_name = 'z{}_a{}'.format(*population)
    populations[machine_name] = {'Y': 0.25, 'zscale': population[0], 'abun': {element: population[1] for element in ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']}}


