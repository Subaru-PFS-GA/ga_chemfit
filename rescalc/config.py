import numpy as np

settings = {
    'scripts_dir': '/expanse/lustre/projects/csd835/rgerasim/rescalc/scripts/',
    'runs_dir': '/expanse/lustre/projects/csd835/rgerasim/rescalc/runs/',

    'ODF': {
        'output_dir': '/expanse/lustre/projects/csd835/rgerasim/rescalc/ODF/',
    },

    'ATLAS': {
        'output_dir': '/expanse/lustre/projects/csd835/rgerasim/rescalc/ATLAS/',
        'max_flux_error': 1.0,
        'max_flux_error_derivative': 10.0,
    },

    'SYNTHE': {
        'wl_start': 800,
        'wl_end': 1200,
        'res': 600000,
        'air_wl': True,
        'vturb': 1.5,
    },

    'rescalc': {
        'linelist': '/expanse/lustre/projects/csd835/rgerasim/rescalc/linelist',
        'threshold': 0.01,
        'abun': list(np.round(np.arange(-1.0, 1.01, 0.1), 1)),
        'elements': {
           'Ba': [56, 56.01, 56.02, 56.03, 56.04, 56.05],
           'Eu': [63, 63.01, 63.02, 63.03, 63.04, 63.05],
           'Mg': [12, 12.01, 12.02, 12.03, 12.04, 12.05, 112.0, 812.0, 112.01, 10812.0],
           'Si': [14, 14.01, 14.02, 14.03, 14.04, 14.05, 114.0, 814.0, 114.01, 80814.0, 101010114.0, 614.0, 60614.0],
           'Ca': [20, 20.01, 20.02, 20.03, 20.04, 20.05, 120.0, 820.0, 120.01, 10820.0],
           'Ti': [22, 22.01, 22.02, 22.03, 22.04, 22.05, 122.0], #, 822.0],
           'Ni': [28, 28.01, 28.02, 28.03, 28.04, 28.05, 128.0, 828.0],
           'Na': [11, 11.01, 11.02, 11.03, 11.04, 11.05, 111.0, 10811.0],
           'Mn': [25, 25.01, 25.02, 25.03, 25.04, 25.05, 125.0, 825.0],
           'Co': [27, 27.01, 27.02, 27.03, 27.04, 27.05, 127.0, 827.0],
           'Li': [3, 3.01, 3.02, 3.03, 3.04, 3.05, 103.0],
           'Al': [13, 13.01, 13.02, 13.03, 13.04, 13.05, 113.0, 813.0, 113.01],
           'K': [19, 19.01, 19.02, 19.03, 19.04, 19.05, 119.0],
        },
        # For which elements do we want to model the effect on the lines of other elements and continuum?
        'higher_order_impact': ['Mg', 'Si', 'Na', 'Ca', 'Al'],

        'conserve_space': True,
    },

    'slurm': {
        'queue': 'shared',
        'account': 'csd835',
        'nodes': 1,
        'mem_per_cpu': 2000,
    }
}


