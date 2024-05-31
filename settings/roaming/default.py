############################################################
#                                                          #
#             DEFAULT CHEMFIT SETTINGS PRESET              #
#                                                          #
#   This preset sets the default synthetic photometry      #
#   parameters, creates a test spectrograph arm, provides  #
#   basic fitting masks and defines standard convergence   #
#   and output parameters                                  #
#                                                          #
############################################################

settings = {
    ### Synthetic photometry ###
    'filter_dir': script_dir + '/bands/',          # Path to the transmission profile directory
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
