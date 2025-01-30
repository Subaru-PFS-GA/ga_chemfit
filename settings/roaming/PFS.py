############################################################
#                                                          #
#                         PFS PRESET                       #
#                                                          #
#   This preset configures chemfit to analyze PFS spectra. #
#   It defines a set of fitting masks for the key          #
#   parameters (the masks are inherited from Evan's        #
#   original fitter), and defines typical wavelength       #
#   coverage and resultion of the arms of PFS              #
#                                                          #
############################################################


# This is the "aggressive" telluric mask that attempts to completely remove all regions affected by telluric absoprption
telluric_mask = [[6270, 6330], [6860, 6970], [7150, 7400], [7590, 7715], [8100, 8380], [8915, 9910], [10730, 12300], [12450, 12900]]

settings = {
    ### Spectrograph settings ###
    'arms': {
        'blue': {
            'FWHM': 2.07,
            'wl': np.linspace(3800, 6500, 4096),
        },
        'red_lr': {
            'FWHM': 2.63,
            'wl': np.linspace(6300, 9700, 4096),
        },
        'red_mr': {
            'FWHM': 1.368,
            'wl': np.linspace(7100, 8850, 4096),
        },
        'ir': {
            'FWHM': 2.4,
            'wl': np.linspace(9400, 12600, 4096),
        }
    },

    ### Fitting masks ###
    'masks': apply_standard_mask(telluric_mask, original_settings['masks']),
}
