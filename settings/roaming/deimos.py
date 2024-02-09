############################################################
#                                                          #
#                     DEIMOS PRESET                        #
#                                                          #
#   This preset adapts chemfit to the Deimos spectrograph  #
#   data. It defines a blank spectrograph arm and a blank  #
#   fitting mask that can be updated by the user as        #
#   necessary                                              #
#                                                          #
############################################################

settings = {
    ### Spectrograph settings ###
    'arms': {
        'deimos': {
            'FWHM': None,       # Resolution and wavelength sampling must be specified for each observation. No defaults provided
            'wl': None,
            'priority': 1,
        },
    },

    ### Fitting masks ###
    'masks': copy.deepcopy(original_settings['masks']),
}

# The general fitting mask must be specified for each observation
settings['masks']['all']['all'] = None
