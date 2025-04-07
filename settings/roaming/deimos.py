############################################################
#                                                          #
#                     DEIMOS PRESET                        #
#                                                          #
#   This preset adapts chemfit to the Deimos spectrograph  #
#   data. It defines a typical spectrograph arm that can   #
#   be  updated by the user as necessary                   #
#                                                          #
############################################################

# This is the "aggressive" telluric mask that attempts to completely remove all regions affected by telluric absoprption
telluric_mask = [[6270, 6330], [6860, 6970], [7150, 7400], [7590, 7715], [8100, 8380], [8915, 9910], [10730, 12300], [12450, 12900]]

settings = {
    ### Spectrograph settings ###
    'arms': {
        'deimos': {
            'sigma': 0.583,
            'wl': np.linspace(6690, 9310, 8192),
        },
    },

    ### Fitting masks ###
    'masks': apply_standard_mask(telluric_mask, original_settings['masks']),
}
