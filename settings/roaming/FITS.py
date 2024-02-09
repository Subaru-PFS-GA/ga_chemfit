############################################################
#                                                          #
#                BasicATLAS/Palantir PRESET                #
#                                                          #
#   This preset allows chemfit to load synthetic spectra   #
#   from FITS files produced by BasicATLAS and/or palantir #
#   PHOENIX setups                                         #
#                                                          #
############################################################

settings = {
    ### Model grid settings ###
    'griddir': None,    # Model directory must be specified in local settings

    ### Which parameters to fit? ###
    'fit_dof': ['zscale', 'alpha', 'teff', 'logg'],
}
