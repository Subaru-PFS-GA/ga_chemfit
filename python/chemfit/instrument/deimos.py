############################################################
#                                                          #
#                     DEIMOS PRESET                        #
#                                                          #
#   This preset adapts chemfit to the Deimos spectrograph  #
#   data. It defines a typical spectrograph arm that can   #
#   be  updated by the user as necessary                   #
#                                                          #
############################################################

import copy
import numpy as np

from .instrument import Instrument

class Deimose(Instrument):
    def __init__(self):
        super().__init__()

    def get_default_settings(self, original_settings):
        return {
            ### Spectrograph settings ###
            'arms': {
                'deimos': {
                    'sigma': 0.583,
                    'wl': np.linspace(6690, 9310, 8192),
                    'priority': 1,
                },
            },

            ### Fitting masks ###
            'masks': copy.deepcopy(original_settings['masks']),
        }
