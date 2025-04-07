############################################################
#                                                          #
#                 LIVE SYNTHESIS PRESET                    #
#                                                          #
#  This preset configures chemfit to use runtime spectral  #
#  synthesis instead of interpolating an existing model    #
#  grid. This mode of operation incurs high computational  #
#  demand. It is recommended to use a reduced molecular    #
#  line list, e.g. by excluding TiO and water. Spectral    #
#  synthesis is carried out with BasicATLAS which must be  #
#  installed.                                              #
#                                                          #
############################################################

import pickle
import os, shutil
import zipfile
import sys
import random, string
import copy
import scipy as scp
import concurrent.futures

# This is the "defective" mask that removes parts of the spectrum where the models do not match the spectra of 
# Arcturus and the Sun well
defective_mask = [[3800, 4000], [4006, 4012], [4065, 4075], [4093, 4110], [4140, 4165], [4170, 4180], [4205, 4220],
                 [4285, 4300], [4335, 4345], [4375, 4387], [4700, 4715], [4775, 4790], [4855, 4865], [5055, 5065],
                 [5145, 5160], [5203, 5213], [5885, 5900], [6355, 6365], [6555, 6570], [7175, 7195], [7890, 7900],
                 [8320, 8330], [8490, 8505], [8530, 8555], [8650, 8672]]

settings = {
    ### All directories are specified in local settings ###
    'structures_dir': original_settings['structures_dir'],
    'structures_index': original_settings['structures_index'],
    'rescalc_path': original_settings['rescalc_path'],
    'scratch': original_settings['scratch'],

    ### Which elements to fit using live synthesis? [min, max, step] ###
    'elements': {'O': [-0.5, 0.5, 0.1], 'C': [-0.5, 0.5, 0.1], 'Fe': [-1.0, 1.0, 0.1]},

    ### Binning factor for new spectra ###
    'binning': 10,

    ### Suppress status messages? ###
    'silent': False,

    ### RESCALC settings ###
    'rescalc': {
        'threshold': 0.01,
        'abun': list(np.round(np.arange(-1.0, 1.01, 0.1), 1)),
        'elements': {},
        'higher_order_impact': ['Mg', 'Si', 'Na', 'Ca', 'Al'],
        'conserve_space': True,
        'atoms': 'BasicATLAS',
        'air_wl': True,
        'wl_start': 350,
        'wl_end': 1300,
        'res': 300000,
    },

    ### Work directory name for this session. `False` to randomize ###
    'workdir': False,

    ### Enable parallel computation of models (inherited from local settings) ###
    'parallel': original_settings['parallel'],

    ### Which parameters to fit? The abundances of individual elements enabled through live synthesis ###
    ### are not listed here as they will be added automatically ###
    'fit_dof': ['zscale', 'alpha', 'teff', 'logg', 'redshift'],

    ### Virtual grid bounds ###
    'virtual_dof': {'redshift': [-300, 300]},

    ### Default initial guesses ###
    'default_initial': {'redshift': 0.0},

    ### Fitting masks ###
    'masks': apply_standard_mask(defective_mask, original_settings['masks']),
}

# Import RESCALC
sys.path.append(settings['rescalc_path'])
import rescalc

# Relationship between C12/C13 isotope ratio and surface gravity from Kirby+2015
C12C13_kirby_2015 = lambda logg: np.where(logg > 2.7, 50, np.where(logg <= 2.0, 6, 63 * logg - 120))
# Relationship between VTURB and LOGG based on Deimos spectra from Gerasimov+2025
VTURB_LOGG = lambda logg: 2.792 * np.exp(-0.241 * logg -0.118)

# Work directory for live synthesis
livesyn_workdir = False

def notify(message):
    """Issue a status notification
    
    Wrapper for the print() function that respects the `silent` setting
    
    Parameters
    ----------
    message : str
        Message to print
    """
    if not settings['silent']:
        print(message)

def init_livesyn():
    """Initialize the work directory for the session
    
    The work directory is where the model spectra created with real-time spectral synthesis are
    stored. Since there is likely little overlap in the models required to fit different spectra,
    the contents of the work directory are typically discarded at the end of the session. For instance,
    when running live synthesis fitting on a compute cluster, the work directory may be created in the
    local storage of the node that will be automatically discarded once the job is complete. Note that
    deletion of the work directory must be carried out by the system or the user: it will not be done
    by chemfit

    Since the contents of the work directory are often disposable, the work directory can have a random
    name. If `settings['workdir']` is set to `False`, the work directory will be created by this function
    in the scratch directory (`settings['scratch']`) with a random name

    If it is instead preferred to reuse the same models in multiple sessions, a previously used work
    directory (or an empty directory) can be provided in `settings['workdir']`, and all new models will be
    stored there instead, while the models that already exist in the directory will be made available to
    the fitter
    """
    global livesyn_workdir

    # Generate the work directory for this session
    if type(settings['workdir']) == bool:
        livesyn_workdir = '{}/livesyn_{}/'.format(settings['scratch'], ''.join(random.choices(string.ascii_letters + string.digits, k = 20)))
    else:
        livesyn_workdir = settings['workdir']
    if not os.path.isdir(livesyn_workdir):
        os.mkdir(livesyn_workdir)
        notify('LIVESYN session started in {}'.format(livesyn_workdir))
    else:
        notify('LIVESYN session resumed in {}'.format(livesyn_workdir))


def build_linelists():
    """Build the line lists for the fitting session

    The line lists are going to be the same for every fitting session so long as the RESCALC configuration
    is unchanged. For this reason, it is only necessary to precompute them once. This function will check
    for an existing line list in `settings['scratch'] + '/linelist'`, and use RESCALC to generate it if it
    does not exist. The line lists are generated for each gravity in the structures grid due to distinct
    C12/C13 isotope ratios and turbulent velocities (turbulent velocity strictly speaking does not change
    the line list; however, it changes the SYNTHE configuration file, fort.93, and that file is created
    at the same time as the line list)
    """
    global livesyn_grid

    # Check if linelist exists
    linelist_dir = settings['scratch'] + '/linelist'
    if os.path.isdir(linelist_dir):
        return
    os.mkdir(linelist_dir)
    notify('Generating linelist...')

    for logg in livesyn_grid['logg']:
        rescalc.settings = {
            **settings['rescalc'],
            'C12C13': np.round(C12C13_kirby_2015(logg), 2),
            'vturb': np.round(VTURB_LOGG(logg), 2),
        }
        linelist = '{}/logg_{}'.format(linelist_dir, logg)
        rescalc.build_linelist([], linelist, invert = True)
        notify('Linelist for logg={} done'.format(logg))


def select_structure(teff, logg, zscale, eheu):
    """Choose the most appropriate structure from the structures grid for a given chemical composition

    The structure is chosen in three steps. First, we require the desired metallicity, as well as
    temperature and gravity, to be matched to the structure parameters exactly. If no such structures
    are available, an exception is raised. Second, out of the structures with matching metallicities,
    temperatures and gravities, we choose the one with the closest C/O ratio to the desired chemistry.
    Finally, if there are multiple structures with equally close C/O ratios, we choose the one with the
    closest Mg abundance
    
    Parameters
    ----------
    teff : float
        Desired effective temperature
    logg : float
        Desired surface gravity
    zscale : float
        Desired metallicity, [M/H]
    eheu : dict
        Dictionary of desired abundances, [A/M]
    
    Returns
    -------
    structure : str
        Unique identifier of the chosen structure
    alpha : float
        Alpha-enhancement of the chosen structure
    carbon: float
        Carbon abundance, [C/Fe], of the chosen structure
    
    Raises
    ------
    ValueError
        [description]
    """
    global livesyn_grid, livesyn_structures_index

    # Determine the requested alpha-enhancement using Mg as an indicator
    if 'Mg' not in eheu:
        eheu['Mg'] = 0
    alpha_required = np.round(eheu['Mg'], 2)

    # teff, logg and zscale must match the structure exactly
    if (teff not in livesyn_grid['teff']) or (logg not in livesyn_grid['logg']) or (zscale not in livesyn_grid['zscale']):
        raise ValueError('No structure available for (TEFF,LOGG,ZSCALE)=({},{},{})'.format(teff, logg, zscale))

    # Choose alpha and carbon to first get as close as possible to the desired C/O ratio, then Mg abundance
    if 'O' not in eheu:
        eheu['O'] = 0
    if 'C' not in eheu:
        eheu['C'] = 0
    CO_required = eheu['C'] - eheu['O']
    candidates = np.array([structure for structure in livesyn_structures_index if (livesyn_structures_index[structure]['teff'] == teff) and (livesyn_structures_index[structure]['logg'] == logg) and (livesyn_structures_index[structure]['zscale'] == zscale)])
    CO_available, alpha_available = np.array([[livesyn_structures_index[structure]['carbon'] - livesyn_structures_index[structure]['alpha'], livesyn_structures_index[structure]['alpha']] for structure in candidates]).T
    CO_diff = np.abs(CO_available - CO_required); alpha_diff = np.abs(alpha_available - alpha_required)
    index = np.where((CO_diff == np.min(CO_diff)) & (alpha_diff == np.min(alpha_diff[CO_diff == np.min(CO_diff)])))[0][0]

    return candidates[index], alpha_available[index], CO_available[index] + alpha_available[index]


def synthesize(output, structure, linelist, eheu):
    global livesyn_structures_index

    # Prepare the model directory
    if os.path.isdir(output):
        return
    os.mkdir(output)

    # Extract the structure from the grid
    if structure not in livesyn_structures_index:
        raise ValueError('Unknown structure {}'.format(structure))
    z = zipfile.ZipFile(settings['structures_dir'] + '/' + structure[:structure.find('/')], 'r')
    structure_files = [filename for filename in z.namelist() if filename.startswith(structure[structure.find('/') + 1:]) and filename.endswith('.out')]
    for filename in structure_files:
        data = z.read(filename)
        f = open('{}/{}'.format(output, filename.split('/')[-1]), 'wb')
        f.write(data)
        f.close()
    z.close()

    # Run spectral synthesis
    rescalc.synthesize(output, eheu, linelist, output + '/model')

def generate_spectrum(model):
    global livesyn_workdir

    # Do not include O in live synthesis alpha
    alpha_elements = ['Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']

    # Select the structure and determine abundance offsets
    eheu = {}
    for element in alpha_elements:
        eheu[element] = model['alpha']
    for element in settings['elements']:
        if element not in eheu:
            eheu[element] = 0
        eheu[element] += model['live_{}'.format(element)]
    eheu = {element: np.round(eheu[element], 2) for element in eheu}
    structure, structure_alpha, structure_carbon = select_structure(model['teff'], model['logg'], model['zscale'], eheu)
    if 'O' not in eheu:
        eheu['O'] = 0
    for element in alpha_elements + ['O']:
        eheu[element] -= structure_alpha
    if 'C' not in eheu:
        eheu['C'] = 0
    eheu['C'] -= structure_carbon
    model_name = 't{teff:.2f}_g{logg:.2f}_z{zscale:.2f}_a{alpha:.2f}'.format(**model)
    for element in settings['elements']:
        model_name += '_{}{:.2f}'.format(element, model['live_{}'.format(element)])
    model_name = model_name.replace('-0.00', '0.00')
    eheu = {element: np.round(eheu[element], 2) for element in eheu}

    # Run the synthesis
    if type(livesyn_workdir) is bool:
        init_livesyn()
    model_dir = '{}/{}'.format(livesyn_workdir, model_name)
    offsets = copy.deepcopy(eheu)
    alpha_offsets = []
    for element in alpha_elements:
        if element in eheu:
            alpha_offsets += [eheu[element]]
        else:
            alpha_offsets += [0]
    if np.all(np.array(alpha_offsets) == alpha_offsets[0]):
        offsets = {element: offsets[element] for element in offsets if element not in alpha_elements}
        offsets['"a"'] = alpha_offsets[0]
    notify('Calculating spectrum for {} | {}'.format(model_name, offsets))
    notify('Structure: {}'.format(structure))
    synthesize(model_dir, structure, settings['scratch'] + '/linelist/logg_{}'.format(model['logg']), eheu)

    # Parse results and bin them down
    wl, flux = np.loadtxt(model_dir + '/model/synthe_1/spectrum.asc', skiprows = 2, unpack = True, usecols = [0, 1])
    wl = wl[:(len(wl) // settings['binning']) * settings['binning']].reshape(-1, settings['binning']).mean(axis = 1)
    flux = flux[:(len(flux) // settings['binning']) * settings['binning']].reshape(-1, settings['binning']).mean(axis = 1)
    np.save(model_dir + '.npy', np.array([wl, flux]))
    shutil.rmtree(model_dir)

def generate_spectra(params):
    build_linelists()
    if len(params) == 1:
        return generate_spectrum(params[0])
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(generate_spectrum, params)

def read_grid_dimensions():
    global livesyn_grid

    grid = {param: livesyn_grid[param] for param in ['zscale', 'alpha', 'teff', 'logg']}
    for element in settings['elements']:
        grid['live_{}'.format(element)] = np.round(np.arange(settings['elements'][element][0], settings['elements'][element][1] + 0.0001, settings['elements'][element][2]), 2)
    return grid

def read_grid_model(params, grid, batch = False):
    global livesyn_workdir

    if type(livesyn_workdir) is bool:
        init_livesyn()

    # Parallel mode
    if batch:
        request = []
        for instance in params:
            model_name = 't{teff:.2f}_g{logg:.2f}_z{zscale:.2f}_a{alpha:.2f}'.format(**instance)
            for element in settings['elements']:
                model_name += '_{}{:.2f}'.format(element, instance['live_{}'.format(element)])
            model_name = model_name.replace('-0.00', '0.00')
            model = '{}/{}.npy'.format(livesyn_workdir, model_name)
            if not os.path.isfile(model):
                request += [instance]
        notify('Pre-computing {} new models'.format(len(request)))
        generate_spectra(request)
        return

    # Regular mode
    model_name = 't{teff:.2f}_g{logg:.2f}_z{zscale:.2f}_a{alpha:.2f}'.format(**params)
    for element in settings['elements']:
        model_name += '_{}{:.2f}'.format(element, params['live_{}'.format(element)])
    model_name = model_name.replace('-0.00', '0.00')
    model = '{}/{}.npy'.format(livesyn_workdir, model_name)
    if os.path.isfile(model):
        wl, flux = np.load(model)
    else:
        generate_spectra([params])
        wl, flux = np.load(model)

    # Trim the spectrum on both sides to make sure we can do redshift corrections
    wl_range = [np.min(wl * (1 + settings['virtual_dof']['redshift'][1] * 1e3 / scp.constants.c)), np.max(wl * (1 + settings['virtual_dof']['redshift'][0] * 1e3 / scp.constants.c))]
    mask_left = wl < wl_range[0]; mask_right = wl > wl_range[1]; mask_in = (~mask_left) & (~mask_right)
    meta = {'left': [wl[mask_left], flux[mask_left]], 'right': [wl[mask_right], flux[mask_right]]}

    return wl, flux, meta

def preprocess_grid_model(wl, flux, params, meta):
    # Restore the full (untrimmed) spectrum
    wl_full = np.concatenate([meta['left'][0], wl, meta['right'][0]])
    flux_full = np.concatenate([meta['left'][1], flux, meta['right'][1]])

    # Apply the redshift
    wl_redshifted = wl_full * (1 + params['redshift'] * 1e3 / scp.constants.c)

    # Re-interpolate back into the original wavelength grid
    flux = np.interp(wl, wl_redshifted, flux_full)
    return flux

# Load the structure index
f = open(settings['structures_index'], 'rb')
livesyn_structures_index = pickle.load(f)
f.close()

# Extract the grid axes
livesyn_grid = {}
for structure in livesyn_structures_index:
    for param in livesyn_structures_index[structure]:
        if param not in livesyn_grid:
            livesyn_grid[param] = []
        if livesyn_structures_index[structure][param] not in livesyn_grid[param]:
            livesyn_grid[param] += [livesyn_structures_index[structure][param]]
for param in livesyn_grid:
    livesyn_grid[param] = np.unique(livesyn_grid[param])

# Add live synthesis elements to the list of degrees of freedom
settings['fit_dof'] = settings['fit_dof'] + ['live_{}'.format(element) for element in settings['elements']]

# Provide default initial guesses for live synthesis elements
settings['default_initial'] = {**settings['default_initial'], **{'live_{}'.format(element): np.round((settings['elements'][element][0] + settings['elements'][element][1]) / 2.0, 2) for element in settings['elements']}}