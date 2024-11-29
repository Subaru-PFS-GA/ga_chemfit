import atlas as basicatlas
from config import settings
import griddef
import glob, os, shutil
import numpy as np
import tqdm

import pick

# Main menu options
menu = [
    ['Clean', 'clean'],
    ['Opacity Distribution Functions', 'dfsynthe'],
    ['Structures', 'atlas'],
    ['Delete failed structures', 'delete_atlas'],
    ['Line list', 'linelist'],
    ['Spectra', 'rescalc'],
    ['Interpolator', 'interpolator'],
    ['Prepare calculation', 'run'],
]

# Create directories for scripts and runs
if not os.path.isdir(settings['scripts_dir']):
    os.mkdir(settings['scripts_dir'])
if not os.path.isdir(settings['runs_dir']):
    os.mkdir(settings['runs_dir'])

# Anything to clean?
if len(glob.glob('{}/*'.format(settings['scripts_dir'])) + glob.glob('{}/*'.format(settings['runs_dir']))) > 0:
    menu[0][0] = '** ' + menu[0][0] + ' **'

# What are we doing?
action = int(pick.pick([menu[i][0] for i in range(len(menu))], indicator = '=>')[1])


#############################################################################################


dfsynthe_template = """
import atlas

settings = atlas.Settings()
settings.Y = {Y}
settings.zscale = {zscale}
settings.abun = {abun}
atlas.dfsynthe('{output}', settings = settings)
""".strip()

atlas_template = """
import os
import numpy as np
import atlas

settings = atlas.Settings()
settings.Y = {Y}
settings.zscale = {zscale}
settings.abun = {abun}
settings.logg = {logg}
restarts = {restarts}
output = {output}
for teff in {teff}:
    settings.teff = teff
    model_name = output[teff]
    restart = restarts[teff]
    atlas.atlas(model_name, settings = settings, ODF = '{ODF}', restart = restart)
    os.remove('{{}}/odf_1.ros'.format(output[teff]))
    os.remove('{{}}/odf_9.bdf'.format(output[teff]))
""".strip()

synthe_template = """
import os, shutil
import atlas

output = '{output}'
while output[-1] == '/':
    output = output[:-1]
shutil.copytree(output, rundir := (os.path.expandvars('/scratch/$USER/job_$SLURM_JOB_ID/{{}}'.format(os.path.basename(output)))))
atlas.synthe(rundir, {wl_start}, {wl_end}, res = {res}, vturb = {vturb}, air_wl = {air_wl}, progress = False)
shutil.copy('{{}}/spectrum.dat'.format(rundir), '{{}}/spectrum.dat'.format(output))
""".strip()

rescalc_template = """
import numpy as np
import os, shutil, glob
import atlas

import sys
sys.path.append('{rescalc_path}')
import rescalc
rescalc.settings = {settings}

output = '{output}'
while output[-1] == '/':
    output = output[:-1]
shutil.copytree(output, rundir := (os.path.expandvars('/scratch/$USER/job_$SLURM_JOB_ID/{{}}'.format(os.path.basename(output)))))
for filename in glob.glob(rescalc.settings['linelist'] + '/*/fort.*') + glob.glob(rescalc.settings['linelist'] + '/*/packager.com') + glob.glob(rescalc.settings['linelist'] + '/*/c12c13.dat'):
    dest = rundir + '/linelist/' + os.path.relpath(filename, rescalc.settings['linelist'])
    os.makedirs(os.path.dirname(dest), exist_ok = True)
    shutil.copy(filename, dest)
try:
    rescalc.compute_response(rundir + '/response', rundir, rundir + '/linelist')
    shutil.copy('{{}}/response/{{}}_response.pkl'.format(rundir, os.path.basename(output)), '{{}}/response.pkl'.format(output))
    shutil.rmtree(rundir + '/response')
except Exception as e:
    shutil.copytree(rundir, output + '/incomplete', symlinks = True)
    raise e
""".strip()

interpolator_template = """
import numpy as np
import scipy as scp
import pickle
import os
import copy

teff = {teff}
logg = {logg}
spectra = {spectra}
exists = {exists}

teff = np.array(teff); logg = np.array(logg); spectra = np.array(spectra); exists = np.array(exists)

if not np.any(exists):
    raise ValueError('No avialble models in the cell')

f = open(spectra[exists][-1], 'rb')
prototype = pickle.load(f)
f.close()
del prototype['meta']['teff'], prototype['meta']['logg'], prototype['meta']['software']

grid_teff = []
grid_logg = []
grid_cont = []
grid_line = []

for spectrum in spectra[exists]:
    f = open(spectrum, 'rb')
    response = pickle.load(f)
    f.close()
    if response['meta']['software'] == 'interp':
        continue
    grid_teff += [response['meta']['teff']]
    grid_logg += [response['meta']['logg']]
    del response['meta']['teff'], response['meta']['logg'], response['meta']['software']
    assert prototype['meta'] == response['meta']
    grid_cont += [response['null']['cont']]
    grid_line += [response['null']['line']]

interpolator_cont = scp.interpolate.LinearNDInterpolator(np.array([grid_teff, grid_logg]).T, grid_cont)
interpolator_line = scp.interpolate.LinearNDInterpolator(np.array([grid_teff, grid_logg]).T, grid_line)

for model_teff, model_logg, spectrum in zip(teff[~exists], logg[~exists], spectra[~exists]):
    if os.path.isfile(spectrum):
        raise ValueError('{{}} already exists!'.format(spectrum))
    os.makedirs(os.path.dirname(spectrum), exist_ok = True)
    response = {{'meta': copy.deepcopy(prototype['meta']), 'response': {{}}, 'null': {{'line': interpolator_line([model_teff, model_logg])[0], 'cont': interpolator_cont([model_teff, model_logg])[0]}}}}
    response['meta']['software'] = 'interp'
    response['meta']['teff'] = model_teff
    response['meta']['logg'] = model_logg
    f = open(spectrum, 'wb')
    pickle.dump(response, f)
    f.close()
""".strip()

slurm_template = """
#!/bin/bash
#SBATCH --partition={queue}
#SBATCH --account={account}
#SBATCH --job-name="{job_name}"
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cpus}
#SBATCH --mem={mem}M
#SBATCH -t {walltime}:00:00
#SBATCH --output="{output}"
#SBATCH --error="{error}"
#SBATCH --export=ALL
""".strip()


def sparse_sampling(grid_teff, grid_logg, ref_teff, ref_logg, pad_teff = 100, pad_logg = 0.1, sparse_teff = 5, sparse_logg = 5):
    """
    Allow sparse sampling of the teff-logg space away from regions where real stars are likely to be encountered

    The function takes a set of (teff,logg) pairs (references) that represent the expected distribution of real stars. These
    references may be extracted e.g. from model isochrones. Regular sampling (according to the defined model grid) is used
    near those reference points, where "near" is defined as within `pad_logg` in logg and within `pad_teff` in teff.

    Far away from reference points (further than `pad_teff` and `pad_logg`), sparse sampling in teff-logg is used, where only
    every mth and nth grid points are taken in teff and logg. The values of m and n are given in `sparse_teff` and `sparse_logg`

    The function returns all pairs of (teff,logg) values where models are required

    Parameters
    ----------
    grid_teff : array_like
        teff points in the regular (dense) model grid
    grid_logg : array_like
        logg points in the regular (dense) model grid
    ref_teff : array_like
        teff of reference points that define the regions of the teff-logg space where regular sampling is required
    ref_logg : array_like
        Corresponding logg of reference points
    pad_teff : float, optional
        Regular sampling is applied within this distance of reference points in teff
    pad_logg : float, optional
        Regular sampling is applied within this distance of reference points in logg
    sparse_teff: int, optional
        Sparse sampling uses every mth point in the regular teff grid, where m is given by this parameter
    sparse_logg: int, optional
        Sparse sampling uses every nth point in the regular logg grid, where n is given by this parameter

    Returns
    -------
    teff : array_like
        teff of grid points to include in the model grid
    logg : array_like
        Corresponding logg
    """
    full_teff, full_logg = np.meshgrid(grid_teff, grid_logg, indexing = 'ij')
    full_teff = full_teff.flatten(); full_logg = full_logg.flatten()
    include = np.full(len(full_teff), False)

    # Add regular sampling points
    for pad_teff in [-pad_teff, 0, +pad_teff]:
        for pad_logg in [-pad_logg, 0, +pad_logg]:
            for ref in zip(ref_teff + pad_teff, ref_logg + pad_logg):
                if ref[0] < np.min(grid_teff) or ref[0] > np.max(grid_teff):
                    raise ValueError('Grid does not accommodate all reference points in teff')
                if ref[1] < np.min(grid_logg) or ref[1] > np.max(grid_logg):
                    raise ValueError('Grid does not accommodate all reference points in logg')

                include_teff = []
                include_logg = []
                if ref[0] in grid_teff:
                    include_teff += [ref[0]]
                if len(subgrid := grid_teff[grid_teff > ref[0]]) > 0:
                    include_teff += [np.min(subgrid)]
                if len(subgrid := grid_teff[grid_teff < ref[0]]) > 0:
                    include_teff += [np.max(subgrid)]
                if ref[1] in grid_logg:
                    include_logg += [ref[1]]
                if len(subgrid := grid_logg[grid_logg > ref[1]]) > 0:
                    include_logg += [np.min(subgrid)]
                if len(subgrid := grid_logg[grid_logg < ref[1]]) > 0:
                    include_logg += [np.max(subgrid)]

                include |= np.in1d(full_teff, include_teff) & np.in1d(full_logg, include_logg)

    # Add sparse sampling points
    sparse_teff = grid_teff[np.arange(0, len(grid_teff), sparse_teff)]
    sparse_logg = grid_logg[np.arange(0, len(grid_logg), sparse_logg)]
    if np.max(grid_teff) not in sparse_teff:
        sparse_teff = np.concatenate([sparse_teff, [np.max(grid_teff)]])
    if np.max(grid_logg) not in sparse_logg:
        sparse_logg = np.concatenate([sparse_logg, [np.max(grid_logg)]])
    include |= np.in1d(full_teff, sparse_teff) & np.in1d(full_logg, sparse_logg)

    return full_teff[include], full_logg[include]


def dfsynthe():
    tally = {'created': 0, 'successful': 0, 'failed': 0}
    messages = []

    if not os.path.isdir(settings['ODF']['output_dir']):
        os.mkdir(settings['ODF']['output_dir'])

    for population in tqdm.tqdm(griddef.populations):
        # Verify population
        run_dir = '{}/{}/'.format(settings['ODF']['output_dir'], population)
        if os.path.isdir(run_dir):
            if os.path.isfile(run_dir + '/kappa.ros') and basicatlas.validate_run(run_dir, silent = True):
                tally['successful'] += 1
            else:
                tally['failed'] += 1
                messages += ['\033[91m{} failed\033[0m'.format(population)]
            continue

        # Calculate new ODF
        f = open('{}/{}.py'.format(settings['scripts_dir'], population), 'w')
        f.write(dfsynthe_template.format(output = run_dir, **griddef.populations[population]))
        f.close()
        tally['created'] += 1

    for message in messages:
        print(message)
    print('\033[91m{failed} failed\033[0m, \033[92m{successful} successful\033[0m, \033[93m{created} created\033[0m of {total}'.format(**tally, total = len(griddef.populations)))

def input_default(prompt, default):
    result = input('{} ({})\n'.format(prompt, default))
    if result == '':
        result = default
    return result

def run():
    # Get job parameters
    params = {
        'size': int(input_default('Batch size?', 1)),
        'ppn': int(input_default('Cores?', 1)),
        'walltime': int(input_default('Walltime?', 24)),
    }

    # Compile the jobs
    script_count = 0
    run_count = 0
    runs = glob.glob('{}/*.py'.format(settings['scripts_dir']))
    for run in runs:
        if run_count == 0:
            script_count += 1
            f = open('{}/run_{}.tqs'.format(settings['runs_dir'], script_count), 'w')
            f.write(slurm_template.format(job_name = 'run_{}'.format(script_count), cpus = params['ppn'], mem = settings['slurm']['mem_per_cpu'] * params['ppn'], walltime = params['walltime'], output = '{}/run_{}.out'.format(settings['runs_dir'], script_count), error = '{}/run_{}.err'.format(settings['runs_dir'], script_count), **settings['slurm']))
            f.write('\n\n')
        run_count += 1
        f.write('python {}\n'.format(run))
        if run_count == params['size']:
            run_count = 0
    if len(runs) > 0:
        f.close()

    # Build the dispatcher
    f = open('{}/launch.com'.format(settings['runs_dir']), 'w')
    runs = glob.glob('{}/*.tqs'.format(settings['runs_dir']))
    for run in runs:
        f.write('sbatch {}\n'.format(run))
    f.close()

def clean():
    files = glob.glob('{}/*'.format(settings['scripts_dir'])) + glob.glob('{}/*'.format(settings['runs_dir']))
    for file in files:
        os.remove(file)


def atlas():
    tally = {'created': 0, 'converged': 0, 'unconverged': 0, 'failed': 0}
    messages = []

    if not os.path.isdir(settings['ATLAS']['output_dir']):
        os.mkdir(settings['ATLAS']['output_dir'])

    for population in tqdm.tqdm(griddef.populations):
        for logg in griddef.logg:
            # Nominal run parameters
            ODF = '{}/{}/'.format(settings['ODF']['output_dir'], population)
            if not os.path.isdir(ODF):
                messages += ['\033[91m{} misses ODF\033[0m'.format(population)]
                continue
            teff = np.sort(griddef.teff)[::-1]
            output = {}; restarts = {}
            for current, prev in zip(teff, ['auto'] + list(teff[:-1])):
                output[current] = '{}/{}_{}_{}/'.format(settings['ATLAS']['output_dir'], population, current, logg)
                if prev == 'auto':
                    restarts[current] = 'auto'
                else:
                    restarts[current] = '{}/{}_{}_{}/'.format(settings['ATLAS']['output_dir'], population, prev, logg)

            # Evaluate existing structures
            excluded = np.full(len(teff), False)
            for i, model in enumerate(teff):
                if os.path.isdir(output[model]):
                    excluded[i] = True
                    filename = output[model]
                    while filename[-1] == '/':
                        filename = filename[:-1]
                    try:
                        structure = basicatlas.read_structure(filename)[0]
                    except:
                        messages += ['\033[91mCannot load {}\033[0m'.format(os.path.basename(filename))]
                        tally['failed'] += 1
                        continue
                    e1 = np.max(np.abs(structure['flux_error']))
                    e2 = np.max(np.abs(structure['flux_error_derivative']))
                    if e1 > settings['ATLAS']['max_flux_error'] or e2 > settings['ATLAS']['max_flux_error_derivative']:
                        messages += ['\033[91m{} did not converge. Errors {} / {}\033[0m'.format(os.path.basename(filename), e1, e2)]
                        tally['unconverged'] += 1
                        continue
                    tally['converged'] += 1
            teff = teff[~excluded]

            # Calculate new structures
            if len(teff) > 0:
                f = open('{}/{}_{}.py'.format(settings['scripts_dir'], population, logg), 'w')
                f.write(atlas_template.format(**griddef.populations[population], restarts = restarts, output = output, logg = logg, teff = '[' + ', '.join(teff.astype(str)) + ']', ODF = ODF))
                f.close()
                tally['created'] += len(teff)

    for message in messages:
        print(message)
    print('\033[91m{failed} failed\033[0m, \033[96m{unconverged} unconverged\033[0m, \033[92m{converged} converged\033[0m, \033[93m{created} created\033[0m of {total}'.format(**tally, total = len(griddef.populations) * len(griddef.logg) * len(griddef.teff)))


def delete_atlas():
    tally = 0

    models = glob.glob('{}/*'.format(settings['ATLAS']['output_dir']))
    for model in tqdm.tqdm(models):
        try:
            structure = basicatlas.read_structure(model)
        except:
            shutil.rmtree(model)
            tally += 1

    print('\033[91m{} deleted\033[0m'.format(tally))

def linelist():
    import rescalc as rescalc_module

    if os.path.isdir(settings['rescalc']['linelist']):
        print('\033[91mExisting linelist found in {}\033[0m'.format(settings['rescalc']['linelist']))
        return
    os.mkdir(settings['rescalc']['linelist'])

    C12C13 = np.unique(list(settings['SYNTHE']['C12C13'].values()))

    for value in C12C13:
        if os.path.isdir(linedir := (settings['rescalc']['linelist'] + '/C12C13_{}'.format(value))):
            continue
        rescalc_module.settings = {**settings['SYNTHE'], **settings['rescalc']}
        rescalc_module.settings['C12C13'] = value
        rescalc_module.settings['vturb'] = 2.0 # Dummy placeholder, VTURB not used in linelist compilation
        rescalc_module.build_all_linelists(linedir)
        print('C12C13={} completed'.format(value))

    return


def rescalc():
    tally = {'successful': 0, 'incomplete': 0, 'excluded': 0, 'created': 0}
    messages = []

    if type(settings['sparse_sampling']['ref_teff_logg']) is not bool:
        ref_teff, ref_logg = np.load(settings['sparse_sampling']['ref_teff_logg'])
        included = sparse_sampling(griddef.teff, griddef.logg, ref_teff, ref_logg, pad_teff = settings['sparse_sampling']['pad_teff'], pad_logg = settings['sparse_sampling']['pad_logg'], sparse_teff = settings['sparse_sampling']['sparse_teff'], sparse_logg = settings['sparse_sampling']['sparse_logg'])

    models = glob.glob('{}/*'.format(settings['ATLAS']['output_dir']))
    for model in tqdm.tqdm(models):
        if os.path.isfile('{}/response.pkl'.format(model)):
            tally['successful'] += 1
            continue

        if os.path.isdir('{}/incomplete'.format(model)):
            messages += ['\033[91m{} is incomplete\033[0m'.format(model)]
            tally['incomplete'] += 1
            continue

        if type(settings['sparse_sampling']['ref_teff_logg']) is not bool:
            teff, logg = model.split('_')[-2:]
            if not np.any((included[0] == float(teff)) & (included[1] == float(logg))):
                tally['excluded'] += 1
                continue

        # Calculate new spectra
        f = open('{}/{}.py'.format(settings['scripts_dir'], os.path.basename(model)), 'w')
        rescalc_settings = {**settings['SYNTHE'], **settings['rescalc']}
        teff, logg = model.split('_')[-2:]
        if float(logg) not in settings['SYNTHE']['C12C13']:
            raise ValueError('C12/C13 ratio for logg={} not defined'.format(logg))
        C12C13 = settings['SYNTHE']['C12C13'][float(logg)]
        rescalc_settings['C12C13'] = C12C13
        rescalc_settings['linelist'] = os.path.realpath(rescalc_settings['linelist'] + '/C12C13_{}'.format(C12C13))
        if not os.path.isdir(rescalc_settings['linelist']):
            raise ValueError('Linelist {} not found'.format(rescalc_settings['linelist']))
        rescalc_settings['vturb'] = settings['SYNTHE']['vturb'][float(logg)]
        f.write(rescalc_template.format(output = model, settings = rescalc_settings, rescalc_path = os.path.dirname(os.path.realpath(__file__))))
        f.close()
        tally['created'] += 1

    for message in messages:
        print(message)
    print('\033[92m{successful} successful\033[0m, \033[96m{excluded} excluded\033[0m, \033[91m{incomplete} incomplete\033[0m, \033[93m{created} created\033[0m of {total}'.format(**tally, total = len(griddef.populations) * len(griddef.logg) * len(griddef.teff)))


def interpolator():
    tally = {'full': 0, 'empty': 0, 'created': 0, 'excluded': 0}
    messages = []

    # Identify cells of the sparse grid
    sparse_grid = sparse_sampling(griddef.teff, griddef.logg, np.array([]), np.array([]), pad_teff = settings['sparse_sampling']['pad_teff'], pad_logg = settings['sparse_sampling']['pad_logg'], sparse_teff = settings['sparse_sampling']['sparse_teff'], sparse_logg = settings['sparse_sampling']['sparse_logg'])
    teff = np.unique(sparse_grid[0])
    logg = np.unique(sparse_grid[1])
    assert len(teff) * len(logg) == len(sparse_grid[0])
    cells = []
    teff_pad = np.min(np.abs(np.diff(griddef.teff))) / 2
    logg_pad = np.min(np.abs(np.diff(griddef.logg))) / 2
    for teff_left, teff_right in zip(teff[:-1], teff[1:]):
        for logg_left, logg_right in zip(logg[:-1], logg[1:]):
            cells += [[teff_left - teff_pad, teff_right + teff_pad, logg_left - logg_pad, logg_right + logg_pad]]

    # Get a list of all models expected in the full grid
    teff, logg = np.meshgrid(griddef.teff, griddef.logg, indexing = 'ij')
    teff = teff.flatten(); logg = logg.flatten()

    # Create an interpolation call for every cell that has missing models
    for population in tqdm.tqdm(griddef.populations):
        added = np.full(len(teff), False)
        for i, cell in enumerate(cells):
            cell_models = (teff > cell[0]) & (teff < cell[1]) & (logg > cell[2]) & (logg < cell[3])
            cell_spectra = ['{}/{}_{}_{}/response.pkl'.format(settings['ATLAS']['output_dir'], population, *model) for model in zip(teff[cell_models], logg[cell_models])]
            exists = np.array([os.path.isfile(spectrum) for spectrum in cell_spectra])
            if np.all(exists):
                tally['full'] += 1
                continue

            if not np.any(exists):
                tally['empty'] += 1
                messages += ['\033[91m{}:{}-{}:{}-{} is empty\033[0m'.format(population, *cell)]
                continue

            # Do not compute spectra already assigned to other cells
            cell_spectra = np.array(cell_spectra)[~added[cell_models]].tolist()
            exists = exists[~added[cell_models]]
            if len(cell_spectra) == 0:
                tally['excluded'] += 1
                continue

            f = open('{}/{}_cell{}.py'.format(settings['scripts_dir'], population, i), 'w')
            f.write(interpolator_template.format(teff = list(teff[cell_models & (~added)]), logg = list(logg[cell_models & (~added)]), spectra = cell_spectra, exists = list(exists)))
            f.close()
            cell_models[cell_models & (~added)] &= (~exists)
            added |= cell_models & (~added)
            tally['created'] += 1

    for message in messages:
        print(message)
    print('\033[91m{empty} empty\033[0m, \033[92m{full} full\033[0m, \033[96m{excluded} excluded\033[0m, \033[93m{created} created\033[0m'.format(**tally))





#############################################################################################


globals()[menu[action][1]]()
