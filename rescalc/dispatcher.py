import atlas as basicatlas
from config import settings
import griddef
import glob, os
import numpy as np
import tqdm

import pick

# Main menu options
menu = [
    ['Clean', 'clean'],
    ['Opacity Distribution Functions', 'dfsynthe'],
    ['Structures', 'atlas'],
    ['Spectra', 'synthe'],
    ['Response functions', 'rescalc'],
    ['Prepare calculation', 'run'],
]

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
for filename in glob.glob(rescalc.settings['linelist'] + '/*/fort.*'):
    dest = rundir + '/linelist/' + os.path.relpath(filename, rescalc.settings['linelist'])
    os.makedirs(os.path.dirname(dest), exist_ok = True)
    shutil.copy(filename, dest)
try:
    rescalc.compute_response(rundir + '/response', rundir, rundir + '/linelist')
except Exception as e:
    shutil.copytree(rundir, output + '/incomplete', symlinks = True)
    raise e
shutil.copy('{{}}/response/{{}}_response.pkl'.format(rundir, os.path.basename(output)), '{{}}/response.pkl'.format(output))
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


def dfsynthe():
    tally = {'created': 0, 'successful': 0, 'failed': 0}
    messages = []

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


def synthe():
    tally = {'successful': 0, 'created': 0}

    models = glob.glob('{}/*'.format(settings['ATLAS']['output_dir']))
    for model in tqdm.tqdm(models):
        if os.path.isfile('{}/spectrum.dat'.format(model)):
            tally['successful'] += 1
            continue

        # Calculate new spectra
        f = open('{}/{}.py'.format(settings['scripts_dir'], os.path.basename(model)), 'w')
        f.write(synthe_template.format(output = model, **settings['SYNTHE']))
        f.close()
        tally['created'] += 1

    print('\033[92m{successful} successful\033[0m, \033[93m{created} created\033[0m of {total}'.format(**tally, total = len(griddef.populations) * len(griddef.logg) * len(griddef.teff)))


def rescalc():
    tally = {'successful': 0, 'incomplete': 0, 'created': 0}
    messages = []

    models = glob.glob('{}/*'.format(settings['ATLAS']['output_dir']))
    for model in tqdm.tqdm(models):
        if os.path.isfile('{}/response.pkl'.format(model)):
            tally['successful'] += 1
            continue

        if os.path.isdir('{}/incomplete'.format(model)):
            messages += ['\033[91m{} is incomplete\033[0m'.format(model)]
            tally['incomplete'] += 1
            continue

        # Calculate new spectra
        f = open('{}/{}.py'.format(settings['scripts_dir'], os.path.basename(model)), 'w')
        f.write(rescalc_template.format(output = model, settings = {**settings['SYNTHE'], **settings['rescalc']}, rescalc_path = os.path.dirname(os.path.realpath(__file__))))
        f.close()
        tally['created'] += 1

    for message in messages:
        print(message)
    print('\033[92m{successful} successful\033[0m, \033[91m{incomplete} incomplete\033[0m, \033[93m{created} created\033[0m of {total}'.format(**tally, total = len(griddef.populations) * len(griddef.logg) * len(griddef.teff)))







#############################################################################################


globals()[menu[action][1]]()
