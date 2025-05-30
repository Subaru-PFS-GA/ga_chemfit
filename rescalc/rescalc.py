import numpy as np
import atlas
import subprocess, sys
import os, glob
import shutil
import pickle
import time

settings = {}

def build_linelist(species, output_dir, invert = False):
    # Helper function to convert Kurucz species codes into NELION index
    def code_to_nelion(code):
        code_mol = [101., 106., 107., 108., 606., 607., 608., 707., 708., 808., 112., 113., 114., 812., 813., 814., 116., 120., 816., 820., 821., 822., 823., 103., 104., 105., 109., 115., 117., 121., 122., 123., 124., 125., 126.,106.01, 107.01,108.01,112.01,113.01,114.01,120.01, 111., 119.,10101.01, 817., 824., 825., 826.,10108.,60808.,10106.,60606., 127., 128., 129., 827., 828., 829.,608.01, 408., 508., 815., 10808.,10811.,10812.,10820.,10106.,10107.,10116.,10606.,10607., 10608.,10708.,60816.,61616.,70708.,70808.,80814.,80816.,1010106.,1010107.,1010606.,101010106.,101010114., 614.,60614.,60607., 6060707.,6060607., 839., 840., 857.]
        nelion_mol = [240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 300, 306, 312, 318, 324, 330, 336, 342, 348, 354, 360, 366, 372, 378, 384, 390, 396, 402, 408, 414, 420, 426, 432, 438, 444, 450, 456, 462, 468, 474, 480, 486, 492, 498, 504, 510, 516, 522, 528, 534, 540, 546, 552, 558, 564, 570, 576, 582, 588, 594, 600, 606, 612, 618, 624, 630, 636, 642, 648, 654, 660, 666, 672, 678, 684, 690, 696, 702, 708, 714, 720, 726, 732, 738, 744, 750, 756, 762, 768, 774, 780, 786, 792]

        if code > 100:
            return nelion_mol[np.where(np.array(code_mol) == code)[0][0]]

        Z = int(np.floor(code))
        charge = int(np.round((code - Z) * 100.0))
        nelion = Z * 6 - 6 + (charge + 1)
        if Z > 19 and Z < 29 and charge > 5:
            nelion = 6 * (Z + charge * 10 - 30) - 1
        return nelion

    if os.path.isdir(output_dir):
        return
    os.mkdir(output_dir)

    # Get the bit of the SYNTHE launcher script that prepares the line list
    script = atlas.templates.synthe_control
    start = script.find('# synberg.exe initializes the computation')
    end = script.find('# synthe.exe computes line opacities')
    if start == -1 or end == -1:
        raise ValueError('rescalc no longer compatible with BasicATLAS')
    script = 'cd {output}\n' + script[start:end]

    # Run the script
    C13 = 1 / (settings['C12C13'] + 1)
    C12 = 1 - C13
    C12C13 = 'echo "{} {}" > c12c13.dat'.format(np.log10(C12), np.log10(C13))
    atoms = os.path.realpath(atlas.python_path + '/data/synthe_files/{}.dat'.format(settings['atoms']))
    if not os.path.isfile(atoms):
        raise ValueError('Linelist {} not found'.format(atoms))
    cards = {
        's_files': atlas.python_path + '/data/synthe_files/',
        'd_files': atlas.python_path + '/data/dfsynthe_files/',
        'synthe_suite': atlas.python_path + '/bin/',
        'airorvac': ['VAC', 'AIR'][settings['air_wl']],
        'wlbeg': settings['wl_start'],
        'wlend': settings['wl_end'],
        'resolu': settings['res'],
        'turbv': settings['vturb'],
        'linelist': atoms,
        'C12C13': C12C13,
        'ifnlte': 0,
        'linout': -1,
        'cutoff': 0.0001,
        'ifpred': 1,
        'nread': 0,
        'output': os.path.realpath(output_dir),
    }
    f = open(output_dir + '/packager.com', 'w')
    f.write(script.format(**cards))
    f.close()
    command = 'bash {}/packager.com'.format(output_dir)
    session = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
    stdout, stderr = session.communicate()
    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()
    if stderr != '':
        raise ValueError('Command {} returned an error: {}'.format(command, stderr))
    if stdout != '':
        print(stdout, file = sys.stderr)

    if len(species) > 0 or (not invert):
        # Filter lines to include species we want
        f = open(output_dir + '/fort.12', 'rb')
        dt = np.dtype('i4,i4,f4,i4,f4,f4,f4,f4,f4,i4')
        linelist = np.fromfile(f, dtype = dt, count = -1)
        f.close()
        mask = np.full(len(linelist), False)
        for element in species:
            mask |= linelist['f3'] == code_to_nelion(element)
        if invert:
            mask = ~mask
        f = open(output_dir + '/fort.12', 'wb')
        linelist[mask].tofile(f)
        f.close()

        # Update the total number of lines in fort.93
        f = open(output_dir + '/fort.93', 'rb')
        dt = np.dtype('i4,i4')
        headers = np.fromfile(f, dtype = dt, count = -1)
        f.close()
        headers[0][1] = np.count_nonzero(mask)
        f = open(output_dir + '/fort.93', 'wb')
        headers.tofile(f)
        f.close()

def update_abundances(filename, eheu):
    elements, params = atlas.parse_atlas_abundances(filename, lookbehind = 1, params = ['ABUNDANCE SCALE'])
    settings = atlas.Settings().abun_atlas_to_std(elements, np.log10(params['ABUNDANCE SCALE']))
    for element in eheu:
        if element not in settings['abun']:
            settings['abun'][element] = 0
        settings['abun'][element] += eheu[element]
    elements = atlas.Settings().abun_std_to_atlas(**settings)
    template = atlas.templates.atlas_control
    template = template[template.find('ABUNDANCE SCALE'):template.find('\n', template.find('ABUNDANCE CHANGE 99'))]
    sub = {'element_{}'.format(i): elements[i] for i in range(1, 100)}
    template = template.format(abundance_scale = 10 ** settings['zscale'], **sub)
    f = open(filename, 'r')
    content = f.read()
    f.close()
    start = content.find('ABUNDANCE SCALE')
    end = content.find('\n', content.find('ABUNDANCE CHANGE 99'))
    content = content[:start] + template + content[end:]
    f = open(filename, 'w')
    f.write(content)
    f.close()

def get_opacity_header_size(filename):
    size = 1
    f = open(filename, 'rb')
    dt = np.dtype('i4')
    data = np.fromfile(f, dtype = dt, count = 1)
    for i in range(2):
        while data[0] != 9860:
            size += 1
            data = np.fromfile(f, dtype = dt, count = 1)
        size += 1
        data = np.fromfile(f, dtype = dt, count = 1)
    f.close()
    return size

def load_opacity_table(filename):
    size = get_opacity_header_size(filename)
    f = open(filename, 'rb')
    dt = np.dtype('i4')
    header = np.fromfile(f, dtype = dt, count = size)
    dt = np.dtype(','.join(['f4'] * 99 + ['i4','i4']))
    data = np.fromfile(f, dtype = dt, count = -1)
    dt = np.dtype('i4')
    footer = np.fromfile(f, dtype = dt, count = -1)
    f.close()
    return header, data, footer

def save_opacity_table(filename, header, data, footer):
    f = open(filename, 'wb')
    header.tofile(f)
    data.tofile(f)
    footer.tofile(f)
    f.close()

def synthesize(atmosphere, eheu, linelist, output_dir, update_opacity = [], update_continuum = False):
    start_time = time.time()

    if os.path.isdir(output_dir):
        return
    os.mkdir(output_dir)
    shutil.copy(atmosphere + '/output_summary.out', output_dir + '/')

    # Update abundances
    update_abundances(output_dir + '/output_summary.out', eheu)

    # Convert the ATLAS model into SYNTHE input
    f = open(output_dir + '/output_summary.out', 'r')
    model = f.read()
    f.close()
    # Remove ATLAS turbulent velocity from the output
    model = model.split('\n')
    in_output = False
    for i, line in enumerate(model):
        if line.find('FLXRAD,VCONV,VELSND') != -1:
            in_output = True
            continue
        if line.find('PRADK') != -1:
            in_output = False
            continue
        if in_output:
            model[i] = line[:-40] + ' {:9.3E}'.format(0) + line[-30:]
    model = '\n'.join(model)
    f = open(output_dir + '/output_synthe.out', 'w')
    f.write(atlas.templates.synthe_prependix + model)
    f.close()

    # Retrieve the SYNTHE launcher script, except replace the synbeg calls and linelist creation with the already computed linelist
    script = atlas.templates.synthe_control
    start = script.find('# synberg.exe initializes the computation')
    mid = script.find('# synthe.exe computes line opacities')
    end = script.find('# spectrv.exe computes the synthetic spectrum')
    if start == -1 or end == -1:
        raise ValueError('rescalc no longer compatible with BasicATLAS')
    script = script[:start] + '\nln -s {linelist}/fort.* ./\nrm fort.14\ncp {linelist}/fort.14 ./\n' + script[mid:end]

    # Run SYNTHE
    cards = {
      's_files': atlas.python_path + '/data/synthe_files/',
      'd_files': atlas.python_path + '/data/dfsynthe_files/',
      'synthe_suite': atlas.python_path + '/bin/',
      'synthe_solar': os.path.realpath(output_dir + '/output_synthe.out'),
      'output_dir': os.path.realpath(output_dir),
      'synthe_num': 1,
      'linelist': os.path.realpath(linelist),
    }
    f = open(output_dir + '/launcher.com', 'w')
    f.write(script.format(**cards))
    f.close()
    command = 'bash {}/launcher.com'.format(output_dir)
    session = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
    stdout, stderr = session.communicate()
    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()
    if stderr != '':
        raise ValueError('Command {} returned an error: {}'.format(command, stderr))
    if stdout != '':
        print(stdout, file = sys.stderr)

    # Save the opacity table for future use
    os.rename(original := os.path.realpath(output_dir + '/synthe_1/fort.9'), new := os.path.realpath(output_dir + '/opacity.9'))
    os.symlink(new, original)
    # Save the continuum
    os.rename(original_chem := os.path.realpath(output_dir + '/synthe_1/fort.10'), new_chem := os.path.realpath(output_dir + '/continuum.10'))
    os.symlink(new_chem, original_chem)

    # Update the opacity table if necessary
    if len(update_opacity) > 0:
        header, data, footer = load_opacity_table(new)
        for opacity in update_opacity:
            update = load_opacity_table(opacity[0])[1]
            for i in range(99):
                data['f{}'.format(i)] += opacity[1] * update['f{}'.format(i)]
        os.remove(original)
        save_opacity_table(original, header, data, footer)

    # Update continuum if necessary
    if type(update_continuum) != bool:
        os.remove(original_chem)
        shutil.copy(update_continuum, original_chem)

    # Run SPECTRV and collect the computed spectrum
    script = atlas.templates.synthe_control
    script = 'cd {output_dir}\ncd synthe_{synthe_num}/\n\n' + script[end:]
    f = open(output_dir + '/launcher.com', 'w')
    f.write(script.format(**cards))
    f.close()
    command = 'bash {}/launcher.com'.format(output_dir)
    session = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
    stdout, stderr = session.communicate()
    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()
    if stderr != '':
        raise ValueError('Command {} returned an error: {}'.format(command, stderr))
    if stdout != '':
        print(stdout, file = sys.stderr)

    np.save(output_dir + '/time.npy', time.time() - start_time)

def build_all_linelists(output_dir, elements = False):
    if type(elements) == bool:
        elements = list(settings['elements'].keys())

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for element in elements:
        if element not in settings['higher_order_impact']:
            species = settings['elements'][element]
        else:
            species = sum(list(settings['elements'].values()), [])
        build_linelist(species, output_dir + '/{}'.format(element))

    # Build the null (all-inclusive) line list as well
    build_linelist([], output_dir + '/null', invert = True)

def compute_response(output_dir, atmosphere, linelist, elements = False, silent = True):
    if type(elements) == bool:
        elements = list(settings['elements'].keys())

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.samefile(linelist, settings['linelist']):
        raise ValueError('Cannot use the original linelist. Linelist must be copied first')

    # Verify that the line list matches the settings and update vturb if needed
    for element in os.listdir(linelist):
        f = open('{}/{}/c12c13.dat'.format(linelist, element), 'r')
        C12, C13 = np.array(f.read().strip().split()).astype(float)
        f.close()
        if np.round(10 ** C12 / 10 ** C13, 2) != np.round(settings['C12C13'], 2):
            raise ValueError('Linelist does not match C12/C13 ratio in config')
        f = open('{}/{}/packager.com'.format(linelist, element), 'r')
        content = f.read()
        f.close()
        index = content.rfind('\n', 0, content.find('rgfalllinesnew.exe'))
        atoms = os.path.basename(content[content.rfind('\n', 0, index) + 1:index].split()[-2])
        if atoms != settings['atoms'] + '.dat':
            raise ValueError('Linelist does not match atomic lines in config')
        f = open('{}/{}/fort.93'.format(linelist, element), 'rb')
        dt = np.dtype('i4,i4,i4,i4,i4,i4,f4,i4,i4,i4,i4,i4')
        headers = np.fromfile(f, dtype = dt, count = 1)
        parsed = {'vturb': headers[0][6], 'air_wl': headers[0][3] == 0}
        dt = np.dtype('f8,f8,f8,f8,f8,f4,i4,i4')
        headers = np.fromfile(f, dtype = dt, count = -1)
        parsed['wl_start'] = headers[-1][0]
        parsed['wl_end'] = headers[-1][1]
        parsed['res'] = headers[-1][2]
        assert headers[-1][-1] == 2848
        f.close()
        if parsed['wl_start'] != settings['wl_start'] or parsed['wl_end'] != settings['wl_end'] or parsed['res'] != settings['res'] or parsed['air_wl'] != settings['air_wl']:
            raise ValueError('Linelist does not much wavelength sampling in config')
        if parsed['vturb'] != settings['vturb']:
            f = open('{}/{}/fort.93'.format(linelist, element), 'rb')
            dt = np.dtype('f4,f4')
            headers = np.fromfile(f, dtype = dt, count = -1)
            headers[3][0] = settings['vturb']
            f.close()
            f = open('{}/{}/fort.93'.format(linelist, element), 'wb')
            headers.tofile(f)
            f.close()

    result = {}

    # Get model parameters
    result['meta'] = atlas.meta(atmosphere)
    del result['meta']['type']
    result['meta']['vturb_structure'] = result['meta']['vturb']
    del result['meta']['vturb']
    result['meta']['wl_start'] = settings['wl_start']
    result['meta']['wl_end'] = settings['wl_end']
    result['meta']['res'] = settings['res']
    result['meta']['vturb_spectrum'] = settings['vturb']
    result['meta']['airwl'] = settings['air_wl']
    result['meta']['linelist'] = settings['atoms']
    result['meta']['C12C13'] = settings['C12C13']
    result['meta']['software'] = 'ATLAS9/SYNTHE'

    # Get model structure
    result['structure'] = {}
    structure, units = atlas.read_structure(atmosphere)
    result['structure']['values'] = structure
    result['structure']['units'] = units

    # Compute the null spectrum
    synthesize(atmosphere, {}, linelist + '/null', output_dir + '/null')
    result['null'] = {}
    result['null']['line'], result['null']['cont'] = np.loadtxt(output_dir + '/null/synthe_1/spectrum.asc', skiprows = 2, unpack = True, usecols = [3, 2])
    result['performance'] = {'null': float(np.load(output_dir + '/null/time.npy'))}
    if not silent: print('Computed null spectrum', flush = True)
    # To save space, we are not saving the wavelength grid. It can be recovered (in A) using
    # wl = np.exp(np.arange(np.ceil(np.log(result['meta']['wl_start']) / (lgr := np.log(1.0 + 1.0 / result['meta']['res']))), np.floor(np.log(result['meta']['wl_end']) / lgr) + 1) * lgr) * 10

    abun = [0, np.min(settings['abun']), np.max(settings['abun'])]
    abun = abun + [value for value in settings['abun'] if value not in abun]

    # Compute element responses
    result['response'] = {}
    for element in elements:
        result['response'][element] = {'abun': [], 'spectra': []}
        result['performance'][element] = 0.0
        for i, value in enumerate(abun):
            null_continuum = output_dir + '/null/continuum.10'
            if element not in settings['higher_order_impact']:
                update_continuum = null_continuum
            else:
                update_continuum = False
            update_opacity = [[output_dir + '/{}_{}/opacity.9'.format(element, 0), -1.0], [output_dir + '/null/opacity.9', +1.0]]

            synthesize(atmosphere, {element: value}, linelist + '/{}'.format(element), output_dir + '/{}_{}'.format(element, value), update_opacity = update_opacity, update_continuum = update_continuum)
            if settings['conserve_space'] and value != 0:
                os.remove(output_dir + '/{}_{}/opacity.9'.format(element, value))

            if i == 2:
                if element not in settings['higher_order_impact']:
                    spectrum_low = output_dir + '/{}_{}'.format(element, abun[1])
                    spectrum_high = output_dir + '/{}_{}'.format(element, abun[2])
                else:
                    spectrum_low = output_dir + '/{}_{}_nullcont'.format(element, abun[1])
                    spectrum_high = output_dir + '/{}_{}_nullcont'.format(element, abun[2])
                    synthesize(atmosphere, {element: abun[1]}, linelist + '/{}'.format(element), spectrum_low, update_continuum = null_continuum, update_opacity = update_opacity)
                    if settings['conserve_space'] and value != 0:
                        os.remove('/{}/opacity.9'.format(spectrum_low))
                    synthesize(atmosphere, {element: abun[2]}, linelist + '/{}'.format(element), spectrum_high, update_continuum = null_continuum, update_opacity = update_opacity)
                    if settings['conserve_space'] and value != 0:
                        os.remove('/{}/opacity.9'.format(spectrum_high))
                    result['performance'][element] += np.load(spectrum_low + '/time.npy'.format(element, value))
                    result['performance'][element] += np.load(spectrum_high + '/time.npy'.format(element, value))
                spectrum_low = np.loadtxt(spectrum_low + '/synthe_1/spectrum.asc', skiprows = 2, usecols = [3])
                spectrum_high = np.loadtxt(spectrum_high + '/synthe_1/spectrum.asc', skiprows = 2, usecols = [3])
                result['response'][element]['mask'] = np.abs(spectrum_high - spectrum_low) > settings['threshold']

            spectrum = np.loadtxt(output_dir + '/{}_{}/synthe_1/spectrum.asc'.format(element, value), skiprows = 2, unpack = True, usecols = [3])
            result['performance'][element] += np.load(output_dir + '/{}_{}/time.npy'.format(element, value))
            result['response'][element]['abun'] += [value]
            result['response'][element]['spectra'] += [np.loadtxt(output_dir + '/{}_{}/synthe_1/spectrum.asc'.format(element, value), skiprows = 2, unpack = True, usecols = [3])]
            if not silent: print('Computed {}={}'.format(element, value), flush = True)
        for i in range(len(result['response'][element]['spectra'])):
            result['response'][element]['spectra'][i] = result['response'][element]['spectra'][i][result['response'][element]['mask']]
        result['response'][element]['spectra'] = np.array(result['response'][element]['spectra']).T

        if settings['conserve_space']:
            for spectrum in glob.glob(output_dir + '/{}_*'.format(element)):
                shutil.rmtree(spectrum)

    f = open(output_dir + '/{}_response.pkl'.format(os.path.basename(os.path.dirname(atmosphere + '/'))), 'wb')
    pickle.dump(result, f)
    f.close()

