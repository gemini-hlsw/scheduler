#!/usr/bin/env python
# coding: utf-8

# Code for manipulating ODB Extractor json files and initial sequence atom definitions
# Bryan Miller
# 2021-11-24

import os
import json
import gzip
import numpy as np

from openpyxl import Workbook
from openpyxl import load_workbook


fpuinst = {'GNIRS': 'instrument:slitWidth', 'GMOS-N': 'instrument:fpu'}


def guide_state(step):
    """Determine if guiding is on/off for a sequence step"""
    # One could also extract the guider used if needed
    guiding = False
    for key in list(step.keys()):
        if 'guideWith' in key:
            if step[key] == 'guide':
                guiding = True
                break
    return guiding


def select_qastate(states):
    """Return the qastate based on precedence

        states: list of observe states from the ODB extractor obsLog
    """
    qastate = ''

    # Precedence order for observation classes.
    qastate_order = ['NONE', 'UNDEFINED', 'FAIL', 'USABLE', 'PASS']

    # Set the qastate for the entire observation based on precedence
    for state in qastate_order:
        if state in states:
            qastate = state
            break

    return qastate

# --------------


def select_obsclass(classes):
    """Return the obsclass based on precedence

        classes: list of observe classes from the ODB extractor
    """
    obsclass = ''

    # Precedence order for observation classes.
    obsclass_order = ['SCIENCE', 'PROGCAL', 'PARTNERCAL', 'ACQ', 'ACQCAL', 'DAYCAL']

    # Set the obsclass for the entire observation based on obsclass precedence
    for oclass in obsclass_order:
        if oclass in classes:
            obsclass = oclass
            break

    return obsclass

# --------------


def findatoms(observation):
    '''Analyze a json observing sequence from the ODB and define atoms.'''

    classes = []
    qastates = []
    guiding = []
    atoms = []
    natom = 0
    nabba = 0

    # Make dictionary out of obslog to get QA state
    obslog = {}
    datalabels = []
    if 'obsLog' in observation.keys():
        for log_entry in observation['obsLog']:
            obslog[log_entry['label']] = {'qaState': log_entry['qaState'], 'filename': log_entry['filename']}
        datalabels = list(obslog.keys())

    # Sequence analysis
    sequence = observation['sequence']
    nsteps = len(sequence)

    for ii, step in enumerate(sequence):
        nextatom = False
        qoffsets = False

        datalab = step['observe:dataLabel']
        if datalab in datalabels:
            qastate = obslog[datalab]['qaState']
        else:
            qastate = 'NONE'
        qastates.append(qastate.upper())

        observe_class = step['observe:class']
        classes.append(observe_class.upper())

        guiding.append(guide_state(step))

        exptime = float(step['observe:exposureTime'])
        inst = step['instrument:instrument']
        #     print(inst, fpuinst[inst])
        fpu = step[fpuinst[inst]]
        if 'instrument:filter' in step.keys():
            filter = step["instrument:filter"]
        else:
            filter = 'None'
        wavelength = float(step['instrument:observingWavelength'])
        step_time = step['totalTime'] / 1000.

        if 'GMOS' in inst:
            coadds = 1
        else:
            coadds = int(step['observe:coadds'])

        disperser = step['instrument:disperser']
        if 'telescope:p' in step.keys():
            p = float(step['telescope:p'])
        else:
            p = 0.0
        if 'telescope:q' in step.keys():
            q = float(step['telescope:q'])
            qoffsets = True
        else:
            q = 0.0

            # Any wavelength/filter change is a new atom
        if ii == 0 or (ii > 0 and wavelength != float(sequence[ii - 1]['instrument:observingWavelength'])):
            nextatom = True
            print('Atom for wavelength')

        #         if observe_class == 'science':

        # AB
        # ABBA
        if qoffsets and nsteps >= 4 and nsteps - ii > 3 and nabba == 0:
            #             print(q, sequence[ii+1]['telescope:q'], sequence[ii+2]['telescope:q'], sequence[ii+3]['telescope:q'])
            if q == float(sequence[ii + 3]['telescope:q']) and q != float(sequence[ii + 1]['telescope:q']) and float(
                    sequence[ii + 1]['telescope:q']) == float(sequence[ii + 2]['telescope:q']):
                nabba = 3
                nextatom = True
                print('Atom for ABBA')
        else:
            nabba -= 1

        if nextatom:
            # Get class and qastate for previous atom
            if natom > 0:
                print(qastates)
                atoms[-1]['qastate'] = select_qastate(qastates)
                print(classes)
                atoms[-1]['class'] = select_obsclass(classes)
                print(guiding)
                atoms[-1]['guiding'] = any(guiding)

            # New atom
            natom += 1
            atoms.append({'id': natom, 'exec_time': 0.0, 'prog_time': 0.0, 'part_time': 0.0,
                          'class': 'NONE', 'qastate': 'NONE', 'guiding': False})
            classes = []
            guiding = []

        atoms[-1]['exec_time'] += step_time

        atomlabel = natom
        if 'partnerCal' in observe_class:
            atomlabel *= 10
            atoms[-1]['part_time'] += step_time
        else:
            atoms[-1]['prog_time'] += step_time

        print('{:22} {:12} {:7.2f} {:3d} {:10} {:15} {:12} {:12} {:7.4f} {:5.2f} {:5.2f} {:3d}'.format(datalab,
                                                                                                       observe_class,
                                                                                                       exptime, coadds,
                                                                                                       inst, fpu,
                                                                                                       filter,
                                                                                                       disperser,
                                                                                                       wavelength, p, q,
                                                                                                       atomlabel))
    #     print(atoms)
    # Get class/state for last atom
    if natom > 0:
        print(qastates)
        atoms[-1]['qastate'] = select_qastate(qastates)
        print(classes)
        atoms[-1]['class'] = select_obsclass(classes)
        print(guiding)
        atoms[-1]['guiding'] = any(guiding)

    return atoms


def group_proc(group):
    '''Process observations within groups'''

    obsnum = []
    for item in list(group.keys()):
        obsid = ''
        if 'OBSERVATION' in item:
            #         obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
            obsid = group[item]['observationId']
            obsnum.append(int(item.split('-')[1]))
    #             print(f" \t {item, obsnum[-1], obsid}")
    #         else:
    #             print(item, group[item])

    if len(obsnum) > 0:
        isrt = np.argsort(obsnum)
        for ii in isrt:
            obs_program_used = 0.0
            obs_partner_used = 0.0
            item = 'OBSERVATION_BASIC-' + str(obsnum[ii])
            #     obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
            obsid = group[item]['observationId']
            print(f" \t {obsnum[ii], obsid}")
            # Atoms in each sequence
            atoms = findatoms(group[item])
            # Summary of atoms
            classes = []
            qastates = []
            for atom in atoms:
                print('Atom ', atom['id'])
                for key in atom.keys():
                    print(f" \t {key}: {atom[key]}")
                    if key == 'class':
                        classes.append(atom[key])
                    if key == 'qastate':
                        qastates.append(atom[key])
                        if atom[key].upper() == 'PASS':
                            obs_program_used += atom['prog_time']
                            obs_partner_used += atom['part_time']
            obsclass = select_obsclass(classes)
            print(f"Obsclass: {obsclass}")
            obs_qastate = select_qastate(qastates)
            print(f"QAstate (atoms): {obs_qastate}")
            print(f"qaState (ODB): {group[item]['qaState']}")
            if group[item]['qaState'].upper() == 'PASS':
                if group[item]['obsClass'] in ['science', 'progCal']:
                    obs_program_used += float(group[item]['setupTime']) / 1000.
                elif group[item]['obsClass'] in ['partnerCal']:
                    obs_partner_used += float(group[item]['setupTime']) / 1000.

            print(f"program_used: {obs_program_used}")
            print(f"partner_used: {obs_partner_used}")

            print()

    return


def prog_proc(program):
    '''Process top-level of program'''

    grpnum = []
    grplist = []
    for prog in list(program.keys()):
        #     print(list(program[prog].keys()))

        # Any observations at the root level?
        group_proc(program[prog])

        # First pass to count and record groups
        for item in list(program[prog].keys()):
            if 'GROUP' in item:
                #                 print(item, program[prog][item]['name'])
                #         print(program[prog][item])
                grpnum.append(int(item.split('-')[1]))
                grplist.append(item.split('-')[0])

        #         gmax = np.max(grpnum)

        if len(grpnum) > 0:
            # Second pass to put the groups in the same order as in the program
            print(grplist)
            print(grpnum)
            isrt = np.argsort(grpnum)
            #     print(grpnum)
            for ii in isrt:
                group = grplist[ii] + '-' + str(grpnum[ii])
                print(group, program[prog][group]['name'])
                group_proc(program[prog][group])

    return


def printseq(sequence, comment='', csv=False, path=''):
    '''Print basic configuration information about a sequence, with an option to write to a csv file'''

    atom = '1'
    if csv and path != '':
        obsid = sequence[0]['ocs:observationId']
        filename = os.path.join(path, obsid + '_seq.csv')
        f = open(filename, 'w')
        print('{},{}'.format('comment', comment), file=f)
        print('{},{},{},{},{},{},{},{},{},{},{},{}'.format('datalab', 'class', 'exptime', 'coadds', 'inst', 'fpu',
                                                        'filter_name', 'disperser', 'wavelength', 'p', 'q', 'atom'), file=f)

    for step in list(sequence):
        datalab = step['observe:dataLabel']
        observe_class = step['observe:class']
        exptime = step['observe:exposureTime']
        inst = step['instrument:instrument']
    #     print(inst, fpuinst[inst])
        fpu = step[fpuinst[inst]]
        if 'instrument:filter_name' in step.keys():
            filter_name = step["instrument:filter_name"]
        else:
            filter_name = 'None'
        wavelength = step['instrument:observingWavelength']
        if 'GMOS' in inst:
            coadds = '1'
            # convert wavelength to microns
#             wavelength = '{:5.3f}'.format(float(wavelength) / 1000.)
        else:
            coadds = step['observe:coadds']
        disperser = step['instrument:disperser']
        if 'telescope:p' in step.keys():
            p = step['telescope:p']
        else:
            p = '0.0'
        if 'telescope:q' in step.keys():
            q = step['telescope:q']
        else:
            q = '0.0'    
        print('{:25} {:10} {:7} {:3} {:10} {:20} {:12} {:12} {:7} {:5} {:5}'.format(datalab, observe_class, exptime, coadds,
                                                                       inst, fpu, filter_name, disperser, wavelength, p, q))
        if csv and path != '':
            print('{},{},{},{},{},{},{},{},{},{},{},{}'.format(datalab, observe_class, exptime, coadds, inst, fpu,
                                                               filter_name, disperser, wavelength, p, q, atom), file=f)

    if csv and path != '':
        f.close()


def seqxlsx(sequence, comment='', path=''):
    '''Write sequence information to an Excel spreadsheet'''

    obsid = sequence[0]['ocs:observationId']
    filename = os.path.join(path, obsid + '_seq.xlsx')
    wb = Workbook()
    ws = wb.active
    
    atom = '1'
    
    # Comment
    ws['A1'] = 'comment'
    ws['B1'] = comment

    # Columns
    columns = ['datalab', 'class', 'inst', 'exptime', 'coadds', 'fpu', 'filter_name',
               'disperser', 'wavelength', 'p', 'q', 'atom']
    
    row = 2
    for ii, col in enumerate(columns):
        _ = ws.cell(column=ii+1, row=row, value="{0}".format(col))
    row += 1
        
#     print('{},{}'.format('comment', comment), file=f)
#     print('{},{},{},{},{},{},{},{},{},{},{}'.format('datalab', 'class', 'exptime', 'coadds', 'inst', 'fpu', 
#                                                            'disperser', 'wavelength', 'p', 'q', 'atom'), file=f)

    for step in list(sequence):
        data = []
        data.append(step['observe:dataLabel'])
        data.append(step['observe:class'])
        inst = step['instrument:instrument']
        data.append(inst)
    #     print(inst, fpuinst[inst])
        data.append(float(step['observe:exposureTime']))
        if 'GMOS' in inst:
            coadds = '1'
            # convert wavelength to microns
#             wavelength = '{:5.3f}'.format(float(wavelength) / 1000.)
        else:
            coadds = step['observe:coadds']
        data.append(int(coadds))
        data.append(step[fpuinst[inst]])
        if 'instrument:filter_name' in step.keys():
            filter_name = step["instrument:filter_name"]
        else:
            filter_name = 'None'
        data.append(filter_name)
        data.append(step['instrument:disperser'])
        data.append(float(step['instrument:observingWavelength']))
        if 'telescope:p' in step.keys():
            p = step['telescope:p']
        else:
            p = '0.0'
        data.append(float(p))
        if 'telescope:q' in step.keys():
            q = step['telescope:q']
        else:
            q = '0.0'  
        data.append(float(q))
        data.append(int(atom))
        print(data)
        
        for ii in range(len(columns)):
            _ = ws.cell(column=ii+1, row=row, value=data[ii])
        row += 1
    
    wb.save(filename)
    return


def readseq(file, path):
    '''Read sequence information from a csv file'''

    filename = os.path.join(path, file)
    f = open(filename, 'r')
    
    sequence = {}
    
    # Read and parse csv file: first line is a comment, second has column headings
    nline = 0
    for line in f:
#         line = line.rstrip('\n')
        values = line.rstrip('\n').split(',')
        if nline == 0:
            sequence['comment'] = values[1]
        elif nline == 1:
            columns = list(values)
            print(columns)
            for col in columns:
                sequence[col.strip(' ')] = []
        else:
            for i, val in enumerate(values):
                sequence[columns[i].strip(' ')].append(val.strip(' '))
        nline += 1
        
    f.close()
    
    return sequence


def xlsxseq(file, path):
    '''Read sequence information from an Excel spreadsheet'''

    filename = os.path.join(path, file)
    
    wb = load_workbook(filename=filename)
    ws = wb.active
    
    sequence = {}
    
    row = 1
    sequence['comment'] = ws.cell(column=2, row=row).value
    row += 1
    
    columns = []
    # Eventually ready the number of columns in the sheet
    for ii in range(26):
        col = ws.cell(column=ii+1, row=row).value
        if col is not None:
            columns.append(col)
            sequence[col] = []
        else:
            break
    row += 1
#     print(columns)

    while ws.cell(column=1, row=row).value is not None:
        for jj, col in enumerate(columns):
            sequence[col].append(ws.cell(column=jj+1, row=row).value)
        row += 1
    
    return sequence


if __name__ == '__main__':

    path = './data'
    print(path)

    # Tellurics
    file = 'GN-2018B-Q-101.json.gz'

    # with open(os.path.join(path, file), 'r') as f:
    #     program = json.load(f)
    #     program = f.read()
    with gzip.open(os.path.join(path, file), 'r') as fin:
        json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')
    program = json.loads(json_str)

    # print(list(program.keys()))

    # Top-level program information
    print('Evaluating: ', file)
    print('Top-level program information')
    for prog in list(program.keys()):
    #     print(list(program[prog].keys()))
        for item in list(program[prog].keys()):
            print(item)
    #         print(program[prog][item])
            if 'INFO' in item:
                print(f"\t {program[prog][item]['title']}")
            if all(type not in item for type in ['INFO', 'GROUP']):
                print(f" \t {program[prog][item]}")
    print()

    prog = list(program.keys())[0]
    group = 'GROUP_GROUP_SCHEDULING-23'
    print(group)
    obsnum = []
    for item in list(program[prog][group].keys()):
        obsid = ''
        if 'OBSERVATION' in item:
#         obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
            obsid = program[prog][group][item]['observationId']
            obsnum.append(item.split('-')[1])
            print(item, obsnum[-1], obsid)
        else:
            print(item, program[prog][group][item])
    print()

    # Sort groups to the order defined in the observation
    print('Sorted into defined order:')
    obsnum.sort()
    # print(obsnum)
    for ii in obsnum:
        item = 'OBSERVATION_BASIC-' + ii
    #     obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
        obsid = program[prog][group][item]['observationId']
        print(ii, obsid)
    print()

    obs = 'OBSERVATION_BASIC-1'
    print(obs)
    for item in list(program[prog][group][obs].keys()):
        print(item)
        if 'INFO' in item:
                print(f"\t {program[prog][group][obs][item]['title']}")
        if any(type in item for type in ['CONDITIONS', 'TARGETENV']):
            for key in list(program[prog][group][obs][item].keys()):
                print(f"\t {key}")
        if all(type not in item for type in ['INFO', 'sequence', 'obsLog']):
                print(f" \t {program[prog][group][obs][item]}")

    tot_seq_time = 0.0
    for step in list(program[prog][group][obs]['sequence']):
        print(step['observe:dataLabel'], step['observe:class'], step['observe:exposureTime'],
              step['observe:coadds'], step['instrument:instrument'], step['instrument:slitWidth'],
              step['instrument:disperser'], step['instrument:observingWavelength'], step['telescope:p'],
              step['telescope:q'], step['totalTime']/(1000.))
        tot_seq_time += step['totalTime']/(1000.)
    print('Acq overhead: ', program[prog][group][obs]['setupTime']/1000.)
    print('Sequence time:', tot_seq_time)
    exec_time = program[prog][group][obs]['setupTime']/1000. + tot_seq_time
    print('Total observation time: ', exec_time, exec_time/3600.)
    print()

    # printseq(program[prog][group][obs]['sequence'])

    print('Sequence Atoms')
    atoms = findatoms(program[prog][group][obs])
    # Summary of atoms
    for atom in atoms:
        print('Atom ', atom['id'])
        for key in atom.keys():
            print(f" \t {key}: {atom[key]}")
    print()

    print('Process program to define atoms')
    prog_proc(program)
    print()

    targenv = 'TELESCOPE_TARGETENV-2'
    print(targenv)
    for item in list(program[prog][group][obs][targenv].keys()):
        print(item)
        print(program[prog][group][obs][targenv][item])
    print()

    constraints = 'SCHEDULING_CONDITIONS-1'
    print(constraints)
    for item in list(program[prog][group][obs][constraints].keys()):
        print(item, ':', program[prog][group][obs][constraints][item])
    print()
    print()

    # ToO/Timing Window Example
    file = 'GN-2018B-Q-106.json.gz'
    # with open(os.path.join(path, file), 'r') as f:
    #     program = json.load(f)
    with gzip.open(os.path.join(path, file), 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    program = json.loads(json_str)

    # Top-level program information
    print('Evaluating: ', file)
    print('Top-level program information')
    for prog in list(program.keys()):
        # print(list(program[prog].keys()))
        for item in list(program[prog].keys()):
            print(item)
            print(program[prog][item])
            if 'INFO' in item:
                print(f"\t {program[prog][item]['title']}")
            if all(type not in item for type in ['INFO', 'GROUP']):
                print(f" \t {program[prog][item]}")
    print()

    prog = list(program.keys())[0]
    group = 'GROUP_GROUP_SCHEDULING-12'
    print(group)
    obsnum = []
    for item in list(program[prog][group].keys()):
        obsid = ''
        if 'OBSERVATION' in item:
            obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
            obsnum.append(item.split('-')[1])
            print(item, obsnum[-1], obsid)
        else:
            print(item, program[prog][group][item])
    print()

    print('Sorted into defined order:')
    obsnum.sort()
    # print(obsnum)
    for ii in obsnum:
        item = 'OBSERVATION_BASIC-' + ii
        obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
        print(ii, obsid)
    print()

    obs = 'OBSERVATION_BASIC-1'
    print(obs)
    for item in list(program[prog][group][obs].keys()):
        print(item)
        if 'INFO' in item:
                print(f"\t {program[prog][group][obs][item]['title']}")
        if any(type in item for type in ['CONDITIONS', 'TARGETENV']):
            for key in list(program[prog][group][obs][item].keys()):
                print(f"\t {key}")
        if all(type not in item for type in ['INFO', 'sequence', 'obsLog']):
                print(f" \t {program[prog][group][obs][item]}")
    print()

    # obs = 'OBSERVATION_BASIC-1'
    # for step in program[prog][group][obs]['sequence']:
    #     print(step['observe:dataLabel'], step['observe:class'], step['observe:exposureTime'], step['instrument:instrument'], step['instrument:fpu'],
    #           step['instrument:disperser'], step['instrument:observingWavelength'], step['telescope:p'], step['telescope:q'])
    #
    #
    # print(fpuinst['GMOS-N'])
    #
    # print(len(program[prog][group][obs]['sequence']))

    # obs = 'OBSERVATION_BASIC-1'
    # for step in program[prog][group][obs]['sequence']:
    #     datalab = step['observe:dataLabel']
    #     observe_class = step['observe:class']
    #     exptime = step['observe:exposureTime']
    #     inst = step['instrument:instrument']
    # #     print(inst, fpuinst[inst])
    #     fpu = step[fpuinst[inst]]
    #     if 'GMOS' in inst:
    #         coadds = '1'
    #     else:
    #         coadds = step['observe:coadds']
    #     disperser = step['instrument:disperser']
    #     wavelength = step['instrument:observingWavelength']
    #     if 'telescope:p' in step.keys():
    #         p = step['telescope:p']
    #     else:
    #         p = '0.0'
    #     if 'telescope:q' in step.keys():
    #         q = step['telescope:q']
    #     else:
    #         q = '0.0'
    #     print('{:25} {:10} {:7} {:3} {:10} {:20} {:12} {:7} {:5} {:5}'.format(datalab, observe_class, exptime, coadds, inst, fpu,
    #                                                            disperser, wavelength, p, q))

    #printseq(program[prog][group][obs]['sequence'], comment='Split by wavelength', csv=True, path=path)
    # printseq(program[prog][group][obs]['sequence'])

    print('Sequence Atoms')
    atoms = findatoms(program[prog][group][obs])
    # Summary of atoms
    for atom in atoms:
        print('Atom ', atom['id'])
        for key in atom.keys():
            print(f" \t {key}: {atom[key]}")

    # seq = readseq('GN-2018B-Q-106-43_seq.csv', path=path)
    # print(seq.keys())
    # print(seq['comment'])
    # for ii in range(len(seq['datalab'])):
    #     datalab = seq['datalab'][ii]
    #     observe_class = seq['class'][ii]
    #     exptime = seq['exptime'][ii]
    #     coadds = seq['coadds'][ii]
    #     inst = seq['inst'][ii]
    #     fpu = seq['fpu'][ii]
    #     disperser = seq['disperser'][ii]
    #     wavelength = seq['wavelength'][ii]
    #     p = seq['p'][ii]
    #     q = seq['q'][ii]
    #     atom = seq['atom'][ii]
    #
    #     print('{:20} {:12} {:7} {:3} {:10} {:20} {:12} {:7} {:5} {:5} {:3}'.format(datalab, observe_class, exptime, coadds, inst, fpu,
    #                                                            disperser, wavelength, p, q, atom))


    # seqxlsx(program[prog][group][obs]['sequence'], comment='Split by wavelength', path=path)
    #
    #
    # seq = xlsxseq('GN-2018B-Q-106-43_seq.xlsx', path=path)
    # print(seq.keys())
    # print(seq['comment'])
    # for ii in range(len(seq['datalab'])):
    #     datalab = seq['datalab'][ii]
    #     observe_class = seq['class'][ii]
    #     exptime = str(seq['exptime'][ii])
    #     coadds = str(seq['coadds'][ii])
    #     inst = seq['inst'][ii]
    #     fpu = seq['fpu'][ii]
    #     disperser = seq['disperser'][ii]
    #     wavelength = seq['wavelength'][ii]
    #     p = seq['p'][ii]
    #     q = seq['q'][ii]
    #     atom = seq['atom'][ii]
    #
    #     print('{:20} {:12} {:7} {:3} {:10} {:20} {:12} {:7} {:5} {:5} {:3}'.format(datalab, observe_class, exptime, coadds, inst, fpu,
    #                                                            disperser, wavelength, p, q, atom))


    # FT/nonsidereal/cadence/no groups
    # file = 'GN-2018B-FT-206.json'
    # with open(os.path.join(path, file), 'r') as f:
    #     program = json.load(f)
    #
    #
    # for prog in list(program.keys()):
    # #     print(list(program[prog].keys()))
    #     for item in list(program[prog].keys()):
    #         print(item)
    # #         print(program[prog][item])
    #         if 'INFO' in item:
    #             print(f"\t {program[prog][item]['title']}")
    #         if all(type not in item for type in ['INFO', 'GROUP']):
    #             print(f" \t {program[prog][item]}")
    #
    #
    # obs = 'OBSERVATION_BASIC-6'
    # for item in list(program[prog][obs].keys()):
    #     print(item)
    #     if 'INFO' in item:
    #             print(f"\t {program[prog][obs][item]['title']}")
    #     if any(type in item for type in ['CONDITIONS', 'TARGETENV']):
    #         for key in list(program[prog][obs][item].keys()):
    #             print(f"\t {key}")
    #     if all(type not in item for type in ['INFO', 'sequence', 'obsLog']):
    #             print(f" \t {program[prog][obs][item]}")
    #
    #
    # # DD/Nonsidereal/Colossus
    # file = 'GN-2018B-DD-104.json'
    # with open(os.path.join(path, file), 'r') as f:
    #     program = json.load(f)
    #
    #
    # for prog in list(program.keys()):
    # #     print(list(program[prog].keys()))
    #     for item in list(program[prog].keys()):
    #         print(item)
    # #         print(program[prog][item])
    #         if 'INFO' in item:
    #             print(f"\t {program[prog][item]['title']}")
    #         if all(type not in item for type in ['INFO', 'GROUP']):
    #             print(f" \t {program[prog][item]}")
    #
    #
    # prog = list(program.keys())[0]
    # group = 'GROUP_GROUP_SCHEDULING-13'
    # obsnum = []
    # for item in list(program[prog][group].keys()):
    #     obsid = ''
    #     if 'OBSERVATION' in item:
    #         obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
    #         obsnum.append(int(item.split('-')[1]))
    #         print(item, obsnum[-1], obsid)
    #     else:
    #         print(item, program[prog][group][item])
    #
    # obsnum.sort()
    # # print(obsnum)
    # for ii in obsnum:
    #     item = 'OBSERVATION_BASIC-' + str(ii)
    #     obsid = program[prog][group][item]['sequence'][0]['ocs:observationId']
    #     print(ii, obsid, program[prog][group][item]['sequence'][0]['instrument:instrument'])
    #
    # obs = 'OBSERVATION_BASIC-7'
    # for item in list(program[prog][group][obs].keys()):
    #     print(item)
    #     if 'INFO' in item:
    #             print(f"\t {program[prog][group][obs][item]['title']}")
    #     if any(type in item for type in ['CONDITIONS', 'TARGETENV']):
    #         for key in list(program[prog][group][obs][item].keys()):
    #             print(f"\t {key}")
    #     if all(type not in item for type in ['INFO', 'sequence', 'obsLog']):
    #             print(f" \t {program[prog][group][obs][item]}")

