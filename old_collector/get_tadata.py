from __future__ import print_function
from astropy.table import Table
import astropy.units as u
import numpy as np
import sys
import requests
from xml.dom import minidom
import os
from logging import Logger

from common.structures.site import Site


def get_report(site: Site, report, path, verbose=False):
    """
    Get report from the ODB server, must be run from inside the Gemini firewall.

    Parameters
        site: A Site as defined in the Site file.
        report: report name (str)
        path: directory for resulting file (str)
        verbose: verbose output? (bool)
    """

    outfile = os.path.join(path, report)

    # TODO: We will need to generalize this.
    url = 'http://' + site.value + 'odb.gemini.edu:8442/batch/' + report

    if verbose:
        print(site, report, path)
        print(url)
        print(outfile)

    response = requests.get(url, verify=False)
    try:
        response.raise_for_status()
        # print(response.headers['content-type'])
        fob = open(outfile, 'w')
        fob.write(response.text)
        fob.close()
    except requests.exceptions.HTTPError as exc:
        print('Request failed: {}'.format(response.content))
        raise exc

    return


def get_tas(tasfile):
    try:
        tas = open(tasfile)
    except IOError:
        print ('Error: file '+tasfile+' could not be found.')
        sys.exit(1)

    # tab = Table([[''], [''], [''], [0.0], [0.0], [0.0], [''], ['']],
    #           names=('utc', 'progid', 'inst', 'prg', 'cal', 'total', 'account', 'comment'))
    # Create an empty table
    tab = Table([[], [], [], [], [], [], [], []],
                       names=('utc', 'progid', 'inst', 'prg', 'cal', 'total', 'account', 'comment'),
                       dtype=('S8', 'S15', 'S20', 'f8', 'f8', 'f8', 'S2', 'S80'))
    for line in tas:
        # if debug:
        #    print line
        if line[0] == '#':
            continue
        line = line.rstrip('\n')
        vals = line.split(',',7)
        # print vals
        if 'Visitor Instrument' in vals[2]:
            vals[2] = 'Visitor'
            # print 'vals[2]:', vals[2]

        tab.add_row([vals[0],vals[1],vals[2],float(vals[3]),float(vals[4]),float(vals[5]),\
                    vals[6],vals[7]])
        # ii = ii + 1

    tas.close()

    return tab


def get_exec(execfile):
    try:
        execf = open(execfile)
    except IOError:
        print ('Error: file '+execfile+' could not be found.')
        sys.exit(1)

    # tab = Table([[''], [''], [''], [0.0], [0.0], [0.0], [''], ['']],
    #           names=('utc', 'progid', 'inst', 'prg', 'cal', 'total', 'account', 'comment'))
    # Create an empty table
    tab = Table([[], [], [], [], [], []],
                       names=('progid', 'alloc', 'elapsed', 'nocharge', 'part', 'prg'),
                       dtype=('S15', 'f8', 'f8', 'f8', 'f8', 'f8'))
    for line in execf:
        # if debug:
        #    print line
        if line[0] == '#':
            continue
        line = line.rstrip('\n')
        vals = line.split(',',5)
        # print vals
        tab.add_row([vals[0],float(vals[1]),float(vals[2]),float(vals[3]),float(vals[4]),float(vals[5])])

    execf.close()

    return tab


def get_tadata(site='GN', semesters=['2012A','2012B','2013A','2013B','2014B','2015A','2015B'],
               datadir='/Users/bmiller/gemini/sciops/time_accounting/'):

    # Read time accounting data from standard ODB report files

    # Where to find the files
    rootdir = datadir

    # Table for holding program summary info from tas files
    tasprog = Table([[], [], [], [], [], []],
                names=('progid', 'account', 'inst', 'prg', 'cal', 'total'),
                dtype=('S15', 'S2', 'S20', 'f8', 'f8', 'f8'))

    # Table for holding exec summary info from execHrs files
    exsum = Table([[], [], [], [], [], [], [], []],
                  names=('progid', 'sem', 'account', 'alloc', 'elapsed', 'nocharge', 'part', 'prg'),
                  dtype=('S15', 'S5', 'S2', 'f8', 'f8', 'f8', 'f8', 'f8'))

    print ('Reading files')
    for sem in semesters:
        # Time accounting file, includes support partner
        tasfile = 'tas_'+site+'_'+sem+'.txt'
        print (tasfile)
        tas = get_tas(rootdir+tasfile)
        # print tas['progid'].size
        # print tas['progid'][0:9]

        # Executed hours, includes allocated time
        execfile = 'execHours_'+site+'_'+sem+'.txt'
        print (execfile)
        exhrs = get_exec(rootdir+execfile)
        # print exhrs

        # List of unique program ids
        # print sorted(tas['progid'])
        # prgids = uniquelist(tas['progid'])
        # nprgid = len(prgids)
        # print 'Unique program ids: ',len(prgids)
        #
        # empty = ['' for ii in range(nprgid)]
        # zeros = np.zeros(nprgid)
        # # Table for holding program summary info from tas file
        # tasprog = Table([prgids, empty, zeros, zeros, zeros, empty],
        #             names=('progid', 'inst', 'prg', 'cal', 'total', 'account'),
        #             dtype=('S15', 'S8', 'f8', 'f8', 'f8', 'S2'))

        for ii in range(tas['progid'].size):
            ip = np.where([prg == str(tas['progid'][ii]).rstrip() for prg in tasprog['progid']])[0]
            if ip.size == 0:
                tasprog.add_row([str(tas['progid'][ii]).rstrip(), str(tas['account'][ii]).rstrip(), \
                                 str(tas['inst'][ii]).rstrip(), tas['prg'][ii], \
                                 tas['cal'][ii], tas['total'][ii]])
            else:
                tasprog['prg'][ip] += tas['prg'][ii]
                tasprog['cal'][ip] += tas['cal'][ii]
                tasprog['total'][ip] += tas['total'][ii]
                # print (tasprog['inst'][ip], tas['inst'][ii])
                if str(tas['inst'][ii]).rstrip() not in str(tasprog['inst'][ip]):
                    # print ip, tasprog['inst'][ip], tas['inst'][ii]
                    tasprog['inst'][ip] = tasprog['inst'][ip[0]] + ',' + str(tas['inst'][ii])

        #print tasprog['progid','account','inst','prg','cal','total']

        exhrs['account'] = 'NA'

        # ipt = {'AR':[], 'AU':[], 'BR':[], 'CA':[], 'CL':[], 'UH':[], 'US':[]}

        for ii in range(exhrs['progid'].size):
            ip = np.where([prg == str(exhrs['progid'][ii]).rstrip() for prg in exsum['progid']])[0]
            if ip.size == 0:
                # Entry in tasprog (for account)
                itas = np.where([prg == str(exhrs['progid'][ii]).rstrip() for prg in tasprog['progid']])[0]
                if itas.size == 0:
                    if 'FT' in exhrs['progid'][ii]:
                        account = ['FT']
                    elif 'LP' in exhrs['progid'][ii]:
                        account = ['LP']
                    elif 'SV' in exhrs['progid'][ii]:
                        account = ['SV']
                    elif 'DD' in exhrs['progid'][ii]:
                        account = ['DD']
                    else:
                        account = ['NA']
                else:
                    account = tasprog['account'][itas]
                    # if account[0] in partners[site]:
                    #     ipt[account[0]].append(ii)
                exhrs['account'][ii] = account[0]
                exsum.add_row([str(exhrs['progid'][ii]).rstrip(), sem, account,
                                 exhrs['alloc'][ii], exhrs['elapsed'][ii],
                                 exhrs['nocharge'][ii], exhrs['part'][ii], exhrs['prg'][ii]])
            else:
                exsum['elapsed'][ip] += exhrs['elapsed'][ii]
                exsum['nocharge'][ip] += exhrs['nocharge'][ii]
                exsum['part'][ip] += exhrs['part'][ii]
                exsum['prg'][ip] += exhrs['prg'][ii]
                # Add allocated times for LP?

        #print exsum
        # exsum['chdivprg'] = np.where(exsum['prg'] > 0.0,(exsum['part']+exsum['prg'])/exsum['prg'],np.nan)
        #exsum['chdivprg'] = (exsum['part'][ic]+exsum['prg'][ic])/exsum['prg'][ic]
        #print exsum['chdivprg']
        # print 'Mean (partner+program)/program =', np.nanmean(exsum['chdivprg'])

    print ('Done')

    return exsum, tasprog


def get_odbprog(sites=['gs', 'gn'], semesters=['2015B', ], overwrite=False, verbose=False,
    datadir='/Users/bmiller/gemini/sciops/time_accounting/'):

    # Get time accounting data using the 'ODB browser' API

    # Location for XML data files
    path = datadir

    # Program account translations
    accounts = {'Argentina': 'AR',
                'Australia': 'AU',
                'Brazil': 'BR',
                'Canada': 'CA',
                'Chile': 'CL',
                'Republic of Korea': 'KR',
                'United States': 'US',
                'University of Hawaii': 'UH',
                'United Kingdom': 'UK',
                'Gemini Staff': 'GS',
                'Guaranteed Time': 'GT',
                "Director's Time": 'DD',
                'Large Program': 'LP',
                'System Verification': 'SV',
                'Demo Science': 'DS',
                'Fast Turnaround': 'FT',
                'Limited-term Participant': 'LT',
                'Subaru': 'SU',
                'CFHT Exchange': 'CF',
                'Keck Exchange': 'KE',
                'None': 'NA',
                }

    # exsum = Table(rows=[[], [], [], [], [], [], [], [], [], [], []],
    #               names=('progid', 'sem', 'band', 'account', 'inst', 'ao', 'too', 'thesis', 'alloc',
    #                      'prg', 'complete'),
    #               dtype=('S15', 'S5', 'i4', 'S8', 'S15', 'S3', 'S8', 'S3', 'f8', 'f8', 'S3'))
    rows = []
    for sem in semesters:
        # params['programActive'] = 'Yes'
        if verbose:
            print(sem)
        for site in sites:
            # Verbose
            if verbose:
                print(site)
                
            # File with ODB data
            xmlfile = 'odbprogobs_' + site + sem + '.xml'

            # Query ODB if requested or file does not exist
            if overwrite or not os.path.exists(path + xmlfile):
                params = {'programSemester': sem}
                # url = 'http://' + site + 'odb.gemini.edu:8442/odbbrowser/programs'
                url = 'http://' + site + 'odb.gemini.edu:8442/odbbrowser/observations'

                response = requests.get(url, verify=False, params=params)
                try:
                    response.raise_for_status()
                    # print(response.headers['content-type'])
                    fob = open(path + xmlfile, 'w')
                    fob.write(response.text)
                    fob.close()
                    # xmlresp = minidom.parseString(response.text)
                except requests.exceptions.HTTPError as exc:
                    print('Request failed: {}'.format(response.content))
                    raise exc

            # Read XML file
            xmlresp = minidom.parse(path + xmlfile)

            progs = xmlresp.getElementsByTagName('program')
            for prog in progs:
                progid = prog.getElementsByTagName('reference')[0].firstChild.data
                try:
                    band = int(prog.getElementsByTagName('scienceBand')[0].firstChild.data)
                except:
                    band = -1
                active = prog.getElementsByTagName('active')[0].firstChild.data
                complete = prog.getElementsByTagName('completed')[0].firstChild.data
                toostat = prog.getElementsByTagName('tooStatus')[0].firstChild.data
                thesis = prog.getElementsByTagName('thesis')[0].firstChild.data
                alloc = int(prog.getElementsByTagName('allocatedTime')[0].firstChild.data) * u.ms
                remain = int(prog.getElementsByTagName('remainingTime')[0].firstChild.data) * u.ms
                piemails = []
                ngoemail = []
                csemail = []
                try:
                    emails = prog.getElementsByTagName('ngoEmail')[0].firstChild.data.split(',')
                except:
                    emails = []
                [ngoemail.append(email.strip(' ')) for email in emails]

                try:
                    emails = prog.getElementsByTagName('contactScientistEmail')[0].firstChild.data.split(',')
                except:
                    emails = []
                [csemail.append(email.strip(' ')) for email in emails]

                team = prog.getElementsByTagName('investigator')
                # print(progid)
                for member in team:
                    pi = member.getAttribute('pi')
                    if pi == 'true':
                        piname = member.getElementsByTagName('name')[0].firstChild.data
                        try:
                            emails = member.getElementsByTagName('email')[0].firstChild.data.split(',')
                        except:
                            emails = []
                        # print(name, email, pi, member.nodeName, member.nodeValue)
                        [piemails.append(email.strip(' ')) for email in emails]
                        # print(email)

                partdict = {}
                partners = prog.getElementsByTagName('partners')
                maxalloc = 0.0
                maxpart = 'None'
                for partner in partners:
                    partname = partner.getElementsByTagName('name')[0].firstChild.data
                    allocated = float(partner.getElementsByTagName('hoursAllocated')[0].firstChild.data)
                    if allocated > maxalloc:
                        maxalloc = allocated
                        maxpart = partname
                    partdict[accounts[partname]] = allocated

                observations = prog.getElementsByTagName('observations')
                insts = ''
                useao = 'No'
                for obs in observations:
                    try:
                        inst = obs.getElementsByTagName('instrument')[0].firstChild.data
                    except:
                        pass
                    else:
                        if inst not in insts:
                            insts += inst + ','

                    try:
                        ao = obs.getElementsByTagName('ao')[0].firstChild.data
                    except:
                        pass
                    else:
                        if useao == 'No':
                            # For some reason GSAOI observations have ao=None
                            if inst == 'GSAOI' or ao != 'None':
                                useao = 'Yes'

                insts = insts.rstrip(',')

                # print(progid, piname, piemails, ngoemail, csemail, thesis, active, toostat, complete, active,
                #       alloc.to_value('h'), remain.to_value('h'), partdict, maxpart, insts, useao)

                # exsum.add_row([progid, sem, band, accounts[maxpart], insts, useao, toostat, thesis, alloc.to_value('h'),
                #                (alloc - remain).to_value('h'), complete])

                # It's much faster to append to a list and then create the table at the end
                rows.append([progid, sem, band, accounts[maxpart], insts, useao, toostat, thesis, alloc.to_value('h'),
                               (alloc - remain).to_value('h'), complete])

    exsum = Table(rows=rows,
                  names=('progid', 'sem', 'band', 'account', 'inst', 'ao', 'too', 'thesis', 'alloc',
                         'prg', 'complete'),
                  dtype=('S15', 'S5', 'i4', 'S8', 'S15', 'S3', 'S8', 'S3', 'f8', 'f8', 'S3'))
    exsum['alloc'].info.format = '8.3f'
    exsum['prg'].info.format = '8.3f'

    return exsum


def sumtas_date(tas, tadate):
    # Select all entries up to the given date
    # tas = time accounting summary information from get_tas
    # tadate = string date to sum through, YYYYMMDD, e.g. '20201012'

    iut = np.where(tas['utc'] <= tadate)[0][:]
    # print(tas['utc'][iut])
    # print(tas['comment'][iut[0]])

    # Sum charged program time through tadate
    progta = {}
    for jj in range(len(iut)):
        ii = iut[jj]
        # Program
        if tas['progid'][ii] not in progta.keys():
            progta[tas['progid'][ii]] = {'prgtime': tas['prg'][ii] * u.hour, 'caltime': tas['cal'][ii] * u.hour,
                                         'account': tas['account'][ii]}
        else:
            progta[tas['progid'][ii]]['prgtime'] += tas['prg'][ii] * u.hour
            progta[tas['progid'][ii]]['caltime'] += tas['cal'][ii] * u.hour

        # Observation
        i1 = tas['comment'][ii].find('[') + 1
        i2 = tas['comment'][ii].find(']')
        obsid = tas['progid'][ii] + '-' + tas['comment'][ii][i1:i2]
        if obsid not in progta[tas['progid'][ii]].keys():
            progta[tas['progid'][ii]][obsid] = {'prgtime': tas['prg'][ii] * u.hour, 'caltime': tas['cal'][ii] * u.hour}
        else:
            progta[tas['progid'][ii]][obsid]['prgtime'] += tas['prg'][ii] * u.hour
            progta[tas['progid'][ii]][obsid]['caltime'] += tas['cal'][ii] * u.hour

    return progta

