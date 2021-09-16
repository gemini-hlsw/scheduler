import datetime
import calendar
from astropy.time import Time

warnstates = ['FOR-ACTIVATION', 'ONGOING', 'READY']


### ORIGINAL source: https://github.com/bryanmiller/odb
def GetProgramID(Program):
    progid = Program.attrib.get('name')
    return(progid)   

def GetProgNotes(Program):
    notes = []
    for container in Program.findall('container'):
        if container.attrib.get('type') == 'Info':
            paramset = container.find('paramset')
            title = None
            content = None
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'title':
                    title = param.attrib.get('value')
                elif param.attrib.get('name') == 'NoteText':
                    content = param.attrib.get('value')
                    if content is None: # long content will be in a separate <value> tag
                        value = param.find('value')
                        content = value.text
            #print(f'Note Title = {title}')
            #print(f'Note Content = {content}')
            notes.append([title,content])
    return notes 

def GetMode(Program):
    mode = 'UNKNOWN'
    for paramset in Program.findall('paramset'):
        if paramset.attrib.get('name') == 'Science Program':
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'programMode':
                    mode = param.attrib.get('value')
                    break
    return(mode)

def GetBand(Program):
        band = 'UNKNOWN'
        for paramset in Program.findall('paramset'):
            if paramset.attrib.get('name') == 'Science Program':
                for param in paramset.findall('param'):
                    if param.attrib.get('name') == 'queueBand':
                        band = param.attrib.get('value')
                        break

        #print(f'Band = {band}')
        return(band)

def GetAwardedTime(Program):
        
    sciprogram = Program.find("paramset[@name='Science Program'][@kind='dataObj']")
    # timeAcct = sciprogram.find(paramset[@name='timeAcct']")
    awardedTime = sciprogram.find("param[@name='awardedTime']")
    # <param name="awardedTime" value="3.7" units="hours"/>
    value = float(awardedTime.attrib.get('value'))
    units = awardedTime.attrib.get('units')
    #print(f'value = {value}')
    #print(f'units = {units}')
    return value, units

def GetThesis(Program):
    paramset = Program.find("paramset[@name='Science Program'][@kind='dataObj']")
    param = paramset.find("param[@name='isThesis']")
    if param is not None:
        thesis = param.attrib.get('value')
        if thesis == 'true':
            thesis = True
    else:
        thesis = False
    return thesis

def GetTooStatus(Program):
    for paramset in Program.findall('paramset'):
        if paramset.attrib.get('name') == 'Science Program':
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'tooType':
                    too = param.attrib.get('value')

    #print(f'ToO Status = {too}')
    return(too)

def GetClass(Observation):
    obsclasses = []
    for container in Observation.findall('.//container'):
        if container.attrib.get("type") == 'Observer':
            paramset = container.find("paramset")
            for param in paramset.findall("param"):
                if param.attrib.get("name") == "class":
                    obsclasses.append(param.attrib.get("value"))
    return(obsclasses)

def GetObsStatus(Observation):

    execstatus = 'AUTO'
    paramset = Observation.find('paramset')
    for param in paramset.findall('param'):

        # In 2014A the Status was split into Phase-2 and Exec status
        # Here I recombine them into one as they were before 2014A:

        if param.attrib.get('name') == 'phase2Status':
            phase2status = param.attrib.get('value')
            #logger.debug('Raw Phase 2 Status = %s', phase2status)

        if param.attrib.get('name') == 'execStatusOverride':
            execstatus = param.attrib.get('value')

    if  execstatus == 'OBSERVED':
        obsstatus = 'OBSERVED'

    elif execstatus == 'ONGOING':
        obsstatus = 'ONGOING'

    elif execstatus == 'PENDING':
        obsstatus = 'READY'

    elif execstatus == 'AUTO':
        if phase2status == 'PI_TO_COMPLETE':
            obsstatus = 'PHASE2'
        elif phase2status == 'NGO_TO_REVIEW':
            obsstatus = 'FOR-REVIEW'
        elif phase2status == 'NGO_IN_REVIEW':
            obsstatus = 'IN-REVIEW'
        elif phase2status == 'GEMINI_TO_ACTIVATE':
            obsstatus = 'FOR-ACTIVATION'
        elif phase2status == 'ON_HOLD':
            obsstatus = 'ON-HOLD'
        elif phase2status == 'INACTIVE':
            obsstatus = 'INACTIVE'
        elif phase2status == 'PHASE_2_COMPLETE':
            
            obslog = GetObsLog(Observation)  # returns a triple: (time, event, datalabel)
            nsteps = GetNumSteps(Observation)
            nobs = GetNumObserved(Observation)

            if nobs == 0 and len(obslog[0]) == 0:
                obsstatus = 'READY'
                
            elif nobs >= nsteps:
                obsstatus = 'OBSERVED'
                
            else:
                obsstatus = 'ONGOING'
                
        else:
            print('UNKNOWN PHASE-2 STATUS: %s', phase2status)
    return(obsstatus)

def GetObsLog(Observation):
        
        event = []
        time = []
        datalabel = []

        for container in Observation.findall('container'):
            if container.attrib.get('kind') == 'obsExecLog':
                for paramset in container.findall('paramset'):
                    if paramset.attrib.get('name') == 'Observation Exec Log':
                        for paramset2 in paramset.findall('paramset'):
                            if paramset2.attrib.get('name') == 'obsExecRecord':
                                for paramset3 in paramset2.findall('paramset'):
                                    if paramset3.attrib.get('name') == 'events':
                                        for paramset4 in paramset3.findall('paramset'):
                                            event.append(paramset4.attrib.get('kind'))
                                            label = False
                                            for param in paramset4.findall('param'):
                                                if param.attrib.get('name') == 'timestamp':
                                                    time.append(datetime.datetime.utcfromtimestamp(int(param.attrib.get('value'))/1000.))
                                                elif param.attrib.get('name') == 'datasetLabel':
                                                    datalabel.append(param.attrib.get('value'))
                                                    label = True
                                                for paramset5 in paramset4.findall('paramset'):
                                                    if paramset5.attrib.get('name') == 'dataset':
                                                        for param in paramset5.findall('param'):
                                                            if param.attrib.get('name') == 'datasetLabel':
                                                                datalabel.append(param.attrib.get('value'))
                                                                label = True

                                            if not label:
                                                datalabel.append('')

        return(time,event,datalabel)

def GetNumSteps(Observation):
       
        def Count(tree, total, multiplier, obsid):
            cname = tree.attrib.get('name')
            ctype = tree.attrib.get('type')

            # If container is an Instrument iterator:
            if cname in ['AcqCam Sequence', 'Flamingos2 Sequence', 'GMOS-N Sequence', 'GMOS-S Sequence', 'GNIRS Sequence',
                        'GPI Sequence', 'GSAOI Sequence', 'NICI Sequence', 'NIFS Sequence', 'NIRI Sequence']:
                paramset = tree.find('paramset')
                nstep = None
                for param in paramset.findall('param'):
                    nstep = 0
                    for val in param.findall('value'):
                        nstep += 1
                    if nstep == 0: # If the iterator only has a single step there won't be any "val" parameters
                        nstep = 1
                # logger.debug('...instrument iterator has %i steps', nstep)
                # logger.debug('...multiplier = %s', multiplier)
                if nstep is None:
                    print('%s : %s has zero steps', obsid, cname)
                else:
                    multiplier *= nstep

            # If container is a Repeat iterator:
            elif cname == 'Repeat':
                paramset = tree.find('paramset')
                for param in paramset.findall('param'):
                    if param.attrib.get("name") == 'repeatCount':
                        nrepeats = int(param.attrib.get('value'))
                # logger.debug('...repeat iterator has = %i steps', nrepeats)
                multiplier *= nrepeats
                # logger.debug('...multiplier = %s', multiplier)

            # If container is an Offset iterator:
            elif cname == 'Offset' or cname == 'NICI Offset':
                noffsets = 0
                paramset1 = tree.find('paramset')
                for paramset2 in paramset1.findall('paramset'):
                    for paramset3 in paramset2.findall('paramset'):
                        noffsets += 1
                # logger.debug('...offset iterator has %i steps', noffsets)
                # logger.debug('...multiplier = %s', multiplier)
                if noffsets == 0:
                    print('%s : Offset iterator has zero steps', obsid)
                else:
                    multiplier *= noffsets

            # If container is an Observe/Dark/Flat:
            elif ctype == 'Observer':
                paramset = tree.find('paramset')
                for param in paramset.findall('param'):
                    if param.attrib.get("name") == 'repeatCount':
                        nobserves = int(param.attrib.get('value'))
                # logger.debug('...nobserve = %i', nobserves)
                total += nobserves * multiplier
                # logger.debug('...total = %s', total)

            else:
                print('Unknown container: %s', cname)

            return(total,multiplier)

        total = 0
        multiplier1 = 1
        obsid = Observation.attrib.get('name') # to pass for error reporting

        for container1 in Observation.findall('container'):
            # logger.debug('Container %s %s', container1.attrib.get("kind"), container1.attrib.get("type"))
            if container1.attrib.get("name") == 'Sequence': # top-level sequence

                for container2 in container1.findall("container"):
                    if container2.attrib.get("kind") == "seqComp":
                        # logger.debug('...container2')
                        multiplier2 = multiplier1
                        (total,multiplier2) = Count(container2,total,multiplier2,obsid)

                        for container3 in container2.findall("container"):
                            if container3.attrib.get("kind") == "seqComp":
                                # logger.debug('...container3')
                                multiplier3 = multiplier2
                                (total,multiplier3) = Count(container3,total,multiplier3,obsid)

                                for container4 in container3.findall("container"):
                                    if container4.attrib.get("kind") == "seqComp":
                                        # logger.debug('...container4')
                                        multiplier4 = multiplier3
                                        (total,multiplier4) = Count(container4,total,multiplier4,obsid)

                                        for container5 in container4.findall("container"):
                                            if container5.attrib.get("kind") == "seqComp":
                                                # logger.debug('...container5')
                                                multiplier5 = multiplier4
                                                (total,multiplier5) = Count(container5,total,multiplier5,obsid)

                                                for container6 in container5.findall("container"):
                                                    if container6.attrib.get("kind") == "seqComp":
                                                        # logger.debug('...container6')
                                                        multiplier6 = multiplier5
                                                        (total,multiplier6) = Count(container6,total,multiplier6,obsid)

                                                        for container7 in container6.findall("container"):
                                                            if container7.attrib.get("kind") == "seqComp":
                                                                # logger.debug('...container7')
                                                                multiplier7 = multiplier6
                                                                (total,multiplier7) = Count(container7,total,multiplier7,obsid)

                                                                for container8 in container7.findall("container"):
                                                                    if container8.attrib.get("kind") == "seqComp":
                                                                        logger.error('%s : Too many nested loops', obsid)


        

        if total == 0:
            print('%s has zero steps', obsid)

        return(total)

def GetNumObserved(Observation):
    nsteps = 0

    for container in Observation.findall('container'):
        #logger.debug('Container %s %s', container.attrib.get("kind"), container.attrib.get("type"))
        if container.attrib.get('type') == 'ObsLog':
            ObsLog = container
            ObsExecLog = ObsLog.find('paramset')
            ObsExecRecord = ObsExecLog.find('paramset')

            for paramset1 in ObsExecRecord.findall('paramset'):
                if paramset1.attrib.get('name') == 'configMap':

                    for paramset2 in paramset1.findall('paramset'):
                        if paramset2.attrib.get('name') == 'configMapEntry':

                            for param in paramset2.findall('param'):
                                if param.attrib.get('name') == 'datasetLabels':
                                    if param.attrib.get('value'): # single value
                                        nsteps += 1
                                    else: # multiple values
                                        for value in param.findall('value'):
                                            nsteps += 1

    #logger.debug('Number of observed steps = %s', nsteps)
    return(nsteps)

def GetObsID(Observation):
        obsid = 'UNKNOWN'
        obsid = Observation.attrib.get('name')
        return(obsid)

def GetObsTime(Observation):

        # TO-DO:
        # Include overhead for filter changes (specifically 50s for F2)
        # Include iterating over the read mode

        obstime = 0     # total observation time
        ope = 0         # overhead per exposure
        coadds = [1]    # list of coadds
        exptime = []    # list of exposure times from instrument sequencers
        repeat = 1
        noffsets = 1    # number of offset positions

        for container in Observation.findall('container'):
            #logger.debug('Container %s %s', container.attrib.get("kind"), container.attrib.get("type"))

            if container.attrib.get('type') == 'Instrument':
                instrument = container.attrib.get('name')
                #logger.debug('Instrument = %s', instrument)
                paramset = container.find('paramset')

                fwo = 0.0  # The file-write overhead, which I have only implemented for NIRI as of 2013 Jun 29

                if instrument == 'NIFS':
                    obstime = 11 * 60       # Acquisition overhead (s)
                    for param in paramset.findall("param"):
                        if param.attrib.get("name") == 'readMode':
                            readmode = param.attrib.get('value')
                    #logger.debug('...readmode = %s', readmode)
                    if readmode == 'FAINT_OBJECT_SPEC':
                        ope = 99
                    elif readmode == 'MEDIUM_OBJECT_SPEC':
                        ope = 35
                    elif readmode == 'BRIGHT_OBJECT_SPEC':
                        ope = 19.3
                    else:
                        print('UNKNOWN READ MODE')

                elif instrument == 'GNIRS':
                    obstime = 15 * 60       # Acquisition overhead (s)
                    for param in paramset.findall("param"):
                        if param.attrib.get("name") == 'readMode':
                            readmode = param.attrib.get('value')
                    #logger.debug('...readmode = %s', readmode)
                    if readmode == 'VERY_FAINT':
                        ope = 26.9
                    elif readmode == 'FAINT':
                        ope = 13.8
                    elif readmode == 'BRIGHT':
                        ope = 1.2
                    elif readmode == 'VERY_BRIGHT':
                        ope = 0.9
                    else:
                        print('UNKNOWN READ MODE')

                elif instrument == 'NIRI':
                    obstime = 6 * 60       # Acquisition overhead (s)
                    for param in paramset.findall("param"):
                        if param.attrib.get("name") == 'readMode':
                            readmode = param.attrib.get('value')
                    #logger.debug('...readmode = %s', readmode)
                    # Assuming full-frame:
                    fwo = 2.78
                    if readmode == 'IMAG_SPEC_NB':
                        ope = 11.15
                    elif readmode == 'IMAG_1TO25':
                        ope = 0.7
                    elif readmode == 'IMAG_SPEC_3TO5':
                        ope = 0.18
                    else:
                       print('UNKNOWN READ MODE')

                elif 'GMOS' in instrument:

                    disperser = None
                    fpu = None
                    xbin = None
                    ybin = None
                    roi = None

                    for param in paramset.findall("param"):
                        if param.attrib.get("name") == 'disperser':
                            disperser = param.attrib.get('value')
                        elif param.attrib.get("name") == 'fpu':
                            fpu = param.attrib.get('value')
                        elif param.attrib.get("name") == 'ccdXBinning':
                            xbin = param.attrib.get('value')
                        elif param.attrib.get("name") == 'ccdYBinning':
                            ybin = param.attrib.get('value')
                        elif param.attrib.get("name") == 'builtinROI':
                            roi = param.attrib.get('value')

                    #logger.debug('Disperser = %s', disperser)
                    #logger.debug('FPU = %s', fpu)
                    #logger.debug('Binning = %s x %s', xbin, ybin)
                    #logger.debug('ROI = %s', roi)

                    if disperser == 'MIRROR':
                        obstime = 6 * 60    # Imaging acquisition overhead (s)
                    elif 'SLIT' in fpu or 'NS' in fpu:
                        obstime = 16 * 60   # Long slit acquisition overhead (s)
                    elif 'IFU' in fpu:
                        obstime = 18 * 60   # IFU acquisition overhead (s)
                    elif fpu == 'CUSTOM_MASK':
                        obstime = 18 * 60   # MOS acquisition overhead (s)
                    else:
                        print('Unknown acquisition overhead for %s', GetObsID(Observation))

                    if roi == 'FULL_FRAME' or roi == 'CCD2':
                        if   xbin == 'ONE' and ybin == 'ONE':
                            ope = 72.9
                        elif xbin == 'ONE' and ybin == 'TWO':
                            ope = 41.2
                        elif xbin == 'ONE' and ybin == 'FOUR':
                            ope = 27.0
                        elif xbin == 'TWO' and ybin == 'ONE':
                            ope = 46.6
                        elif xbin == 'TWO' and ybin == 'TWO':
                            ope = 28.0
                        elif xbin == 'TWO' and ybin == 'FOUR':
                            ope = 27.0
                        elif xbin == 'FOUR' and ybin == 'ONE':
                            ope = 33.7
                        elif xbin == 'FOUR' and ybin == 'TWO':
                            ope = 27.0
                        elif xbin == 'FOUR' and ybin == 'FOUR':
                            ope = 17.0
                    elif roi == 'CENTRAL_SPECTRUM':
                        if   xbin == 'ONE' and ybin == 'ONE':
                            ope = 27.0
                        else:
                            ope = 17.0
                    elif roi == 'CENTRAL_STAMP':
                        ope = 17.0
                    elif roi == 'CUSTOM':
                        ope = 17.0  # Depends on the size of the ROI
                    else:
                        print('Unknown binning & ROI combination')

                elif instrument == 'NICI':
                    obstime = 10 * 60       # Acquisition overhead (s)
                    ope = 8.4

                elif instrument == 'GPI':
                    obstime = 10 * 60       # Acquisition overhead (s)
                    ope = 20.

                elif instrument == 'GSAOI':
                    obstime = 30 * 60       # Acquisition overhead (s)
                    ope = 43.4

                elif instrument == 'Flamingos2':
                    obstime = 15 * 60       # Acquisition overhead (s)
                    filterchange = 50       # filter change overhead (s)               # UNIMPLEMENTED!

                    for param in paramset.findall("param"):
                        if param.attrib.get("name") == 'readMode':
                            readmode = param.attrib.get('value')
                    #logger.debug('...readmode = %s', readmode)

                    if readmode == 'BRIGHT_OBJECT_SPEC':
                        ope = 8.0
                    elif readmode == 'MEDIUM_OBJECT_SPEC':
                        ope = 14.0
                    elif readmode == 'FAINT_OBJECT_SPEC':
                        ope = 20.0
                    else:
                        print('UNKNOWN READ MODE')

                elif instrument == 'Phoenix':
                    obstime = 20 * 60       # Acquisition overhead (s)
                    ope = 18.               # Overhead per exposure (s)

                elif instrument == 'Texes':
                    obstime = 20 * 60       # Acquisition overhead (s)
                    ope = 1.                # Overhead per exposure (s)

                elif instrument == 'Visitor Instrument':
                    obstime = 10 * 60       # Acquisition overhead (s)
                    ope = 1.                # Overhead per exposure (s)

                elif instrument == 'Acquisition Camera':
                    obstime = 10 * 60       # Acquisition overhead (s)
                    ope = 1.                # Overhead per exposure (s)

                else:
                    print('Unknown acquisition overhead for %s in %s', instrument, GetObsID(Observation))

                #logger.debug('Aacquisition overhead = %s', str(datetime.timedelta(seconds=obstime)))
                #logger.debug('OPE = %s', ope)

                for param in paramset.findall("param"):
                    if param.attrib.get("name") == "exposureTime":
                        toplevelexptime = float(param.attrib.get("value"))
                        exptime.append(toplevelexptime)
                        #logger.debug('Exposure time = %s', exptime)
                    if param.attrib.get('name') == 'coadds':
                        coadds[0] = int(param.attrib.get('value'))
                        #logger.debug('Coadds = %s', coadds)

            if container.attrib.get("name") == 'Sequence': # top-level sequence
                for container2 in container.findall("container"):
                    if container2.attrib.get("kind") == "seqComp":
                        (exptime, coadds, repeat, noffsets, obstime) = SumObsTime(container2, exptime, coadds, repeat, noffsets, ope, fwo, obstime)
                        for container3 in container2.findall("container"):
                            if container3.attrib.get("kind") == "seqComp":
                                (exptime, coadds, repeat, noffsets, obstime) = SumObsTime(container3, exptime, coadds, repeat, noffsets, ope, fwo, obstime)
                                for container4 in container3.findall("container"):
                                    if container4.attrib.get("kind") == "seqComp":
                                        (exptime, coadds, repeat, noffsets, obstime) = SumObsTime(container4, exptime, coadds, repeat, noffsets, ope, fwo, obstime)
                                        for container5 in container4.findall("container"):
                                            if container5.attrib.get("kind") == "seqComp":
                                                (exptime, coadds, repeat, noffsets, obstime) = SumObsTime(container5, exptime, coadds, repeat, noffsets, ope, fwo, obstime)

                                            #print '...resetting exptime list 5'
                                            exptime = []
                                            exptime.append(toplevelexptime)
                                    #print '...resetting exptime list 4'
                                    exptime = []
                                    exptime.append(toplevelexptime)

                            #print '...resetting exptime list 3'
                            exptime = []
                            exptime.append(toplevelexptime)

                    #print '...resetting exptime list 2'
                    exptime = []
                    exptime.append(toplevelexptime)

        datetime_obstime = datetime.timedelta(seconds=int(obstime))
        return(datetime_obstime)

def GetInstrument(Observation):
    #logger = logging.getLogger('odb.GetInstrument')
    instrument = None
    for container in Observation.findall('container'):
        if container.attrib.get("type") == 'Instrument':
            instrument = container.attrib.get("name")
            break

    if instrument:
        #print('Instrument = %s', instrument)
        pass
    else:
        status = GetObsStatus(Observation)
        if status in warnstates:
            print('%s is missing instrument [%s]', GetObsID(Observation), status)
        else: 
            print('%s is missing instrument [%s]', GetObsID(Observation), status)
        instrument = 'INDEF'

    return instrument

def GetInstConfigs(Observation):
    """
    Get all instrument configurations
    Return a dictionary of configurations
    Each dictionary item has an entry for each *change* of configuration (NOT each observe)
    This does NOT keep the correct order for embedded sequences!
    This is because the OT has logic that keeps the first iterator synchronized with the top-level,
    but lower sequences are not synchronized and will revert to the original sequence.  For example,
    if the binning is 1 in the top level then there is a GMOS sequence that sets the binning to 2 then
    there is another GMOS sequences which does not set the binning, it will be 1.
    """
    
    def add(dictionary, key, value):
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    # The subtype is the instrument name, except for GMOS-N (GMOS) and GMOS-S (GMOSSouth):
    instrumentcontainer = Observation.find("container[@kind='obsComp'][@type='Instrument']")
    if instrumentcontainer is None:
        status = GetObsStatus(Observation)
        if status in warnstates:
            print('%s is missing instrument [%s]', GetObsID(Observation), status)
        else: 
            print('%s is missing instrument [%s]', GetObsID(Observation), status)
        return None
    
    instrument = instrumentcontainer.attrib.get('name')
    #logger.debug('Instrument = %s', instrument)
    if instrument == 'GMOS-N':
        subtype = 'GMOS'
    elif instrument == 'GMOS-S':
        subtype = 'GMOSSouth'
    elif instrument == 'Visitor Instrument':
        subtype = 'Visitor'
    else:
        subtype = instrument
    #logger.debug('subtype = %s', subtype)

    config = {}
    for container in Observation.findall(".//container[@subtype='%s']" % subtype):
        #logger.debug('Container: type = %s, name = %s', container.attrib.get('type'), container.attrib.get('name'))
        paramset = container.find('paramset')
        #logger.debug('Paramset: %s', paramset.attrib.get('name'))

        maxconfiglength = 0
        for param in paramset.findall('param'):
            name  = param.attrib.get('name')
            value = param.attrib.get('value')
            #logger.debug('Param %s = %s', name, value)
            if value and value != None: # catch empty string (empty iterator)
                add(config, name, value)
                configlength = len(config[name])
                #logger.debug('   -> Config Length = %s', configlength)
                if configlength > maxconfiglength: maxconfiglength = configlength
            else: # multiple values
                for val in param.findall('value'):
                    #logger.debug('...val = %s', val.text)
                    if val.text is not None:
                        add(config, name, val.text)
                        configlength = len(config[name])
                        #logger.debug('      -> Config Length = %s', configlength)
                        if configlength > maxconfiglength: maxconfiglength = configlength

        # I'm not 100% about this, but it seems that the decker in the top-level OT component
        # is sometimes incorrectly set to acquisition.  Here I reset the *TOP LEVEL* GNIRS
        # decker component based on the camera and XD, as I think it should be.
        # I think the reason this OT bug is okay is because the seqexec automatically sets the 
        # decker based on the camera and XD if it is not explicitly set.
        # PS. Don't use the camera, it's wrong.  Use the pixel scale.
        if  container.attrib.get('name') == 'GNIRS' and \
            container.attrib.get('type') == 'Instrument' and \
            config['decker'] == ['ACQUISITION']:
            #logger.debug('++++++++++++++++++++++++++++++++++++')
            #logger.debug('config[decker] = %s', config['decker'])
            #logger.debug('config[pixelScale] = %s', config['pixelScale'])
            #logger.debug('config[crossDispersed] = %s', config['crossDispersed'])

            if config['pixelScale'] == ['PS_015'] and config['crossDispersed'] == ['NO']:
                new = ['SHORT_CAM_LONG_SLIT']
            elif config['pixelScale'] == ['PS_015'] and config['crossDispersed'] == ['SXD']:
                new = ['SHORT_CAM_X_DISP']
            elif config['pixelScale'] == ['PS_005'] and config['crossDispersed'] == ['NO']:
                new = ['LONG_CAM_LONG_SLIT']
            elif config['pixelScale'] == ['PS_005'] and config['crossDispersed'] == ['LXD']:
                new = ['LONG_CAM_X_DISP']
            elif config['pixelScale'] == ['PS_005'] and config['crossDispersed'] == ['SXD']:
                new = ['SHORT_CAM_X_DISP']
            else:
                #logger.error('%s has unknown GNIRS config: %s + %s [%s]', self.GetObsID(Observation), config['pixelScale'], config['crossDispersed'], GetObsStatus(Observation))
                new = ['UNKNOWN']
            #logger.debug('Fixing GNIRS decker: ACQUISITION -> %s', new)
            config['decker'] = new
            #logger.debug('config[decker] = %s', config['decker'])

        # For each new configuration *all* configurations must be incremented,
        # using the last value if a componenet is not modified.
        #logger.debug('Updating non-modified components to have len = %d:', maxconfiglength)
        for name in list(config.keys()):
            #logger.debug('...len(config[%s]) = %d', name, len(config[name]))
            if len(config[name]) < maxconfiglength:
                #logger.debug('......extending %s by %d', name, maxconfiglength - len(config[name]))
                config[name].extend([config[name][-1] for i in range(maxconfiglength - len(config[name]))])

    #logger.debug('Config = %s', config)
    return config


    #logger = logging.getLogger('odb.GetObsTooStatus')

    if tooType == 'none':
        too = 'None'

    elif tooType == 'standard':
        too = 'Standard'

    else:
        paramset = Observation.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get('name') == 'tooOverrideRapid':
                override = param.attrib.get('value')
                if override == 'false':
                    too = 'Rapid'
                elif override == 'true':
                    too = 'Standard'
                else:
                    #logger.error('Unknown TOO status: %s', too)
                    print('Unknown TOO status: %s', too)

    #logger.debug('ToO Status = %s', too)
    return(too)

def GetConditions(Observation, label=False):

    container = Observation.find(".//container[@kind='obsComp'][@name='Observing Conditions']")

    if container is None:
        return None
    
    cc = iq = bg = wv = 'NULL'

    paramset = container.find('paramset')
    for param in paramset.findall('param'):
        if param.attrib.get('name') == 'CloudCover':
            cc = param.attrib.get("value")
        elif param.attrib.get('name') == 'ImageQuality':
            iq = param.attrib.get("value")
        elif param.attrib.get('name') == 'SkyBackground':
            bg = param.attrib.get("value")
        elif param.attrib.get('name') == 'WaterVapor':
            wv = param.attrib.get("value")

    cc = cc.replace('PERCENT_','')
    iq = iq.replace('PERCENT_','')
    bg = bg.replace('PERCENT_','')
    wv = wv.replace('PERCENT_','')

    if label:
        conditions = 'IQ' + iq + ',CC' + cc + ',BG' + bg + ',WV' + wv
    else:
        conditions = iq + ',' + cc + ',' + bg + ',' + wv

    #logger.debug('Conditions = %s', conditions)

    return conditions

def GetElevation(Observation):
    #logger = logging.getLogger('odb.GetElevation')

    container = Observation.find(".//container[@kind='obsComp'][@name='Observing Conditions']")

    if container is None:
        return None

    etype = emin = emax = 'NONE'

    paramset = container.find('paramset')
    for param in paramset.findall('param'):
        if param.attrib.get('name') == 'ElevationConstraintType':
            etype = param.attrib.get("value")
        elif param.attrib.get('name') == 'ElevationConstraintMin':
            emin = param.attrib.get("value")
        elif param.attrib.get('name') == 'ElevationConstraintMax':
            emax = param.attrib.get("value")

    #logger.debug('Elevation = %s %s %s', etype, emin, emax)

    return etype, emin, emax

def SumObsTime(tree, exptime, coadds, repeat, noffsets, ope, fwo, obstime):
    #logger = logging.getLogger('odb.SumObsTime')

    name = tree.attrib.get('name')
    #logger.debug('NAME = %s', name)

    if name == 'GMOS-N Sequence' or \
        name == 'NIFS Sequence' or \
        name == 'NICI Sequence' or \
        name == 'NIRI Sequence' or \
        name == 'GSAOI Sequence' or \
        name == 'GNIRS Sequence' or \
        name == 'Flamingos2 Sequence':
        paramset = tree.find('paramset')
        exptimeset = False
        for param in paramset.findall('param'):

            if param.attrib.get("name") == 'exposureTime':
                exptimeset = True
                #logger.debug('...embedded exptime = %s', param.attrib.get('value'))

                if str(param.attrib.get('value')) != 'None' and str(param.attrib.get('value')) != '':
                    #logger.debug('...using embedded value')
                    exptime[0] = float(param.attrib.get('value'))

                else:
                    #logger.debug('...extracting exptime from sequence')
                    i = 0
                    for val in param.findall('value'):
                        #logger.debug('...exptime[%i] = %s', i, val.text)
                        if i==0:
                            exptime[0] = float(val.text)
                        elif i>0:
                            exptime.append(float(val.text))
                        i += 1

            elif param.attrib.get("name") == 'coadds':
                #logger.debug('...embedded coadds = %s', param.attrib.get('value'))
                if str(param.attrib.get('value')) != 'None':
                    #logger.debug('...using embedded value')
                    coadds[0] = float(param.attrib.get('value'))
                else:
                    #logger.debug('...extracting coadds from sequence')
                    i = 0
                    for val in param.findall('value'):
                        #logger.debug('...coadds[%i] = %s', i, val.text)
                        if i==0:
                            coadds[0] = int(val.text)
                        elif i>0:
                            coadds.append(int(val.text))
                        i += 1

        if not exptimeset:
            i = 0
            param = paramset.find('param') # take the first since it doesn't matter which
            for val in param.findall('value'):
                #logger.debug('...VAL = %s', val.text)
                if i>0:
                    exptime.append(exptime[len(exptime)-1])
                i += 1
        #logger.debug('...exptimes = %s', exptime)

    elif name == 'Repeat':
        paramset = tree.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get("name") == 'repeatCount':
                repeat = int(param.attrib.get('value'))
                #logger.debug('...repeat = %i', repeat)

    elif name == 'Offset':
        noffsets = 0
        paramset1 = tree.find('paramset')
        paramset2 = paramset1.find('paramset')
        for paramset in paramset2.findall('paramset'):
            noffsets += 1
        #logger.debug('...noffsets = %i', noffsets)

    elif name == 'Observe':
        #logger.debug('Calculating observation time...')
        paramset = tree.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get("name") == 'repeatCount':
                nobserve = int(param.attrib.get('value'))
                #logger.debug('...nobserve = %i', nobserve)

        #logger.debug('...repeat = %i', repeat)
        #logger.debug('...exptime = %s', exptime)
        #logger.debug('...coadds = %s', coadds)
        #logger.debug('...noffsets = %i', noffsets)
        #logger.debug('...overhead/exposure = %f', ope)
        #logger.debug('...nobserve = %i', nobserve)

        i = 0
        for e in exptime:
            if len(coadds) > 1:
                nc = coadds[i]
            else:
                nc = coadds[0]
            #logger.debug('...exptime = %f   coadds = %i', e, nc)
            obstime = obstime + repeat * noffsets * nobserve * (fwo + nc * (e + ope)) + noffsets * 7.1
            #logger.debug('...obstime = %f (%s)', obstime, str(datetime.timedelta(seconds=obstime)))
            i += 1

    return(exptime, coadds, repeat, noffsets, obstime)

def GetTargetCoords(Observation):
    

    targetEnv = Observation.find(".//paramset[@name='targetEnv']")
    
    if targetEnv is None:  # No target component
        status = GetObsStatus(Observation)
        if status in warnstates and list(set(GetClass(Observation))) != ['DAY_CAL']:
            print(f'{GetObsID(Observation)} is missing target component [{status}]')
        else:
            print(f'{GetObsID(Observation)} is missing target component [{status}]')
        name = ra = dec = None
    else:
        asterism = targetEnv.find("paramset[@name='asterism']")
        target = asterism.find("paramset[@name='target']")
        name = target.find("param[@name='name']").attrib.get('value')
        #logger.debug('Name = %s', name)
        coordinates = target.find("paramset[@name='coordinates']")
        if coordinates:
            ra  = float(coordinates.find("param[@name='ra']").attrib.get('value'))
            dec = float(coordinates.find("param[@name='dec']").attrib.get('value'))
            if dec > 90:
                dec -= 360.
        else:
            #logger.warning('No coordinates') # non-sidereal target
            ra = None
            dec = None
        #logger.debug('RA = %s', ra)
        #logger.debug('Dec = %s', dec)


    return(name, ra, dec)

def GetTargetMags(Observation, baseonly=False):
    #logger = logging.getLogger('odb.GetTargetMags')
    targets = {}

    for container in Observation.findall('container'):
        if container.attrib.get('type') == 'Telescope':
            for paramset1 in container.findall('paramset'):
                if paramset1.attrib.get('name') == 'Targets':
                    for paramset2 in paramset1.findall('paramset'):
                        if paramset2.attrib.get('name') == 'targetEnv':

                            # The base target is at a higher level than the guide stars and user stars,
                            # so do a search for *all* paramsets below this level:

                            for paramset3 in paramset2.findall('.//paramset'):
                                # logger.debug('paramset3 = %s', paramset3.attrib.get('name'))
                                if  paramset3.attrib.get('name') == 'asterism' or \
                                    (not baseonly and paramset3.attrib.get('name') == 'guideEnv'):
                                    name = []
                                    mags = []

                                    for paramset4 in paramset3.findall('paramset'):
                                        # logger.debug('paramset4 = %s', paramset4.attrib.get('name'))
                                        if paramset4.attrib.get('name') == 'target':

                                            for param in paramset4.findall('param'):
                                                if param.attrib.get('name') == 'name':
                                                    name = param.attrib.get('value')
                                                    #logger.debug('Name = %s', name)

                                            for paramset5 in paramset4.findall('paramset'):
                                                # logger.debug('paramset5 = %s', paramset5.attrib.get('name'))

                                                if paramset5.attrib.get('name') == 'magnitude':
                                                    band = None
                                                    mag = None
                                                    sys = None
                                                    for param in paramset5.findall('param'):
                                                        if param.attrib.get('name') == 'band':
                                                            band = param.attrib.get('value')
                                                        elif param.attrib.get('name') == 'value':
                                                            mag = float(param.attrib.get('value'))
                                                        elif param.attrib.get('name') == 'system':
                                                            sys = param.attrib.get('value')
                                                    #logger.debug('Band = %s, Mag = %s, Sys = %s', band, mag, sys)
                                                    mags.append([band,mag,sys])

                                    if name in targets:
                                        #logger.warning('Target %s already exists; appending additional magnitudes', name)
                                        targets[name].append(mags)
                                    else:
                                        targets[name] = mags

    #logger.debug('Targets = %s', targets)
    return targets

def GetPriority(Observation):
    #logger = logging.getLogger('odb.GetPriority')
    paramset = Observation.find("paramset[@name='Observation'][@kind='dataObj']")
    try:
        param = paramset.find("param[@name='priority']")
        priority = param.attrib.get('value')
    except:
        #logger.info('%s has no priority [%s]', GetObsID(Observation), GetObsStatus(Observation))
        priority = None
    #logger.debug('Priority = %s', priority)
    return priority

def GetTargets(Observation):
    """
    Return a list of targets with a dictionary of properties:
        group (dictionary of name, tag, primary)
        name, ra, dec
        magnitudes (dictionary of band, value pairs)
        primary: whether the guide star will be used (and the group is primary)
        tag: {sidereal, asteroid, nonsidereal}
        type: {base, PWFS1, PWFS2, OIWFS, blindOffset, offAxis, tuning, other}
    """
    #logger = logging.getLogger('odb.GetTargets')

    def add(dictionary, key, value):
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

    targets = []

    targetEnv = Observation.find(".//paramset[@name='targetEnv']")

    if targetEnv is None:
        status = GetObsStatus(Observation)
        # This was throwing warnings for every specphot:
        #if status in warnstates and list(set(GetClass(Observation))) != ['DAY_CAL']:
        #    logger.warning('%s is missing target component [%s]', GetObsID(Observation), status)
        #else:
        #logger.info('%s is missing target component [%s]', GetObsID(Observation), status)
        return None

    # Base -----------------------------------------------------------------------------------------
    
    asterism = targetEnv.find("paramset[@name='asterism']")
    target = asterism.find("paramset[@name='target']")
    base = {}
    magnitudes = {}

    for param in target.findall("param"):
        base[param.attrib.get('name')] = param.attrib.get('value')

    for paramset in target.findall("paramset[@name='magnitude']"):
        for param in paramset.findall("param"):
            if param.attrib.get('name') == 'band':
                band = param.attrib.get('value')
            elif param.attrib.get('name') == 'value':
                value = param.attrib.get('value')
        magnitudes[band] = value
    base['magnitudes'] = magnitudes

    for paramset in \
        target.findall("paramset[@name='coordinates']") + \
        target.findall("paramset[@name='horizons-designation']") + \
        target.findall("paramset[@name='proper-motion']"):
        for param in paramset.findall("param"):

            if param.attrib.get('name') == 'dec':
                # The OT stores the dec as a positive float from 0 - 360.
                # If the dec is > 90 then subtract 360 to shift it into the range -90 < d < +90
                dec = float(param.attrib.get('value'))
                value = str(dec if dec < 90.0 else dec - 360.)
                #logger.debug ('Dec: %s -> %s', dec, value)
            else:
                value = param.attrib.get('value')
            
            base[param.attrib.get('name')] = value

    if base['tag'] == 'nonsidereal':
        status = GetObsStatus(Observation)
        if status != 'OBSERVED':
            print(f"{GetObsID(Observation)} {base['name']} has no HORIZONS ID [{status}]")

    base['type'] = 'base'
    base['group'] = {'name':'Base'}
    #logger.debug('base = %s', base)
    targets.append(base)

    # Guide Environment ----------------------------------------------------------------------------

    guideEnv = targetEnv.find("paramset[@name='guideEnv']")
    
    primary_group = int(guideEnv.find("param[@name='primary']").attrib.get('value'))
    #logger.debug('Primary group = %s', primary_group)

    group_number = -1
    for guideGroup in guideEnv.findall("paramset[@name='guideGroup']"):
        group_number += 1
        group = {}
        for param in guideGroup.findall("param"): # name, tag (auto, manual)
            group[param.attrib.get('name')] = param.attrib.get('value')

        if 'Auto' in group['tag']: # "AutoActiveTag" or "AutoInitialTag" if AGS is disabled
            group['name'] = 'Auto'

        if group_number == primary_group:
            group['primary'] = True
        else:
            group['primary'] = False

        for guider in guideGroup.findall("paramset[@name='guider']"):
            
            guidername = guider.find("param[@name='key']").attrib.get('value')
            try:
                primary_guidestar = int(guider.find("param[@name='primary']").attrib.get('value'))
            except:
                primary_guidestar = -1 # no active star for this guider
            #logger.debug('Primary guide star = %s', primary_guidestar)

            guidestar_number = -1
            for spTarget in guider.findall("paramset[@name='spTarget']"):
                target = spTarget.find("paramset[@name='target']")
                guidestar_number += 1
                star = {}
                star['group'] = group
                star['type'] = guidername
                if guidestar_number == primary_guidestar:
                    star['primary'] = True
                else:
                    star['primary'] = False                  
                magnitudes = {}

                for param in target.findall("param"):
                    star[param.attrib.get('name')] = param.attrib.get('value')

                for paramset in target.findall("paramset[@name='magnitude']"):
                    for param in paramset.findall("param"):
                        if param.attrib.get('name') == 'band':
                            band = param.attrib.get('value')
                        elif param.attrib.get('name') == 'value':
                            value = param.attrib.get('value')
                    magnitudes[band] = value
                star['magnitudes'] = magnitudes

                for paramset in \
                        target.findall("paramset[@name='coordinates']") + \
                        target.findall("paramset[@name='horizons-designation']") + \
                        target.findall("paramset[@name='proper-motion']"):
                    for param in paramset.findall("param"):

                        if param.attrib.get('name') == 'dec':
                            # The OT stores the dec as a positive float from 0 - 360.
                            # If the dec is > 90 then subtract 360 to shift it into the range -90 < d < +90
                            dec = float(param.attrib.get('value'))
                            value = str(dec if dec < 90.0 else dec - 360.)
                            #logger.debug ('Dec: %s -> %s', dec, value)
                        else:
                            value = param.attrib.get('value')

                        star[param.attrib.get('name')] = value

                if star['tag'] == 'nonsidereal':
                    status = GetObsStatus(Observation)
                    if status != 'OBSERVED':
                        print('%s %s has no HORIZONS ID [%s]', GetObsID(Observation), base['name'], status)

                #if star['type'] in ('GNIRS OIWFS', 'NIFS OIWFS', 'NIRI OIWFS'):
                #    logger.warning('%s has a %s defined [%s]', GetObsID(Observation), star['type'], GetObsStatus(Observation))

                star['magnitudes'] = magnitudes
                #logger.debug('star = %s', star)
                targets.append(star)


    # User targets ---------------------------------------------------------------------------------

    userTargets = targetEnv.find("paramset[@name='userTargets']")

    if userTargets is not None:

        for userTarget in userTargets.findall("paramset[@name='userTarget']"):

            star = {}
            star['group'] = {'name':'User'}
            magnitudes = {}

            for param in userTarget.findall("param"):
                star[param.attrib.get('name')] = param.attrib.get('value')

            spTarget = userTarget.find("paramset[@name='spTarget']")
            target = spTarget.find("paramset[@name='target']")

            for param in target.findall("param"):
                star[param.attrib.get('name')] = param.attrib.get('value')

            for paramset in target.findall("paramset[@name='magnitude']"):
                for param in paramset.findall("param"):
                    if param.attrib.get('name') == 'band':
                        band = param.attrib.get('value')
                    elif param.attrib.get('name') == 'value':
                        value = param.attrib.get('value')
                magnitudes[band] = value
            star['magnitudes'] = magnitudes

            for paramset in \
                    target.findall("paramset[@name='coordinates']") + \
                    target.findall("paramset[@name='horizons-designation']") + \
                    target.findall("paramset[@name='proper-motion']"):
                for param in paramset.findall("param"):

                    if param.attrib.get('name') == 'dec':
                        # The OT stores the dec as a positive float from 0 - 360.
                        # If the dec is > 90 then subtract 360 to shift it into the range -90 < d < +90
                        dec = float(param.attrib.get('value'))
                        value = str(dec if dec < 90.0 else dec - 360.)
                        #logger.debug ('Dec: %s -> %s', dec, value)
                    else:
                        value = param.attrib.get('value')

                    star[param.attrib.get('name')] = value

            if star['tag'] == 'nonsidereal':
                status = GetObsStatus(Observation)
                if status != 'OBSERVED':
                    print('%s %s has no HORIZONS ID [%s]', GetObsID(Observation), base['name'], status)

            #logger.debug('star = %s', star)
            targets.append(star)

    #logger.debug('Targets = %s', targets)
    return targets

def GetWindows(Observation):

    start = []
    duration = []
    repeat = []
    period = []

    for container in Observation.findall('container'):
        if container.attrib.get('name') == 'Observing Conditions':
            observing_conditions = container.find('paramset')
            timing_window_list = observing_conditions.find('paramset')
            for timing_window in timing_window_list.findall('paramset'):
                for param in timing_window.findall('param'):
                    if param.attrib.get('name') == 'start':
                        window_start = int(param.attrib.get("value"))
                    elif param.attrib.get('name') == 'duration':
                        window_duration = int(param.attrib.get("value"))
                    elif param.attrib.get('name') == 'repeat':
                        window_repeat = int(param.attrib.get("value"))
                    elif param.attrib.get('name') == 'period':
                        window_period = int(param.attrib.get("value"))

                start.append(window_start)
                duration.append(window_duration)
                repeat.append(window_repeat)
                period.append(window_period)

    return (start, duration, repeat, period)

def GetObsTooStatus(Observation, tooType):
    
    if tooType == 'none':
        too = 'None'

    elif tooType == 'standard':
        too = 'Standard'

    else:
        paramset = Observation.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get('name') == 'tooOverrideRapid':
                override = param.attrib.get('value')
                if override == 'false':
                    too = 'Rapid'
                elif override == 'true':
                    too = 'Standard'
                else:
                    print('Unknown TOO status: %s', too)

    return(too)

def CheckStatus(Program):
    
    paramset = Program.find('paramset')
    for param in paramset.findall('param'):
        if param.attrib.get('name') == 'fetched':
            if param.attrib.get('value') == 'true':
                active = True
            else:
                active = False
            #print(f'Active = {active}')

        if param.attrib.get('name') == 'completed':
            if param.attrib.get('value') == 'true':
                complete = True
            else:
                complete = False
            #print(f'Complete = {complete}')

    return active, complete

#### CREATED BY ST for Gmax Protoype

def GetFTProgramDates(Notes, semester, year, yp):

    progstart, progend = None, None    
    def monthnum(month, months):
        month = month.lower()
        return [i for i, m in enumerate(months) if month in m].pop()+1

    months_list = list(map(lambda x: x.lower(), calendar.month_name[1:])) 
    for title, _ in Notes:
        if title:
            if 'cycle' in title.lower() or 'active' in title.lower():
                fields = title.strip().split(' ')
                months = []
                for field in fields:
                    if '-' in field:
                        months = field.split('-')
                        if len(months) == 3:
                            break
                if len(months) == 0:
                    for field in fields:
                        f = field.lower().strip(' ,')
                        for jj in range(len(months_list)):
                            if f in months_list[jj]:
                                months.append(f)
            
                im1 = monthnum(months[0], months_list)
                m1 = '{:02d}'.format(im1)
                im2 = monthnum(months[-1],months_list)
                m2 = '{:02d}'.format(im2)
                if semester == 'B' and im1 < 6:

                    progstart = Time(yp + "-" + m1 + "-01 00:00:00", format='iso')
                    d2 = '{:02d}'.format(calendar.monthrange(int(yp), im2)[1])
                    progend = Time(yp + "-" + m2 + "-" + d2 + " 00:00:00", format='iso')
                else:
                    #                             print(y, im1, yp, im2)
                    progstart = Time(year + "-" + m1 + "-01 00:00:00", format='iso')
                    if im2 > im1:
                        d2 = '{:02d}'.format(calendar.monthrange(int(year), im2)[1])
                        progend = Time(year + "-" + m2 + "-" + d2 + " 00:00:00", format='iso')
                    else:
                        d2 = '{:02d}'.format(calendar.monthrange(int(yp), im2)[1])
                        progend = Time(yp + "-" + m2 + "-" + d2 + " 00:00:00", format='iso')
                break
    
    return progstart, progend

def GetObservationInfo(XMLProgram):
    raw_info, schedgrps = [],[]
    for container in XMLProgram.findall('container'):
        if container.attrib.get('type') == 'Observation':
            grpkey = container.attrib.get("key")
            grpname = 'None'
            # for paramset in container.findall('paramset'):
            paramset = container.find('paramset')
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'title':
                    grpname = param.attrib.get('value')
                    break
            # self.schedgroup[grpkey] = {'name': grpname, 'idx': []}
            # self.programs[prgid]['groups'].append(grpkey)
            schedgrp = {'key': grpkey, 'name': grpname}
            #self.process_observation(container, schedgrp, selection=selection, obsclasses=obsclasses,
            #                        tas=tas, odbplan=odbplan)
            raw_info.append(container)
            schedgrps.append(schedgrp)
            #return container, schedgrp
        elif container.attrib.get('type') == 'Group':
            grpkey = container.attrib.get("key")
            grptype = 'None'
            grpname = 'None'
            paramset = container.find('paramset')
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'title':
                    grpname = param.attrib.get('value')
                if param.attrib.get('name') == 'GroupType':
                    grptype = param.attrib.get('value')
            # print(grptype)
            if grptype == 'TYPE_SCHEDULING':
                schedgrp = {'key': grpkey, 'name': grpname}
                # self.schedgroup[grpkey] = {'name': grpname, 'idx': []}
                # self.programs[prgid]['groups'].append(grpkey)
            for subcontainer in container.findall('container'):
                if grptype != 'TYPE_SCHEDULING':
                    grpkey = subcontainer.attrib.get("key")
                    grpname = 'None'
                    paramset = subcontainer.find('paramset')
                    for param in paramset.findall('param'):
                        if param.attrib.get('name') == 'title':
                            grpname = param.attrib.get('value')
                            break
                    # self.schedgroup[grpkey] = {'name': grpname, 'idx': []}
                    # self.programs[prgid]['groups'].append(grpkey)
                    schedgrp = {'key': grpkey, 'name': grpname}
                if subcontainer.attrib.get("type") == 'Observation':
                    #return subcontainer, schedgrp
                    raw_info.append(subcontainer)
                    schedgrps.append(schedgrp)
            
    return raw_info, schedgrps


### GMOS FILE
def FpuXmlTranslator(xmlfpu):
    """
    Convert the ODB XML FPU name to a human-readable mask name.
    Input: list of XML FPU names
    Output: list of dictionaries of FPU names and widths
    """


    fpu = []

    for f in xmlfpu:
        if   f == 'FPU_NONE':
            fpu.append({'name':'None', 'width':None})
        elif f == 'LONGSLIT_1':
            fpu.append({'name':'0.25arcsec', 'width':0.25})
        elif f == 'LONGSLIT_2':
            fpu.append({'name':'0.5arcsec', 'width':0.5})
        elif f == 'LONGSLIT_3':
            fpu.append({'name':'0.75arcsec', 'width':0.75})
        elif f == 'LONGSLIT_4':
            fpu.append({'name':'1.0arcsec', 'width':1.0})
        elif f == 'LONGSLIT_5':
            fpu.append({'name':'1.5arcsec', 'width':1.5})
        elif f == 'LONGSLIT_6':
            fpu.append({'name':'2.0arcsec', 'width':2.0})
        elif f == 'LONGSLIT_7':
            fpu.append({'name':'5.0arcsec', 'width':5.0})
        elif f == 'IFU_1':
            fpu.append({'name':'IFU-2', 'width':5.0})
        elif f == 'IFU_2':
            fpu.append({'name':'IFU-B', 'width':3.5})
        elif f == 'IFU_3':
            fpu.append({'name':'IFU-R', 'width':3.5})
        elif f == 'NS_0':
            fpu.append({'name':'NS0.25arcsec', 'width':0.25})
        elif f == 'NS_1':
            fpu.append({'name':'NS0.5arcsec', 'width':0.5})
        elif f == 'NS_2':
            fpu.append({'name':'NS0.75arcsec', 'width':0.75})
        elif f == 'NS_3':
            fpu.append({'name':'NS1.0arcsec', 'width':1.0})
        elif f == 'NS_4':
            fpu.append({'name':'NS1.5arcsec', 'width':1.5})
        elif f == 'NS_5':
            fpu.append({'name':'NS2.0arcsec', 'width':2.0})
        elif f == 'CUSTOM_MASK':
            fpu.append({'name':'CUSTOM_MASK', 'width':None})
        else:
            fpu.append({'name':'UNKNOWN', 'width':None})


    return fpu

def CustomMaskWidth(customwidthstring):

    try:
        width = float(customwidthstring[13:].replace('_','.'))
    except:
        width = None
    return width  