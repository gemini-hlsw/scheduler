from datetime import datetime, timedelta
import calendar
import logging
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree

from astropy.time import Time
import astropy.units as u

from common.helpers import str_to_bool
from common.structures.band import Band
from common.structures.execution_status import ExecutionStatus
from common.structures.observation_status import ObservationStatus
from common.structures.phase2_status import Phase2Status
from common.structures.time_award_units import TimeAwardUnits
from common.structures.too_type import ToOType

CLASSICAL_NIGHT_LEN = 10

WARNING_STATES = frozenset([ObservationStatus.FOR_ACTIVATION,
                            ObservationStatus.ONGOING,
                            ObservationStatus.READY])


# ORIGINAL source: https://github.com/bryanmiller/odb
def get_program_id(program_data: ElementTree) -> str:
    """Extract the program ID from XML."""
    return program_data.attrib.get('name')


def get_program_notes(program_data: ElementTree) -> List[Tuple[Optional[str], Optional[str]]]:
    """Extract program notes from XML."""
    notes = []
    for container in program_data.findall('container'):
        if container.attrib.get('type') == 'Info':
            paramset = container.find('paramset')
            title = None
            content = None
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'title':
                    title = param.attrib.get('value')
                elif param.attrib.get('name') == 'NoteText':
                    content = param.attrib.get('value')
                    if content is None:  # long content will be in a separate <value> tag
                        value = param.find('value')
                        content = value.text
            notes.append((title, content))
    return notes


# TODO: We should probably make program mode an enum.
# TODO: Talk to BRYAN. Right now the only value is QUEUE.
def get_program_mode(program_data: ElementTree) -> str:
    """Extract the program mode from XML."""
    mode = 'UNKNOWN'
    for paramset in program_data.findall('paramset'):
        if paramset.attrib.get('name') == 'Science Program':
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'programMode':
                    mode = param.attrib.get('value')
                    break
    return mode


def get_program_band(program_data: ElementTree) -> Band:
    """Extract the program band from XML."""
    band = 0
    for paramset in program_data.findall('paramset'):
        if paramset.attrib.get('name') == 'Science Program':
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'queueBand':
                    band = int(param.attrib.get('value'))
                    break
    return Band(band)


def get_program_awarded_time(program_data: ElementTree) -> Time:
    """Extract the program awarded time from XML."""
    sciprogram = program_data.find("paramset[@name='Science Program'][@kind='dataObj']")
    # time_acct = sciprogram.find(paramset[@name='timeAcct']")
    awarded_time = sciprogram.find("param[@name='awardedTime']")
    raw_value = float(awarded_time.attrib.get('value'))
    raw_units = TimeAwardUnits(awarded_time.attrib.get('units'))
    return 0.0 * u.hour if raw_value is None or raw_units is None else\
        float(raw_value) * (CLASSICAL_NIGHT_LEN if TimeAwardUnits(raw_units) == TimeAwardUnits.NIGHTS else 1) * u.hour


def is_program_thesis(program_data: ElementTree) -> bool:
    """Determine if the program is a thesis program from XML."""
    paramset = program_data.find("paramset[@name='Science Program'][@kind='dataObj']")
    param = paramset.find("param[@name='isThesis']")
    return param is not None and str_to_bool(param.attrib.get('value'))


def get_too_status(program_data: ElementTree) -> ToOType:
    """Get the target of opportunity status for the program from XML."""
    too = ToOType.NONE
    for paramset in program_data.findall('paramset'):
        if paramset.attrib.get('name') == 'Science Program':
            for param in paramset.findall('param'):
                if param.attrib.get('name') == 'tooType':
                    too = param.attrib.get('value')
    return too


# TODO: What are the possible observation classes?
def get_obs_class(observation_data: ElementTree) -> List[str]:
    """Get the observation classes from the XML."""
    obsclasses = []
    for container in observation_data.findall('.//container'):
        if container.attrib.get("type") == 'Observer':
            paramset = container.find("paramset")
            for param in paramset.findall("param"):
                if param.attrib.get("name") == "class":
                    obsclasses.append(param.attrib.get("value"))
    return obsclasses


def get_obs_status(observation_data: ElementTree) -> ObservationStatus:
    """Get the observation status from the XML."""
    execstatus = ExecutionStatus.AUTO
    paramset = observation_data.find('paramset')
    for param in paramset.findall('param'):

        # In 2014A the Status was split into Phase-2 and Exec status
        # Here I recombine them into one as they were before 2014A:
        if param.attrib.get('name') == 'phase2Status':
            phase2status = Phase2Status[param.attrib.get('value')]

        if param.attrib.get('name') == 'execStatusOverride':
            execstatus = ExecutionStatus[param.attrib.get('value')]

    if execstatus == ExecutionStatus.OBSERVED:
        obsstatus = ObservationStatus.OBSERVED

    elif execstatus == ExecutionStatus.ONGOING:
        obsstatus = ObservationStatus.ONGOING

    elif execstatus == ExecutionStatus.PENDING:
        obsstatus = ObservationStatus.READY

    elif execstatus == ExecutionStatus.AUTO:
        # TODO ERROR: This will not be assigned.
        if phase2status == Phase2Status.PI_TO_COMPLETE:
            obsstatus = ObservationStatus.PHASE2
        elif phase2status == Phase2Status.NGO_TO_REVIEW:
            obsstatus = ObservationStatus.FOR_REVIEW
        elif phase2status == Phase2Status.NGO_IN_REVIEW:
            obsstatus = ObservationStatus.IN_REVIEW
        elif phase2status == Phase2Status.GEMINI_TO_ACTIVATE:
            obsstatus = ObservationStatus.FOR_ACTIVATION
        elif phase2status == Phase2Status.ON_HOLD:
            obsstatus = ObservationStatus.ON_HOLD
        elif phase2status == Phase2Status.INACTIVE:
            obsstatus = ObservationStatus.INACTIVE
        elif phase2status == Phase2Status.PHASE_2_COMPLETE:
            obslog = get_obs_log(observation_data)  # returns a triple: (time, event, datalabel)
            nsteps = get_num_steps(observation_data)
            nobs = get_num_observed(observation_data)

            if nobs == 0 and len(obslog[0]) == 0:
                obsstatus = ObservationStatus.READY

            elif nobs >= nsteps:
                obsstatus = ObservationStatus.OBSERVED

            else:
                obsstatus = ObservationStatus.ONGOING

        else:
            logging.error(f'Unknown Phase2 status: {phase2status}')

    # TODO ERROR: obsstatus may not be initialized.
    return obsstatus


def get_obs_log(observation_data: ElementTree) -> Tuple[List[datetime], List[str], List[str]]:
    time = []
    event = []
    datalabel = []

    for container in observation_data.findall('container'):
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
                                                time.append(datetime.utcfromtimestamp(
                                                    int(param.attrib.get('value')) / 1000.))
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

    return time, event, datalabel


def get_num_steps(observation_data: ElementTree) -> int:
    """Get the number of steps from XML."""
    def count(tree: ElementTree,
              total: int,
              multiplier: int,
              obsid: int) -> Tuple[int, int]:
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
                if nstep == 0:  # If the iterator only has a single step there won't be any "val" parameters
                    nstep = 1

            if nstep is None:
                logging.warning(f'{obsid}: {cname} has zero steps')
            else:
                multiplier *= nstep

        # If container is a Repeat iterator:
        elif cname == 'Repeat':
            paramset = tree.find('paramset')
            for param in paramset.findall('param'):
                if param.attrib.get("name") == 'repeatCount':
                    nrepeats = int(param.attrib.get('value'))
            # TODO ERROR: nrepeats may not be defined.
            multiplier *= nrepeats

        # If container is an Offset iterator:
        elif cname == 'Offset' or cname == 'NICI Offset':
            noffsets = 0
            paramset1 = tree.find('paramset')
            for paramset2 in paramset1.findall('paramset'):
                noffsets += len(paramset2.findall('paramset'))
            if noffsets == 0:
                logging.warning(f'{obsid}: Offset iterator has zero steps')
            else:
                multiplier *= noffsets

        # If container is an Observe/Dark/Flat:
        elif ctype == 'Observer':
            paramset = tree.find('paramset')
            for param in paramset.findall('param'):
                if param.attrib.get("name") == 'repeatCount':
                    nobserves = int(param.attrib.get('value'))

            # TODO ERROR: nobserves might not be initialized.
            total += nobserves * multiplier

        else:
            logging.error(f'Unknown container: {cname}')

        return total, multiplier

    total = 0
    multiplier1 = 1
    obsid = observation_data.attrib.get('name')  # to pass for error reporting

    for container1 in observation_data.findall('container'):
        if container1.attrib.get("name") == 'Sequence':  # top-level sequence

            # TODO: All these weird assignments to multiplier# should not be necessary.
            for container2 in container1.findall("container"):
                if container2.attrib.get("kind") == "seqComp":
                    multiplier2 = multiplier1
                    (total, multiplier2) = count(container2, total, multiplier2, obsid)

                    for container3 in container2.findall("container"):
                        if container3.attrib.get("kind") == "seqComp":
                            multiplier3 = multiplier2
                            (total, multiplier3) = count(container3, total, multiplier3, obsid)

                            for container4 in container3.findall("container"):
                                if container4.attrib.get("kind") == "seqComp":
                                    multiplier4 = multiplier3
                                    (total, multiplier4) = count(container4, total, multiplier4, obsid)

                                    for container5 in container4.findall("container"):
                                        if container5.attrib.get("kind") == "seqComp":
                                            multiplier5 = multiplier4
                                            (total, multiplier5) = count(container5, total, multiplier5, obsid)

                                            for container6 in container5.findall("container"):
                                                if container6.attrib.get("kind") == "seqComp":
                                                    multiplier6 = multiplier5
                                                    (total, multiplier6) = count(container6, total, multiplier6, obsid)

                                                    for container7 in container6.findall("container"):
                                                        if container7.attrib.get("kind") == "seqComp":
                                                            multiplier7 = multiplier6
                                                            (total, multiplier7) = count(container7, total, multiplier7,
                                                                                         obsid)

                                                            for container8 in container7.findall("container"):
                                                                if container8.attrib.get("kind") == "seqComp":
                                                                    logging.error(f'{obsid}: Too many nested loops.')

    if total == 0:
        logging.warning(f'{obsid} has zero steps.')

    return total


def get_num_observed(observation_data: ElementTree) -> int:
    """Get number observed from XML."""
    nsteps = 0

    for container in observation_data.findall('container'):
        if container.attrib.get('type') == 'ObsLog':
            obs_log = container
            obs_exec_log = obs_log.find('paramset')
            obs_exec_record = obs_exec_log.find('paramset')

            for paramset1 in obs_exec_record.findall('paramset'):
                if paramset1.attrib.get('name') == 'configMap':

                    for paramset2 in paramset1.findall('paramset'):
                        if paramset2.attrib.get('name') == 'configMapEntry':

                            for param in paramset2.findall('param'):
                                if param.attrib.get('name') == 'datasetLabels':
                                    if param.attrib.get('value'):  # single value
                                        nsteps += 1
                                    else:  # multiple values
                                        nsteps += len(param.findall('value'))

    return nsteps


def get_obs_id(observation_data: ElementTree) -> str:
    """Get observation ID from XML."""
    obsid = observation_data.attrib.get('name')
    return obsid if obsid else 'UNKNOWN'


# TODO: This function is way too long.
def get_obs_time(observation_data: ElementTree) -> timedelta:
    # TO-DO:
    # Include overhead for filter changes (specifically 50s for F2)
    # Include iterating over the read mode

    obstime = 0  # total observation time
    ope = 0  # overhead per exposure
    coadds = [1]  # list of coadds
    exptime = []  # list of exposure times from instrument sequencers
    repeat = 1
    noffsets = 1  # number of offset positions

    for container in observation_data.findall('container'):
        if container.attrib.get('type') == 'Instrument':
            instrument = container.attrib.get('name')
            paramset = container.find('paramset')

            fwo = 0.0  # The file-write overhead, which I have only implemented for NIRI as of 2013 Jun 29

            if instrument == 'NIFS':
                obstime = 11 * 60  # Acquisition overhead (s)
                for param in paramset.findall("param"):
                    if param.attrib.get("name") == 'readMode':
                        readmode = param.attrib.get('value')

                # TODO ERROR: readmode may not be assigned.
                if readmode == 'FAINT_OBJECT_SPEC':
                    ope = 99
                elif readmode == 'MEDIUM_OBJECT_SPEC':
                    ope = 35
                elif readmode == 'BRIGHT_OBJECT_SPEC':
                    ope = 19.3
                else:
                    print('UNKNOWN READ MODE')

            elif instrument == 'GNIRS':
                obstime = 15 * 60  # Acquisition overhead (s)
                for param in paramset.findall("param"):
                    if param.attrib.get("name") == 'readMode':
                        readmode = param.attrib.get('value')

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
                obstime = 6 * 60  # Acquisition overhead (s)
                for param in paramset.findall("param"):
                    if param.attrib.get("name") == 'readMode':
                        readmode = param.attrib.get('value')

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

                if disperser == 'MIRROR':
                    obstime = 6 * 60  # Imaging acquisition overhead (s)
                elif 'SLIT' in fpu or 'NS' in fpu:
                    obstime = 16 * 60  # Long slit acquisition overhead (s)
                elif 'IFU' in fpu:
                    obstime = 18 * 60  # IFU acquisition overhead (s)
                elif fpu == 'CUSTOM_MASK':
                    obstime = 18 * 60  # MOS acquisition overhead (s)
                else:
                    print('Unknown acquisition overhead for %s', get_obs_id(observation_data))

                if roi == 'FULL_FRAME' or roi == 'CCD2':
                    if xbin == 'ONE' and ybin == 'ONE':
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
                    if xbin == 'ONE' and ybin == 'ONE':
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
                obstime = 10 * 60  # Acquisition overhead (s)
                ope = 8.4

            elif instrument == 'GPI':
                obstime = 10 * 60  # Acquisition overhead (s)
                ope = 20.

            elif instrument == 'GSAOI':
                obstime = 30 * 60  # Acquisition overhead (s)
                ope = 43.4

            elif instrument == 'Flamingos2':
                obstime = 15 * 60  # Acquisition overhead (s)
                filterchange = 50  # filter change overhead (s): UNIMPLEMENTED!

                for param in paramset.findall("param"):
                    if param.attrib.get("name") == 'readMode':
                        readmode = param.attrib.get('value')

                if readmode == 'BRIGHT_OBJECT_SPEC':
                    ope = 8.0
                elif readmode == 'MEDIUM_OBJECT_SPEC':
                    ope = 14.0
                elif readmode == 'FAINT_OBJECT_SPEC':
                    ope = 20.0
                else:
                    print('UNKNOWN READ MODE')

            elif instrument == 'Phoenix':
                obstime = 20 * 60  # Acquisition overhead (s)
                ope = 18.  # Overhead per exposure (s)

            elif instrument == 'Texes':
                obstime = 20 * 60  # Acquisition overhead (s)
                ope = 1.  # Overhead per exposure (s)

            elif instrument == 'Visitor Instrument':
                obstime = 10 * 60  # Acquisition overhead (s)
                ope = 1.  # Overhead per exposure (s)

            elif instrument == 'Acquisition Camera':
                obstime = 10 * 60  # Acquisition overhead (s)
                ope = 1.  # Overhead per exposure (s)

            else:
                logging.warning(f'Unknown acquisition overhead for {instrument} in {get_obs_id(observation_data)}.')

            for param in paramset.findall("param"):
                if param.attrib.get("name") == "exposureTime":
                    toplevelexptime = float(param.attrib.get("value"))
                    exptime.append(toplevelexptime)

                if param.attrib.get('name') == 'coadds':
                    coadds[0] = int(param.attrib.get('value'))

        if container.attrib.get("name") == 'Sequence':  # top-level sequence
            for container2 in container.findall("container"):
                if container2.attrib.get("kind") == "seqComp":
                    (exptime, coadds, repeat, noffsets, obstime) = sum_obs_time(container2, exptime, coadds, repeat,
                                                                                noffsets, ope, fwo, obstime)
                    for container3 in container2.findall("container"):
                        if container3.attrib.get("kind") == "seqComp":
                            (exptime, coadds, repeat, noffsets, obstime) = sum_obs_time(container3, exptime, coadds,
                                                                                        repeat, noffsets, ope, fwo,
                                                                                        obstime)
                            for container4 in container3.findall("container"):
                                if container4.attrib.get("kind") == "seqComp":
                                    (exptime, coadds, repeat, noffsets, obstime) = sum_obs_time(container4, exptime,
                                                                                                coadds, repeat, noffsets,
                                                                                                ope, fwo, obstime)
                                    for container5 in container4.findall("container"):
                                        if container5.attrib.get("kind") == "seqComp":
                                            (exptime, coadds, repeat, noffsets, obstime) = sum_obs_time(container5,
                                                                                                        exptime, coadds,
                                                                                                        repeat, noffsets,
                                                                                                        ope, fwo, obstime)

                                        # print '...resetting exptime list 5'
                                        # TODO ERROR: toplevelexptime may not be assigned.
                                        exptime = [toplevelexptime]

                                # print '...resetting exptime list 4'
                                exptime = [toplevelexptime]

                        # print '...resetting exptime list 3'
                        exptime = [toplevelexptime]

                # print '...resetting exptime list 2'
                exptime = [toplevelexptime]

    return timedelta(seconds=int(obstime))


def get_instrument(observation_data: ElementTree) -> Optional[str]:
    """Get the observation instrument from XML."""
    instrument = None
    for container in observation_data.findall('container'):
        if container.attrib.get("type") == 'Instrument':
            instrument = container.attrib.get("name")
            break

    if instrument is None or not instrument:
        status = get_obs_status(observation_data)
        if status in WARNING_STATES:
            logging.warning(f'{get_obs_id(observation_data)} is missing instrument [{status}]')
        else:
            logging.warning(f'{get_obs_id(observation_data)} is missing instrument [{status}]')
        instrument = 'INDEF'

    return instrument


# TODO: Enums desperately needed here, but as for now, I'm not touching this.
def get_inst_configs(observation_data: ElementTree):
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

    def add(dictionary, k, v):
        dictionary.setdefault(k, [])
        dictionary[k].append(v)

    # The subtype is the instrument name, except for GMOS-N (GMOS) and GMOS-S (GMOSSouth):
    instrumentcontainer = observation_data.find("container[@kind='obsComp'][@type='Instrument']")
    if instrumentcontainer is None:
        status = get_obs_status(observation_data)
        logging.error(f'{get_obs_id(observation_data)} is missing instrument [{status}]')
        return None

    instrument = instrumentcontainer.attrib.get('name')
    if instrument == 'GMOS-N':
        subtype = 'GMOS'
    elif instrument == 'GMOS-S':
        subtype = 'GMOSSouth'
    elif instrument == 'Visitor Instrument':
        subtype = 'Visitor'
    else:
        subtype = instrument

    config = {}
    for container in observation_data.findall(".//container[@subtype='%s']" % subtype):
        paramset = container.find('paramset')
        maxconfiglength = 0
        for param in paramset.findall('param'):
            name = param.attrib.get('name')
            value = param.attrib.get('value')
            if value is not None and value:
                add(config, name, value)
                configlength = len(config[name])
                if configlength > maxconfiglength:
                    maxconfiglength = configlength
            else:  # multiple values
                for val in param.findall('value'):
                    if val.text is not None:
                        add(config, name, val.text)
                        configlength = len(config[name])
                        if configlength > maxconfiglength:
                            maxconfiglength = configlength

        # I'm not 100% about this, but it seems that the decker in the top-level OT component
        # is sometimes incorrectly set to acquisition. Here I reset the *TOP LEVEL* GNIRS
        # decker component based on the camera and XD, as I think it should be.
        # I think the reason this OT bug is okay is because the seqexec automatically sets the 
        # decker based on the camera and XD if it is not explicitly set.
        # PS. Don't use the camera, it's wrong.  Use the pixel scale.
        if container.attrib.get('name') == 'GNIRS' and \
                container.attrib.get('type') == 'Instrument' and \
                config['decker'] == ['ACQUISITION']:
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
                new = ['UNKNOWN']
            config['decker'] = new

        # For each new configuration *all* configurations must be incremented,
        # using the last value if a component is not modified.
        for name in list(config.keys()):
            if len(config[name]) < maxconfiglength:
                config[name].extend([config[name][-1] for i in range(maxconfiglength - len(config[name]))])

    return config


# TODO: If we are going to keep this code, this should return a SkyConditions object instead of a string,
# TODO: and the logic from the Collector should be moved here.
def get_conditions(observation_data: ElementTree, label=False) -> Optional[str]:
    container = observation_data.find(".//container[@kind='obsComp'][@name='Observing Conditions']")

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

    cc = cc.replace('PERCENT_', '')
    iq = iq.replace('PERCENT_', '')
    bg = bg.replace('PERCENT_', '')
    wv = wv.replace('PERCENT_', '')

    if label:
        conditions = 'IQ' + iq + ',CC' + cc + ',BG' + bg + ',WV' + wv
    else:
        conditions = iq + ',' + cc + ',' + bg + ',' + wv

    return conditions


# TODO: Again, this should return an Elevation instead of a tuple of string.
def get_elevation(observation_data: ElementTree) -> Optional[Tuple[str, str, str]]:
    container = observation_data.find(".//container[@kind='obsComp'][@name='Observing Conditions']")

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

    return etype, emin, emax


# TODO: No idea what ope and fwo are?
def sum_obs_time(tree: ElementTree,
                 exptime: List[float],
                 coadds: List[float],
                 repeat: int,
                 noffsets: int,
                 ope, fwo, obstime) -> Tuple[List[float], List[float], int, int, float]:
    name = tree.attrib.get('name')
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

                # TODO: Can we just say param.attrib.value(...) is not None?
                if str(param.attrib.get('value')) != 'None' and str(param.attrib.get('value')) != '':
                    exptime[0] = float(param.attrib.get('value'))

                else:
                    for i, val in enumerate(param.findall('value')):
                        if i == 0:
                            exptime[0] = float(val.text)
                        else:
                            exptime.append(float(val.text))

            elif param.attrib.get("name") == 'coadds':
                # TODO: Again, why the str conversion?
                if str(param.attrib.get('value')) != 'None':
                    coadds[0] = float(param.attrib.get('value'))
                else:
                    for i, val in enumerate(param.findall('value')):
                        if i == 0:
                            coadds[0] = int(val.text)
                        elif i > 0:
                            coadds.append(int(val.text))

        if not exptimeset:
            param = paramset.find('param')  # take the first since it doesn't matter which
            # TODO: Simplify.
            for i in range(len(param.findall('value'))):
                if i > 0:
                    exptime.append(exptime[len(exptime) - 1])

    elif name == 'Repeat':
        paramset = tree.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get("name") == 'repeatCount':
                repeat = int(param.attrib.get('value'))

    elif name == 'Offset':
        paramset1 = tree.find('paramset')
        paramset2 = paramset1.find('paramset')
        noffsets = len(paramset2.findall('paramset'))

    elif name == 'Observe':
        paramset = tree.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get("name") == 'repeatCount':
                nobserve = int(param.attrib.get('value'))

        for i, e in enumerate(exptime):
            if len(coadds) > 1:
                nc = coadds[i]
            else:
                nc = coadds[0]
            # TODO ERROR: nobserve may not be assigned.
            obstime = obstime + repeat * noffsets * nobserve * (fwo + nc * (e + ope)) + noffsets * 7.1

    return exptime, coadds, repeat, noffsets, obstime


def get_target_coords(observation_data: ElementTree) -> Tuple[str, Optional[float], Optional[float]]:
    """Get the name, RA, and dec of a target from XML."""
    target_env = observation_data.find(".//paramset[@name='targetEnv']")

    if target_env is None:
        status = get_obs_status(observation_data)
        logging.warning(f'{get_obs_id(observation_data)} is missing target component [{status}]')
        name = ra = dec = None
    else:
        asterism = target_env.find("paramset[@name='asterism']")
        target = asterism.find("paramset[@name='target']")
        name = target.find("param[@name='name']").attrib.get('value')
        coordinates = target.find("paramset[@name='coordinates']")
        ra, dec = None, None
        if coordinates:
            ra = float(coordinates.find("param[@name='ra']").attrib.get('value'))
            dec = float(coordinates.find("param[@name='dec']").attrib.get('value'))
            if dec > 90:
                dec -= 360.

    return name, ra, dec


# TODO: The return type of this function is so convoluted that I cannot determine it.
# TODO: I have a lot of confusion about why certain things are being done here. See notes below.
def get_target_magnitudes(observation_data: ElementTree,
                          baseonly=False):
    targets = {}

    for container in observation_data.findall('container'):
        if container.attrib.get('type') == 'Telescope':
            for paramset1 in container.findall('paramset'):
                if paramset1.attrib.get('name') == 'Targets':
                    for paramset2 in paramset1.findall('paramset'):
                        if paramset2.attrib.get('name') == 'targetEnv':

                            # The base target is at a higher level than the guide stars and user stars,
                            # so do a search for *all* paramsets below this level:
                            for paramset3 in paramset2.findall('.//paramset'):
                                if paramset3.attrib.get('name') == 'asterism' or \
                                        (not baseonly and paramset3.attrib.get('name') == 'guideEnv'):
                                    # TODO: Why is name a list?
                                    name = []
                                    mags = []

                                    for paramset4 in paramset3.findall('paramset'):
                                        if paramset4.attrib.get('name') == 'target':

                                            for param in paramset4.findall('param'):
                                                if param.attrib.get('name') == 'name':
                                                    name = param.attrib.get('value')

                                            for paramset5 in paramset4.findall('paramset'):
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

                                                    # TODO: mags should not be a list of lists, but a list of tuple.
                                                    # TODO: The fact that the information is fixed in size and type and
                                                    # TODO: yet a list is very confusing.
                                                    mags.append([band, mag, sys])

                                    if name in targets:
                                        targets[name].append(mags)
                                    else:
                                        targets[name] = mags

    return targets


def get_priority(observation_data: ElementTree) -> Optional[str]:
    """Get the priority of an observation from XML."""
    paramset = observation_data.find("paramset[@name='Observation'][@kind='dataObj']")
    try:
        param = paramset.find("param[@name='priority']")
        priority = param.attrib.get('value')
    except:
        priority = None
    return priority


# TODO: This function desperately needs refactoring, especially for magnitude extraction.
# TODO: I'm not going to hazard a return type.
def get_targets(observation_data: ElementTree):
    """
    Return a list of targets with a dictionary of properties:
        group (dictionary of name, tag, primary)
        name, ra, dec
        magnitudes (dictionary of band, value pairs)
        primary: whether the guide star will be used (and the group is primary)
        tag: {sidereal, asteroid, nonsidereal}
        type: {base, PWFS1, PWFS2, OIWFS, blindOffset, offAxis, tuning, other}
    """
    targets = []

    target_env = observation_data.find(".//paramset[@name='targetEnv']")

    if target_env is None:
        return None

    # Base -----------------------------------------------------------------------------------------

    asterism = target_env.find("paramset[@name='asterism']")
    target = asterism.find("paramset[@name='target']")
    base = {}
    magnitudes = {}

    for param in target.findall("param"):
        base[param.attrib.get('name')] = param.attrib.get('value')

    for paramset in target.findall("paramset[@name='magnitude']"):
        band, value = None, None
        for param in paramset.findall("param"):
            if param.attrib.get('name') == 'band':
                band = param.attrib.get('value')
            elif param.attrib.get('name') == 'value':
                value = param.attrib.get('value')
        if band is not None and value is not None:
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
            else:
                value = param.attrib.get('value')

            base[param.attrib.get('name')] = value

    if base['tag'] == 'nonsidereal':
        status = get_obs_status(observation_data)
        if status != 'OBSERVED':
            logging.warning(f"{get_obs_id(observation_data)} {base['name']} has no HORIZONS ID [{status}]")

    base['type'] = 'base'
    base['group'] = {'name': 'Base'}
    targets.append(base)

    # Guide Environment ----------------------------------------------------------------------------

    guide_env = target_env.find("paramset[@name='guideEnv']")
    primary_group = int(guide_env.find("param[@name='primary']").attrib.get('value'))

    group_number = -1
    for guideGroup in guide_env.findall("paramset[@name='guideGroup']"):
        group_number += 1
        group = {}
        for param in guideGroup.findall("param"):  # name, tag (auto, manual)
            group[param.attrib.get('name')] = param.attrib.get('value')

        if 'Auto' in group['tag']:  # "AutoActiveTag" or "AutoInitialTag" if AGS is disabled
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
                primary_guidestar = -1  # no active star for this guider

            guidestar_number = -1
            for sp_target in guider.findall("paramset[@name='spTarget']"):
                target = sp_target.find("paramset[@name='target']")
                guidestar_number += 1
                star = {'group': group, 'type': guidername}
                if guidestar_number == primary_guidestar:
                    star['primary'] = True
                else:
                    star['primary'] = False
                magnitudes = {}

                for param in target.findall("param"):
                    star[param.attrib.get('name')] = param.attrib.get('value')

                for paramset in target.findall("paramset[@name='magnitude']"):
                    band, value = None, None
                    for param in paramset.findall("param"):
                        if param.attrib.get('name') == 'band':
                            band = param.attrib.get('value')
                        elif param.attrib.get('name') == 'value':
                            value = param.attrib.get('value')
                    if band is not None and value is not None:
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
                        else:
                            value = param.attrib.get('value')

                        star[param.attrib.get('name')] = value

                if star['tag'] == 'nonsidereal':
                    status = get_obs_status(observation_data)
                    if status != 'OBSERVED':
                        print('%s %s has no HORIZONS ID [%s]', get_obs_id(observation_data), base['name'], status)

                star['magnitudes'] = magnitudes
                targets.append(star)

    # User targets ---------------------------------------------------------------------------------

    user_targets = target_env.find("paramset[@name='userTargets']")
    if user_targets is not None:
        for userTarget in user_targets.findall("paramset[@name='userTarget']"):
            star = {'group': {'name': 'User'}}
            magnitudes = {}

            for param in userTarget.findall("param"):
                star[param.attrib.get('name')] = param.attrib.get('value')

            sp_target = userTarget.find("paramset[@name='spTarget']")
            target = sp_target.find("paramset[@name='target']")

            for param in target.findall("param"):
                star[param.attrib.get('name')] = param.attrib.get('value')

            for paramset in target.findall("paramset[@name='magnitude']"):
                band, value = None, None
                for param in paramset.findall("param"):
                    if param.attrib.get('name') == 'band':
                        band = param.attrib.get('value')
                    elif param.attrib.get('name') == 'value':
                        value = param.attrib.get('value')
                if band is not None and value is not None:
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
                    else:
                        value = param.attrib.get('value')

                    star[param.attrib.get('name')] = value

            if star['tag'] == 'nonsidereal':
                status = get_obs_status(observation_data)
                if status != 'OBSERVED':
                    logging.warning(f'{get_obs_id(observation_data)} {base["name"]} has no HORIZONS ID [{status}]')
            targets.append(star)

    return targets


def get_windows(observation_data: ElementTree) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Get timing windows for observation from XML."""
    start = []
    duration = []
    repeat = []
    period = []

    for container in observation_data.findall('container'):
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

                # TODO ERROR: All of these values may be unassigned.
                start.append(window_start)
                duration.append(window_duration)
                repeat.append(window_repeat)
                period.append(window_period)

    return start, duration, repeat, period


def get_obs_too_status(observation_data: ElementTree, too_type: str) -> Optional[ToOType]:
    too = None

    if too_type == 'none':
        too = ToOType.NONE

    elif too_type == 'standard':
        too = ToOType.STANDARD

    else:
        paramset = observation_data.find('paramset')
        for param in paramset.findall('param'):
            if param.attrib.get('name') == 'tooOverrideRapid':
                override = param.attrib.get('value')
                if override == 'false':
                    too = ToOType.RAPID
                elif override == 'true':
                    too = ToOType.STANDARD
                else:
                    logging.error(f'Unknown TOO status.')

    return too


def check_status(program_data: ElementTree) -> Tuple[bool, bool]:
    """Check the status of the program in XML."""
    active, complete = False, False
    for param in program_data.find('paramset').findall('param'):
        if param.attrib.get('name') == 'fetched':
            active = param.attrib.get('value') == 'true'

        if param.attrib.get('name') == 'completed':
            complete = param.attrib.get('value') == 'true'

    return active, complete


# TODO: What is yp?
def get_ft_program_dates(notes: List[Tuple[Optional[str], Optional[str]]],
                         semester: str,
                         year: str,
                         yp: str) -> Tuple[Optional[Time], Optional[Time]]:
    progstart, progend = None, None

    def monthnum(month, months):
        month = month.lower()
        return [i for i, m in enumerate(months) if month in m].pop() + 1

    months_list = [x.lower() for x in calendar.month_name[1:]]
    #months_list = list(map(lambda x: x.lower(), calendar.month_name[1:]))
    for title, _ in notes:
        if title is not None and title:
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
                        for j in range(len(months_list)):
                            if f in months_list[j]:
                                months.append(f)

                im1 = monthnum(months[0], months_list)
                m1 = '{:02d}'.format(im1)
                im2 = monthnum(months[-1], months_list)
                m2 = '{:02d}'.format(im2)
                if semester == 'B' and im1 < 6:
                    progstart = Time(f'{yp}-{m1}-01 00:00:00', format='iso')
                    d2 = '{:02d}'.format(calendar.monthrange(int(yp), im2)[1])
                    progend = Time(f'{yp}-{m2}-{d2} 00:00:00', format='iso')
                else:
                    progstart = Time(f'{year}-{m1}-01 00:00:00', format='iso')
                    if im2 > im1:
                        d2 = '{:02d}'.format(calendar.monthrange(int(year), im2)[1])
                        progend = Time(f'{year}-{m2}-{d2} 00:00:00', format='iso')
                    else:
                        d2 = '{:02d}'.format(calendar.monthrange(int(yp), im2)[1])
                        progend = Time(f'{yp}-{m2}-{d2} 00:00:00', format='iso')
                break

    return progstart, progend


# TODO: This method needs to be simplified down. Type cannot be determined.
def get_observation_info(program_data: ElementTree):
    raw_info, schedgrps = [], []
    for container in program_data.findall('container'):
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
            # self.process_observation(container, schedgrp, selection=selection, obsclasses=obsclasses,
            #                        tas=tas, odbplan=odbplan)
            raw_info.append(container)
            schedgrps.append(schedgrp)

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
                    # return subcontainer, schedgrp
                    raw_info.append(subcontainer)

                    # TODO ERROR: schedgrp may not be assigned yet.
                    schedgrps.append(schedgrp)

    return zip(raw_info, schedgrps)


# GMOS FILE
def fpu_xml_translator(xmlfpu: List[str]) -> List[Dict[str, Union[str, Optional[float]]]]:
    """
    Convert the ODB XML FPU name to a human-readable mask name.
    Input: list of XML FPU names
    Output: list of dictionaries of FPU names and widths
    """
    fpu = []

    for f in xmlfpu:
        if f == 'FPU_NONE':
            fpu.append({'name': 'None', 'width': None})
        elif f == 'LONGSLIT_1':
            fpu.append({'name': '0.25arcsec', 'width': 0.25})
        elif f == 'LONGSLIT_2':
            fpu.append({'name': '0.5arcsec', 'width': 0.5})
        elif f == 'LONGSLIT_3':
            fpu.append({'name': '0.75arcsec', 'width': 0.75})
        elif f == 'LONGSLIT_4':
            fpu.append({'name': '1.0arcsec', 'width': 1.0})
        elif f == 'LONGSLIT_5':
            fpu.append({'name': '1.5arcsec', 'width': 1.5})
        elif f == 'LONGSLIT_6':
            fpu.append({'name': '2.0arcsec', 'width': 2.0})
        elif f == 'LONGSLIT_7':
            fpu.append({'name': '5.0arcsec', 'width': 5.0})
        elif f == 'IFU_1':
            fpu.append({'name': 'IFU-2', 'width': 5.0})
        elif f == 'IFU_2':
            fpu.append({'name': 'IFU-B', 'width': 3.5})
        elif f == 'IFU_3':
            fpu.append({'name': 'IFU-R', 'width': 3.5})
        elif f == 'NS_0':
            fpu.append({'name': 'NS0.25arcsec', 'width': 0.25})
        elif f == 'NS_1':
            fpu.append({'name': 'NS0.5arcsec', 'width': 0.5})
        elif f == 'NS_2':
            fpu.append({'name': 'NS0.75arcsec', 'width': 0.75})
        elif f == 'NS_3':
            fpu.append({'name': 'NS1.0arcsec', 'width': 1.0})
        elif f == 'NS_4':
            fpu.append({'name': 'NS1.5arcsec', 'width': 1.5})
        elif f == 'NS_5':
            fpu.append({'name': 'NS2.0arcsec', 'width': 2.0})
        elif f == 'CUSTOM_MASK':
            fpu.append({'name': 'CUSTOM_MASK', 'width': None})
        else:
            fpu.append({'name': 'UNKNOWN', 'width': None})

    return fpu


def custom_mask_width(custom_width_string: str) -> Optional[float]:
    """Parse a custom mask width from a string."""
    try:
        width = float(custom_width_string[13:].replace('_', '.'))
    except ValueError:
        width = None
    return width
