# Subroutines implementing the Greedy-Max scheduling algorithm
# Bryan Miller
# Based on Matt Bonneyman's port of Bryan's IDL code to Python

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.units.quantity import Quantity
import copy
from typing import List, Tuple


def _calibration_time(inst: str, disperser: str) -> Quantity:
    """
    Return the time needed for calibrations (esp. telluric standards).
    """
    tcal = 0.0 * u.min
    if 'GMOS' not in inst:
        if 'mirror' not in disperser.lower() and 'null' not in disperser.lower():
            tcal = 18.0 * u.min
    return tcal


def _acquisition_overhead_time(disperser: str) -> Quantity:
    """
    Get acquisition overhead time.
    """
    if 'mirror' in disperser.lower():
        return 0.2 * u.h
    else:
        return 0.3 * u.h


def _get_order(plan: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    """
    For observations scheduled in plan, get the observation indices in the order they appear, and return the
    array indices of schedule period boundaries observation.

    Example
    -------
    >>> plan = [2, 2, 2, 2, 1, 1, 1, 1, 5, 5, 4, 4, 4, 4]
    >>> ind_order, i_start, i_end = _get_order(plan)
    >>> print(ind_order)
    >>> print(i_start)
    >>> print(i_end)
    [2, 1, 5, 4]
    [0, 4, 8, 10]
    [3, 7, 9, 13]

    Parameters
    ----------
    plan : numpy integer array
        Observation indices throughout night.

    Returns
    -------
    order : list of ints
        order that indices appear in plan.

    i_start : list of ints
        indices of time block beginnings corresponding to plan.

    i_end : list of ints
        indices of time block endings corresponding to plan.
    """

    ind_order = [plan[0]]
    i_start = [0]
    i_end = []
    for i in range(1, len(plan)):
        prev = plan[i-1]
        if plan[i] != prev:
            ind_order.append(plan[i])
            i_end.append(i-1)
            i_start.append(i)
        if i == len(plan)-1:
            i_end.append(i)
    return ind_order, i_start, i_end


def sites_from_colnames(colnames, column='weight'):
    # Return a list of sites from table colnames
    # Sites
    sites = []
    for name in colnames:
        if column in name:
            lenn = len(name)
            sites.append(name[name.rfind('_')+1:lenn])

    return sites


def notscheduled(plan, sites=None):
    # return the total number of unscheduled slots in the plans
    if sites == None:
        sites = list(plan.copy().keys())
    n = 0
    for site in sites:
        n += len(np.where(plan[site] == -1)[0][:])
    return n


def shortid(obsid):
    # Return short form of obsid
    idsp = obsid.split('-')
    #         print(obsidsp)
    return idsp[0][1] + idsp[1][2:5] + '-' + idsp[2] + '-' + idsp[3] + '[' + idsp[4] + ']'


def intervals(indices):
    """
    Find the number and properties of contiguous intervals for an array of indices.

    Parameters
    ----------
    indices : np.ndarray of ints
        array of indices

    Returns
    -------
    indx : np.ndarray of ints

    """

    ni = len(indices)
    cvec = np.zeros(ni, dtype=int)
    nint = 1
    cvec[0] = nint
    for j in range(1, ni):
        if indices[j] != (indices[j - 1] + 1):
            nint = nint + 1
        cvec[j] = nint

    indx = np.digitize(cvec, bins=np.arange(ni) + 1)

    return indx


def deltat(time_strings):
    """
    Get dt

    Parameters
    ----------
    time_strings : array of strings
        iso format strings of utc or local times in timetable

    Returns
    -------
    dt : '~astropy.units.quantity.Quantity'
        differential tot_time length

    """
    return (Time(time_strings[1]) - Time(time_strings[0])).to('hour').round(7)


def greedy_max(plan, obs, targets, dt, tmin=30. * u.min, \
               verbose=False, verbose_final=False):
    """
    Add an observation to the current plan using the greedy-max scheduling algorithm.
    Return the current plan if no observations can be added.

    2018-08-01 addition: ToOs will be scheduled as early in the night as possible.

    Input
    ---------
    plan : list or np.array of integers
        Current plan.

    obs : '~astropy.table.Table'
        Observation information table.
        Required columns: 'prog_ref', 'obs_id', 'obs_time', 'obs_comp', 'tot_time'
        Optional columns: 'user_prior', 'inst', 'disperser'

    targets : '~astropy.table.Table'
        Target info table id and weights.
        Required columns: 'id', 'weight'
        'id' should be the same as 'obs_id' in table obs.

    dt : '~astropy.units.quantity.Quantity'
        Size of time grid spacing.

    tmin : '~astropy.units.quantity.Quantity'
        Length in time of a minimum visit

    verbose : 'boolean'
        Verbose output, for debugging

    verbose_final : 'boolean'
        Print information about the final results

    Returns
    ---------
    plan : list or np.array of integers
        new plan
    """

    verbose_add_to_plan = verbose_final  # print only the final results of the algorithm

    # min scheduling block length
    # nminuse = int(tmin.to(u.h) / dt.to(u.h) + 0.5)

    # -- Add an observation to the plan --
    while True:

        ii = np.where(plan == -1)[0][:]  # empty time slots in schedule
        if len(ii) != 0:
            nt = len(plan)  # number of time intervals in the plan
            n_obs = len(obs)  # number of observations
            indx = intervals(ii)  # intervals of empty time slots
            iint = ii[np.where(indx == 1)[0][:]]  # first interval of indx

            if verbose:
                print('ii:', ii)
                print('indx', indx)
                print('iint: ', iint)

            # -- Try to schedule an observation --
            gow = True
            while gow:
                # -- Select the observation with the maximum weight in time interval iint --
                maxweight = 0.  # maximum weight in time interval
                iimax = -1  # index of target with max. weight
                nminuse = int(np.ceil(tmin.to(u.h) / dt.to(u.h)))  # reset min block length,
                for i in np.arange(n_obs):
                    ipos = np.where(targets['weight'][i][iint] > 0.)[0][:]  # indices of weights>0 in first interval
                    if len(ipos) == 0:
                        continue  # skip to next observation if no non-zero weights in current window
                    else:
                        iwin = iint[ipos]  # indices with pos. weights within first empty window
                        if verbose:
                            print('iwin', iwin)
                        if len(iwin) >= 2:  # if window >= 2 (why 2?)
                            if verbose:
                                print('i, weights:', i, targets['weight'][i][iint])
                            # index of max weight
                            i_wmax = iwin[np.argmax(targets['weight'][i][iwin])]
                            wmax = targets['weight'][i][i_wmax]  # maximum weight
                        else:  # if window size of 1
                            wmax = targets['weight'][i][iwin]  # maximum weight

                        if wmax > maxweight:
                            maxweight = wmax
                            iimax = i
                            iwinmax = iwin

                        if verbose:
                            print('maxweight', maxweight)
                            print('max obs: ', targets['id'][iimax])
                            print('iimax', iimax)

                # -- Determine observation window and length --
                if iimax == -1:
                    gow = False
                else:
                    # Boundaries of available window
                    wstart = iwinmax[0]  # window start
                    wend = iwinmax[-1]  # window end
                    nobswin = wend - wstart + 1

                    # Calibration time
                    ntcal = 0
                    if 'inst' in obs.colnames and 'disperser' in obs.colnames:
                        if verbose:
                            print('Inst: ', obs['inst'][iimax])
                            print('Disperser: ', obs['disperser'][iimax])
                        tcal = _calibration_time(inst=obs['inst'][iimax], disperser=obs['disperser'][iimax])
                        ntcal = int(np.ceil(tcal.to(u.h) / dt.to(u.h)))
                        if verbose:
                            print('ntcal = ', ntcal)

                    # Remaining time (including calibration)
                    ttime = (obs['tot_time'].quantity[iimax] -
                              obs['obs_time'].quantity[iimax])
                              # obs['obs_time'].quantity[iimax] + dt.to(u.h)).round(1))
                    # number of slots needed in time grid
                    nttime = int(np.ceil(ttime.to(u.h) / dt.to(u.h)) + ntcal)

                    # Don't leave little pieces of observations remainimg
                    # Also, short observations are done entirely
                    if nttime - nminuse <= nminuse:
                        nminuse = nttime

                    if verbose:
                        print('ID of chosen ob.', targets['id'][iimax])
                        print('weights of chosen ob.', targets['weight'][iimax])
                        print('Current plan', plan)
                        print('wstart', wstart)
                        print('wend', wend)
                        print('dt', dt)
                        print('tot_time', obs['tot_time'].quantity[iimax])
                        print('obs_time', obs['obs_time'].quantity[iimax])
                        print('ttime', ttime)
                        print('nttime', nttime)
                        print('nobswin', nobswin)
                        print('nminuse', nminuse)

                    # Decide whether or not to add to schedule
                    if np.logical_or(nttime <= nobswin, nobswin >= nminuse):  # Schedule observation
                        gow = False
                    else:  # Do not schedule observation
                        targets['weight'][iimax][iint] = 0.
                        if verbose:
                            print('Block too short to schedule...')

            # -- Place observation in schedule
            if iimax == -1:
                plan[iint] = -2
            else:
                # -- Place observation within available window --
                if np.logical_and(nttime <= nobswin, nttime != 0):
                    jj = np.where(plan == iimax)[0][:]  # check if already scheduled
                    if len(jj) == 0:
                        # Schedule interrupt ToO at beginning of window
                        if 'user_prior' in obs.colnames and 'Interrupt' in obs['user_prior'][iimax]:
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:
                            # Determine schedule placement for maximum integrated weight
                            maxf = 0.0
                            if nttime > 1:
                                # NOTE: integrates over one extra time slot...
                                # ie. if nttime = 14, then the program will choose 15
                                # x values to do trapz integration (therefore integrating
                                # 14 time slots).
                                if verbose:
                                    print('\nIntegrating max obs. over window...')
                                    print('wstart', wstart)
                                    print('wend', wend)
                                    print('nttime', nttime)
                                    print('j values', np.arange(wstart, wend - nttime + 2))
                                for j in range(wstart, wend - nttime + 2):
                                    f = sum(targets['weight'][iimax][j:j + nttime])
                                    if verbose:
                                        print('j range', j, j + nttime - 1)
                                        print('obs weight', targets['weight'][iimax][j:j + nttime])
                                        print('integral', f)
                                    if f > maxf:
                                        maxf = f
                                        jstart = j
                                        jend = jstart + nttime - 1
                            else:
                                jstart = np.argmax(targets['weight'][iimax][iwinmax])
                                maxf = np.amax(targets['weight'][iimax][jstart])
                                jend = jstart + nttime - 1

                            if verbose:
                                print('max integral of weight func (maxf)', maxf)
                                print('index jstart', jstart)
                                print('index jend', jend)

                            # shift to start or end of night if within minimum block time from boundary
                            if jstart < nminuse:
                                if plan[0] == -1 and targets['weight'][iimax][0] > 0.:
                                    jstart = 0
                                    jend = jstart + nttime - 1
                            elif (nt - jend) < nminuse:
                                if plan[-1] == -1 and targets['weight'][iimax][-1] > 0.:
                                    jend = nt - 1
                                    jstart = jend - nttime + 1

                            # Shift to window boundary if within minimum block time of edge.
                            # If near both boundaries, choose boundary with higher weight.
                            wtstart = targets['weight'][iimax][wstart]  # weight at start
                            wtend = targets['weight'][iimax][wend]  # weight at end
                            dstart = jstart - wstart - 1  # difference between start of window and block
                            dend = wend - jend + 1  # difference between end of window and block
                            if dstart < nminuse and dend < nminuse:
                                if wtstart > wtend and wtstart > 0.:
                                    jstart = wstart
                                    jend = wstart + nttime - 1
                                elif wtend > 0.:
                                    jstart = wend - nttime + 1
                                    jend = wend
                            elif dstart < nminuse and wtstart > 0.:
                                jstart = wstart
                                jend = wstart + nttime - 1
                            elif dend < nminuse and wtstart > 0.:
                                jstart = wend - nttime + 1
                                jend = wend

                    # If observation is already in plan, shift to side of window closest to existing obs.
                    # Future - try to shift the plan to join the pieces and save an acq
                    else:
                        if jj[0] < wstart:  # Existing obs in plan before window. Schedule at beginning of window.
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:  # Existing obs in plan after window. Schedule at end of window.
                            jstart = wend - nttime + 1
                            jend = wend

                else:  # if window smaller than observation length
                    jstart = wstart
                    jend = wend

                if verbose:
                    print('Chosen index jstart', jstart)
                    print('Chosen index jend', jend)
                    print('Current obs time: ', obs['obs_time'].quantity[iimax])
                    print('Current tot time: ', obs['tot_time'].quantity[iimax])

                plan[jstart:jend + 1] = iimax  # Add observation to plan
                #                 ntmin = np.minimum(nttime - ntcal, nobswin)  # number of spots in time grid used(excluding calibration)
                ntmin = np.minimum(nttime - ntcal,
                                   jend - jstart + 1)  # number of spots in time grid used(excluding calibration)
                #                 print('ntmin = ', ntmin, ', jend - jstart + 1 = ', jend - jstart + 1)

                obs['obs_time'].quantity[iimax] = obs['obs_time'].quantity[iimax] + dt.to(u.h) * ntmin  # update time
                obs['obs_comp'][iimax] = obs['obs_comp'][iimax] + dt.to(u.h) * ntmin / \
                                         obs['tot_time'].quantity[iimax]  # update completion fraction

                # Adjust weights of scheduled observation
                if obs['obs_comp'][iimax] >= 1.:  # if completed set all to negative values
                    targets['weight'][iimax] = -1. * targets['weight'][iimax]
                else:  # if observation not fully completed, set only scheduled portion negative. Increase remaining.
                    targets['weight'][iimax][jstart:jend + 1] = -1. * targets['weight'][iimax][jstart:jend + 1]
                    wpositive = np.where(targets['weight'][iimax] > 0.0)[0][:]
                    targets['weight'][iimax][wpositive] = targets['weight'][iimax][wpositive] * 1.5

                # Add to total time if observation not complete
                if 'disperser' in obs.colnames:
                    if obs['obs_time'].quantity[iimax] < obs['tot_time'].quantity[iimax]:
                        acqover = _acquisition_overhead_time(obs['disperser'][iimax])
                        obs['tot_time'].quantity[iimax] = obs['tot_time'].quantity[iimax] + acqover

                # increase weights of observations in program
                # ii_obs = np.where(obs.obs_id == obs.prog_ref[iimax])[0][:]  # indices of obs. in same program
                if verbose:
                    print('Current plan: ', plan)
                    print('New obs. weights: ', targets['weight'][iimax])
                    print('nttime - ntcal , nobswin: ', nttime - ntcal, nobswin)
                    print('ntmin: ', ntmin)
                    print('Tot time: ', obs['tot_time'].quantity[iimax])
                    print('New obs time: ', obs['obs_time'].quantity[iimax])
                    print('New comp time: ', obs['obs_comp'][iimax])

                if verbose_add_to_plan:
                    print('\tScheduled: ', iimax, targets['id'][iimax], 'from jstart =', jstart, 'to jend =', jend)
                    print(targets['weight'][iimax])

                break  # successfully added an observation to the plan
        else:
            break  # No available spots in plan

    return plan



def optimize(plan, targets, jj=None, verbose=False, summary=True):
    """
    Attempt to rearrange plan and maximize the sum of the target weighting functions.

    Parameters
    ----------
    plan : np.ndarray of ints
        array of observation indices in plan

    targets : '~astropy.table.Table'
        Target information table

    jj : np.ndarray of ints
        indices of section of 'plan' to be optimized.  Other parts of the plan
        will remain unchanged.

    Returns
    -------
    plan : np.ndarray of ints
        array of observation indices in plan
    """

    # select whole plan to optimize
    if jj is None:
        jj = np.arange(len(plan))

    result = copy.deepcopy(plan)

    newplan = np.full(len(plan), -2)  # empty plan

    i_obs = np.unique(plan[jj])  # obs in plan[jj]

    nid = np.zeros(len(i_obs), dtype=int)  # number of time slots per obs
    plan_weight = 0.

    if verbose:
        print('Full plan: ', plan)
        print('Plan section to optimize (plan[jj]): ', plan[jj])
        print('jj: ', jj)
        print('i_obs: ', i_obs)

    # -- Compute total weight of plan[jj] --
    for i in range(0, len(i_obs)):
        ii = jj[np.where(plan[jj] == i_obs[i])[0][:]]
        nid[i] = int(len(ii))
        if i_obs[i] >= 0:
            plan_weight += np.sum(abs(targets['weight'][i_obs[i]][ii]))

    # Attempt to re-arrange observations and achieve higher total weight
    # Need a check that weights >0 at each new timestep
    nt = len(jj)
    i = jj[0]
    while i < nt:
        if plan[i] >= 0:
            if verbose:
                print('i, i_obs, nid: ', i, i_obs, nid)
            imax = -1
            wmax = 0.
            idmax = 0.
            for j in range(0, len(i_obs)):
                if i_obs[j] >= 0:
                    temp_wmax = abs(targets['weight'][i_obs[j]][i])
                    if temp_wmax > wmax:
                        wmax = temp_wmax
                        imax = j
                        idmax = i_obs[j]
            if verbose:
                print('wmax, imax, idmax: ', wmax, imax, idmax)
            if wmax > 0.:
                if i + nid[imax] <= nt:
                    newplan[i:(i + nid[imax])] = idmax
                    i = i + nid[imax]
                else:
                    newplan[i:nt] = idmax
                    i = nt
                if verbose:
                    print('nid[imax]: ', nid[imax])
                    print('delete j from i_obs: ', imax, i_obs)
                    print('newplan: ', newplan)
                i_obs = np.delete(i_obs, imax, None)
                nid = np.delete(nid, imax, None)
            else:
                i = i + 1
        else:
            i = i + 1

    # return original plan if no changes were made
    #     if newplan[jj].all() == plan[jj].all():
    #         return plan

    # Get total weight of new plan
    i_newobs = np.unique(newplan[jj])  # obs in newplan[jj]
    newplan_weight = 0.
    new_nid = np.zeros(len(i_newobs))
    for i in range(len(i_newobs)):
        ii = jj[np.where(newplan[jj] == i_newobs[i])[0][:]]
        new_nid[i] = int(len(ii))
        if i_newobs[i] >= 0:
            newplan_weight += np.sum(abs(targets['weight'][i_newobs[i]][ii]))

            # -- Compute total weight of plan[jj] --

    if verbose or summary:
        print('Original plan[jj]: ', plan[jj])
        print('Original Weight: ', plan_weight)
        print('New plan[jj]: ', newplan)
        print('New Weight: ', newplan_weight)

    if newplan_weight >= plan_weight:
        result[jj] = newplan[jj]

    return result


def greedy_max_multi(plan, obs, targets, dt, tmin=30. * u.min, sites=None, \
                     verbose=False, verbose_final=False):
    """
    Add an observation to the current plan(s) using the greedy-max scheduling algorithm.
    Return the current plan if no observations can be added.

    2018-08-01 addition: ToOs will be scheduled as early in the night as possible.
    2020-10-10 generalize to multiple sites

    Input
    ---------
    plan : dictionary of lists or np.arrays of integers.
        Keys must match the suffixes on the weight columns in the targets table.
        Current plan.

    obs : '~astropy.table.Table'
        Observation information table.
        Required columns: 'prog_ref', 'obs_id', 'obs_time', 'obs_comp', 'tot_time'
        Optional columns: 'user_prior', 'inst', 'disperser'

    targets : '~astropy.table.Table'
        Target info table id and weights.
        Required columns: 'id', 'weight_<site>'
        'id' should be the same as 'obs_id' in table obs.

    dt : '~astropy.units.quantity.Quantity'
        Size of time grid spacing.

    tmin : '~astropy.units.quantity.Quantity'
        Length in time of a minimum visit

    sites : 'list'
        List of the sites to schedule. Taken from the targets table if not given.

    verbose : 'boolean'
        Verbose output, for debugging

    verbose_final : 'boolean'
        Print information about the final results

    Returns
    ---------
    plan : list or np.array of integers
        new plan
    """

    verbose_add_to_plan = verbose_final  # print only the final results of the algorithm

    # min scheduling block length
    # nminuse = int(tmin.to(u.h) / dt.to(u.h) + 0.5)

    # Sites
    if sites is None:
        sites = sites_from_colnames(targets.colnames)
    # should check that these match the keys in the plan dict

    nt = len(plan[sites[0]])  # number of time intervals in the plan
    n_obs = len(obs['obs_id'])  # number of observations
    #     print(sites, nt, n_obs)

    # -- Add an observation to the plan --
    i_iter = 0
    while True:
        i_iter += 1
        if verbose:
            print('greedy iteration:', i_iter)
        n_unsched = notscheduled(plan, sites=sites)
        if n_unsched != 0:

            # -- Try to schedule an observation --
            i_gow = 0
            gow = True
            while gow:
                i_gow += 1
                if verbose:
                    print('i_gow', i_gow)

                # -- Select the observation with the maximum weight in time interval iint --
                maxweight = 0.  # maximum weight in time interval
                iimax = -1  # index of target with max. weight
                smax = ''  # site of max weight
                iints = {}  # save intervals for each site for later use
                nminuse = int(np.ceil(tmin.to(u.h) / dt.to(u.h))) # reset min block length,

                # Loop over sites
                for s in sites:
                    ii = np.where(plan[s] == -1)[0][:]  # empty time slots in schedule
                    if len(ii) != 0:
                        indx = intervals(ii)  # intervals of empty time slots
                        iint = ii[np.where(indx == 1)[0][:]]  # first interval of indx
                        iints[s] = iint
                        if verbose:
                            print('ii:', ii)
                            print('indx', indx)
                            print('iint: ', iint)
                            print('site: ', s)

                        # Loop over observations
                        for i in range(n_obs):
                            # maximum weight in interval
                            wmax = np.max(targets['weight_' + s][i][iint])

                            # indices of weights>0 in interval
                            ipos = np.where(targets['weight_' + s][i][iint] > 0.)[0][:]

                            # Test
                            if len(ipos) > 0 and wmax > maxweight:
                                iwinmax = iint[ipos]  # indices with pos. weights within first empty window
                                maxweight = wmax
                                iimax = i
                                smax = s

                                if verbose:
                                    print('maxweight', maxweight)
                                    print('max obs: ', targets['id'][iimax])
                                    print('iimax', iimax)
                                    print('smax', smax)

                # -- Determine observation window and length --
                if iimax == -1:
                    gow = False
                else:
                    # Boundaries of available window
                    wstart = iwinmax[0]  # window start
                    wend = iwinmax[-1]  # window end
                    nobswin = wend - wstart + 1

                    # Calibration time
                    ntcal = 0
                    if 'inst' in obs.colnames and 'disperser' in obs.colnames:
                        if verbose:
                            print('Inst: ', obs['inst'][iimax])
                            print('Disperser: ', obs['disperser'][iimax])
                        tcal = _calibration_time(inst=obs['inst'][iimax], disperser=obs['disperser'][iimax])
                        ntcal = int(np.ceil(tcal.to(u.h) / dt.to(u.h)))
                        if verbose:
                            print('ntcal = ', ntcal)

                    # Remaining time (including calibration)
                    ttime = (obs['tot_time'].quantity[iimax] -
                                obs['obs_time'].quantity[iimax])
                              # obs['obs_time'].quantity[iimax] + dt.to(u.h)).round(1))
                    # number of slots needed in time grid, rounding up
                    nttime = int(np.ceil((ttime.to(u.h) / dt.to(u.h)))) + ntcal

                    # Don't leave little pieces of observations remaining
                    # Also, short observations are done entirely
                    if nttime - nminuse <= nminuse:
                        nminuse = nttime

                    if verbose:
                        print('ID of chosen ob.', targets['id'][iimax])
                        print('weights of chosen ob.', targets['weight_' + smax][iimax])
                        print('Current plan', plan[smax])
                        print('wstart', wstart)
                        print('wend', wend)
                        print('dt', dt)
                        print('tot_time', obs['tot_time'].quantity[iimax])
                        print('obs_time', obs['obs_time'].quantity[iimax])
                        print('ttime', ttime)
                        print('nttime', nttime)
                        print('nobswin', nobswin)
                        print('nminuse', nminuse)

                    # Decide whether or not to add to schedule
                    if np.logical_or(nttime <= nobswin, nobswin >= nminuse):  # Schedule observation
                        gow = False
                    else:  # Do not schedule observation
                        targets['weight_' + smax][iimax][iints[smax]] = 0.
                        if verbose:
                            print('Block too short to schedule...')

            # -- Place observation in schedule
            if iimax == -1:
                for site in iints.keys():
                    if len(iints[site]) > 0:
                        plan[site][iints[site]] = -2
            else:
                # -- Place observation within available window --
                if np.logical_and(nttime <= nobswin, nttime != 0):
                    jj = np.where(plan[smax] == iimax)[0][:]  # check if already scheduled
                    if len(jj) == 0:
                        if 'user_prior' in obs.colnames and 'Interrupt' in obs['user_prior'][
                            iimax]:  # Schedule interrupt ToO at beginning of window
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:
                            # Determine schedule placement for maximum integrated weight
                            maxf = 0.0
                            if nttime > 1:
                                # NOTE: integrates over one extra time slot...
                                # ie. if nttime = 14, then the program will choose 15
                                # x values to do trapz integration (therefore integrating
                                # 14 time slots).
                                if verbose:
                                    print('\nIntegrating max obs. over window...')
                                    print('wstart', wstart)
                                    print('wend', wend)
                                    print('nttime', nttime)
                                    print('j values', np.arange(wstart, wend - nttime + 2))
                                for j in range(wstart, wend - nttime + 2):
                                    f = sum(targets['weight_' + smax][iimax][j:j + nttime])
                                    if verbose:
                                        print('j range', j, j + nttime - 1)
                                        print('obs weight', targets['weight_' + smax][iimax][j:j + nttime])
                                        print('integral', f)
                                    if f > maxf:
                                        maxf = f
                                        jstart = j
                                        jend = jstart + nttime - 1
                            else:
                                jstart = np.argmax(targets['weight_' + smax][iimax][iwinmax])
                                maxf = np.amax(targets['weight_' + smax][iimax][jstart])
                                jend = jstart + nttime - 1

                            if verbose:
                                print('max integral of weight func (maxf)', maxf)
                                print('index jstart', jstart)
                                print('index jend', jend)

                            # shift to start or end of night if within minimum block time from boundary
                            if jstart < nminuse:
                                if plan[smax][0] == -1 and targets['weight_' + smax][iimax][0] > 0.:
                                    jstart = 0
                                    jend = jstart + nttime - 1
                            elif (nt - jend) < nminuse:
                                if plan[smax][-1] == -1 and targets['weight_' + smax][iimax][-1] > 0.:
                                    jend = nt - 1
                                    jstart = jend - nttime + 1

                            # Shift to window boundary if within minimum block time of edge.
                            # If near both boundaries, choose boundary with higher weight.
                            wtstart = targets['weight_' + smax][iimax][wstart]  # weight at start
                            wtend = targets['weight_' + smax][iimax][wend]  # weight at end
                            dstart = jstart - wstart - 1  # difference between start of window and block
                            dend = wend - jend + 1  # difference between end of window and block
                            if dstart < nminuse and dend < nminuse:
                                if wtstart > wtend and wtstart > 0.:
                                    jstart = wstart
                                    jend = wstart + nttime - 1
                                elif wtend > 0.:
                                    jstart = wend - nttime + 1
                                    jend = wend
                            elif dstart < nminuse and wtstart > 0.:
                                jstart = wstart
                                jend = wstart + nttime - 1
                            elif dend < nminuse and wtstart > 0.:
                                jstart = wend - nttime + 1
                                jend = wend

                    # If observation is already in plan, shift to side of window closest to existing obs.
                    # Future - try to shift the plan to join the pieces and save an acq
                    else:
                        if jj[0] < wstart:  # Existing obs in plan before window. Schedule at beginning of window.
                            jstart = wstart
                            jend = wstart + nttime - 1
                        else:  # Existing obs in plan after window. Schedule at end of window.
                            jstart = wend - nttime + 1
                            jend = wend

                else:  # if window smaller than observation length
                    jstart = wstart
                    jend = wend

                if verbose:
                    print('Chosen index jstart', jstart)
                    print('Chosen index jend', jend)
                    print('Current obs time: ', obs['obs_time'].quantity[iimax])
                    print('Current tot time: ', obs['tot_time'].quantity[iimax])

                plan[smax][jstart:jend + 1] = iimax  # Add observation to plan
                #                 ntmin = np.minimum(nttime - ntcal, nobswin)  # number of spots in time grid used(excluding calibration)
                ntmin = np.minimum(nttime - ntcal,
                                   jend - jstart + 1)  # number of spots in time grid used(excluding calibration)
                #                 print('ntmin = ', ntmin, ', jend - jstart + 1 = ', jend - jstart + 1)

                obs['obs_time'].quantity[iimax] = obs['obs_time'].quantity[iimax] + dt.to(u.h) * ntmin  # update time
                obs['obs_comp'].quantity[iimax] = obs['obs_comp'].quantity[iimax] + dt.to(u.h) * ntmin / \
                                                  obs['tot_time'].quantity[iimax]  # update completion fraction

                # Adjust weights of scheduled observation
                if obs['obs_comp'][iimax] >= 1.:  # if completed set all to negative values
                    targets['weight_' + smax][iimax] = -1. * targets['weight_' + smax][iimax]
                else:  # if observation not fully completed, set only scheduled portion negative. Increase remaining.
                    targets['weight_' + smax][iimax][jstart:jend + 1] = -1. * targets['weight_' + smax][iimax][jstart:jend + 1]
                    wpositive = np.where(targets['weight_' + smax][iimax] > 0.0)[0][:]
                    targets['weight_' + smax][iimax][wpositive] = targets['weight_' + smax][iimax][wpositive] * 1.5
                    # TODO: Update visfrac and weight, do outside this routine?

                # Set weights to zero for other sites so it won't be scheduled again
                for s in sites:
                    if s != smax:
                        targets['weight_' + s][iimax][:] = 0.0

                # Add to total time if observation not complete
                if 'disperser' in obs.colnames:
                    if obs['obs_time'][iimax] < obs['tot_time'][iimax]:
                        acqover = _acquisition_overhead_time(obs['disperser'][iimax])
                        obs['tot_time'].quantity[iimax] = obs['tot_time'].quantity[iimax] + acqover

                # increase weights of observations in program
                # ii_obs = np.where(obs.obs_id == obs.prog_ref[iimax])[0][:]  # indices of obs. in same program
                if verbose:
                    print('Current plan: ', plan[smax])
                    print('New obs. weights: ', targets['weight_' + smax][iimax])
                    print('nttime - ntcal , nobswin: ', nttime - ntcal, nobswin)
                    print('ntmin: ', ntmin)
                    print('Tot time: ', obs['tot_time'].quantity[iimax])
                    print('New obs time: ', obs['obs_time'].quantity[iimax])
                    print('New comp time: ', obs['obs_comp'][iimax])

                if verbose_add_to_plan:
                    print('\tScheduled: ', iimax, targets['id'][iimax], 'from jstart =', jstart, 'to jend =', jend,
                          'at site', smax)
                    print(targets['weight_' + smax][iimax])

                break  # successfully added an observation to the plan
        else:
            break  # No available spots in plan

    return plan, obs, targets


def schedule_night(obstab, targets, dt, tmin=30. * u.min, verbose=False):
    """
    Schedule a single night using the greedy-max algorithm

    obstab : '~astropy.table.Table'
        Observation information table.
        Required columns: 'prog_ref', 'obs_id', 'obs_time', 'obs_comp', 'tot_time'
        Optional columns: 'user_prior', 'inst', 'disperser'

    targets : '~astropy.table.Table'
        Target info table id and weights.
        Required columns: 'id', 'weight'
        'id' should be the same as 'obs_id' in table obs.

    dt : '~astropy.units.quantity.Quantity'
        Size of time grid spacing.
    """

    # Make a copy of the original target tab so that it won't be modified
    targtab = copy.deepcopy(targets)

    # number of time steps
    nt = len(targtab['weight'][0])

    # ====== Initialize plan parameters ======
    plan = np.full(nt, -1)  # Empty plan

    # ====== Complete current plan iteration ======
    # Fill nightly plan one observation at a time.
    n_iter = 0
    ii = np.where(plan == -1)[0][:]  # unscheduled time slots
    while len(ii) != 0:
        plan_temp = copy.deepcopy(plan)  # make copy of current plan before completing next iteration
        plan = greedy_max(plan=plan_temp, obs=obstab, targets=targtab, dt=dt,
                          tmin=tmin, verbose=False, verbose_final=False)
        #         print(plan_temp)

        ii = np.where(plan == -1)[0][:]
        n_iter += 1

        if verbose:
            # Print current plan
            print('Iteration {:4d}'.format(n_iter))
            print('{:15} {:4} {:4} {:>8}'.format('Obsid', 'i_start', 'i_end', 'Max W'))
            obs_order, i_start, i_end = _get_order(plan=plan)
            sum_score = 0.0
            #             sum_metric = 0.0
            time_used = 0
            for i in range(len(obs_order)):
                if obs_order[i] >= 0:
                    print('{:18} {:>4d}  {:>4d} {:8.4f}'.format(shortid(obstab['obs_id'][obs_order[i]]),
                                                                i_start[i], i_end[i],
                                                                np.max(abs(targtab['weight'][obs_order[i]][
                                                                           i_start[i]:i_end[i]]))))
                    #                 print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
                    sum_score += np.sum(abs(targtab['weight'][obs_order[i]][i_start[i]:i_end[i] + 1]))
                    #                 sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i]+1])
                    time_used += (i_end[i] - i_start[i] + 1)
    if verbose:
        print('Sum score = {:7.2f}'.format(sum_score))
        print('Sum score/time step = {:7.2f}'.format(sum_score / nt))
        #     print('Sum metric = {:7.2f}'.format(sum_metric))
        #     print('Sum metric/time step = {:7.2f}'.format(sum_metric / nt)
        print('Time scheduled = {:5.2f}'.format(time_used * dt.to(u.h)))
        print('Fraction of night scheduled = {:5.2f}'.format(time_used / nt))

    return plan, targtab


def schedule_night_multi(obs, targets, dt, tmin=30. * u.min, sites=None, verbose=False):
    """
    Schedule a single night for multiple sites using the greedy-max algorithm

    obstab : '~astropy.table.Table'
        Observation information table.
        Required columns: 'prog_ref', 'obs_id', 'obs_time', 'obs_comp', 'tot_time'
        Optional columns: 'user_prior', 'inst', 'disperser'

    targets : '~astropy.table.Table'
        Target info table id and weights.
        Required columns: 'id', 'weight'
        'id' should be the same as 'obs_id' in table obs.

    dt : '~astropy.units.quantity.Quantity'
        Size of time grid spacing.

    tmin : '~astropy.units.quantity.Quantity'
        The minimum visit length in minutes

    sites : 'list'
        List of the sites to schedule. Taken from the targets table if not given.

    """

    # Make a copy of the original tables so that they won't be modified
    #     targtab = copy.deepcopy(targets)
    targtab = targets.copy()
    obstab = obs.copy()

    # Sites
    if sites is None:
        sites = sites_from_colnames(targtab.colnames)
    #     print(sites)

    # number of time steps
    nt = len(targtab['weight_' + sites[0]][0])
    if verbose:
        print(sites)
        print('Time steps: ', nt)

    # ====== Initialize plan parameters ======
    plan = {}
    for site_name in sites:
        plan[site_name] = np.full(nt, -1)  # Empty plan
    #     print(plan)

    n_iter = 0
    # unscheduled time slots
    n_unsched = notscheduled(plan, sites=sites)
    #     print(n_unsched)
    while n_unsched != 0:
        plan_temp = copy.deepcopy(plan)  # make copy of current plan before completing next iteration
        plan, obstab, targtab = greedy_max_multi(plan=plan_temp, obs=obstab, targets=targtab, dt=dt,
                                sites=sites, tmin=tmin, verbose=verbose, verbose_final=False)
        #         print(plan_temp)

        n_unsched = notscheduled(plan, sites=sites)
        #         print('Unscheduled time steps: ',n_unsched)

        n_iter += 1
        if verbose:
            # Fill nightly plan one observation at a time.
            sum_score = 0.0
            sum_metric = 0.0
            time_used = 0

            # Print current plan
            print('Iteration {:4d}'.format(n_iter))
            for site_name in sites:
                print(site_name.upper())
                print('{:18} {:>9} {:>8} {:>8} {:>8}'.format('Obsid', 'obs_order', 'i_start', 'i_end', 'Max W'))
                obs_order, i_start, i_end = _get_order(plan=plan[site_name])
                for i in range(len(obs_order)):
                    if obs_order[i] >= 0:
                        print('{:18} {:>9d} {:>8d} {:>8d} {:8.4f}'.format(shortid(obstab['obs_id'][obs_order[i]]),
                                                                    obs_order[i], i_start[i], i_end[i],
                                                                    np.max(abs(
                                                                        targtab['weight_' + site_name][obs_order[i]][
                                                                        i_start[i]:i_end[i] + 1]))))
                        #                 print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
                        sum_score += np.sum(abs(targtab['weight_' + site_name][obs_order[i]][i_start[i]:i_end[i] + 1]))
                        if 'metric' in targtab.colnames:
                            sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i] + 1])
                        time_used += (i_end[i] - i_start[i] + 1)
    if verbose:
        print('Sum score = {:7.2f}'.format(sum_score))
        print('Sum score/time step = {:7.2f}'.format(sum_score / (2 * nt)))
        if 'metric' in targtab.colnames:
            print('Sum metric = {:7.2f}'.format(sum_metric))
            print('Sum metric/time step = {:7.2f}'.format(sum_metric / (2 * nt)))
        print('Time scheduled = {:5.2f}'.format(time_used * dt.to(u.hr)))
        # print('Fraction of night scheduled = {:5.2f}'.format(
        #     (time_used * dt).value / (gs_night_length[0] + gn_night_length[0])))

    return plan, obstab, targtab



