#!/usr/bin/env python3
"""
The main executable.
"""

from greedy_max import *
from astropy.table import Table
from astropy.visualization import time_support
time_support()

from resource_mock import Resource
#from schedule import SchedulingGroup, Observation
#import odb

tabdir = './data/'

if __name__ == '__main__':
    # Multi-site example

    # Target table
    ttab_gngs = Table.read(tabdir + 'targtab_metvisha_gngs_20201123.fits')

    # Observation table
    otab_gngs = Table.read(tabdir + 'obstab_gngs_20201123.fits')
    night_date = '2018-06-08'
    # Remove inst and disperser columns so that greedy-max doesn't add time for partner cals
    # otab_gngs.remove_columns(['inst', 'disperser'])

    sites = list(sites_from_column_names(ttab_gngs.colnames))

    # Resource API mock 
    resource = Resource('/resource_mock/data')
    resource.connect(['n','s'])
    
    fpu = {site: resource.night_info('fpu', site, night_date) for site in sites}
    fpur = {site: resource.night_info('fpur', site, night_date) for site in sites}
    grat = {site: resource.night_info('grat', site, night_date) for site in sites}
    instruments = {site: resource.night_info('instr', site, night_date) for site in sites}
    lgs = {site: resource.night_info('LGS', site, night_date) for site in sites}
    modes = {site: resource.night_info('mode', site, night_date) for site in sites}
    ifus = {'FPU':None,'FPUr':None}
    ifus['FPU'] = {site: resource.night_info('fpu-ifu', site, night_date) for site in sites}
    ifus['FPUr'] = {site: resource.night_info('fpur-ifu', site, night_date) for site in sites}
    fpu2b = resource.fpu_to_barcode
    
    print(fpu)    
    print(grat)
    print(fpur)

    # Load observation 
    obs_ids = [row['obs_id'] for row in otab_gngs]
    bands = {obs_id: Band(otab_gngs[obs_id == obs_id]['band']) for obs_id in obs_ids}
    if 'inst' in otab_gngs.colnames or 'disperser' in otab_gngs.colnames:
        instruments = {obs_id: otab_gngs['inst'][idx] for idx, obs_id in enumerate(obs_ids)} 
        dispersers = {obs_id: otab_gngs['disperser'][idx] for idx, obs_id in enumerate(obs_ids)}
    else:
        instruments = {obs_id: None for idx, obs_id in enumerate(obs_ids)} 
        dispersers = {obs_id: None for idx, obs_id in enumerate(obs_ids)}

    user_priorities = {obs_id: otab_gngs['user_prior'][idx] for idx, obs_id in enumerate(obs_ids)} 
    
    # Time management for observation
    # TODO: Must manage time inside observation in a more elegant and not-astropy-related way
    times = {obs_id: otab_gngs['obs_time'].quantity[idx] for idx, obs_id in enumerate(obs_ids)}
    tot_times = {obs_id: otab_gngs['tot_time'].quantity[idx] for idx, obs_id in enumerate(obs_ids)}
    completions = {obs_id: otab_gngs['obs_comp'].quantity[idx] for idx, obs_id in enumerate(obs_ids)}
   
    sites_by_obs = {obs_id: 'gs' if sum(ttab_gngs['weight_gs'][idx]) > 0 else 'gn' for idx, obs_id in enumerate(obs_ids)}

    
    observations = [Observation(obs_id, bands[obs_id], sites_by_obs[obs_id],
                                instruments[obs_id], dispersers[obs_id], user_priorities[obs_id],
                                times[obs_id], tot_times[obs_id], completions[obs_id])
                    for obs_id in obs_ids]
    
    weights_gs = {idx: ttab_gngs['weight_gs'][idx] for idx, obs_id in enumerate(obs_ids)}
    weights_gn = {idx: ttab_gngs['weight_gn'][idx] for idx, obs_id in enumerate(obs_ids)}
    weights = {Site.GS: weights_gs, Site.GN: weights_gn}

    # Time array
    timestab_gngs = Table.read(tabdir + 'timetab_gngs_20201123.fits')
  
    dt = time_slot_length(timestab_gngs['time'])
    nt = len(ttab_gngs['weight_' + Site(sites[0]).name.lower()][0])
    #TODO: Calculate nt by site
    time_slots = TimeSlots(dt, weights, nt, fpu, fpur, grat, 
                            instruments, lgs, modes, fpu2b, ifus)

    print(dt.to(u.min))

    # Convert to UT
    uttime_gngs = Time(timestab_gngs['time'], format='iso')
    print(uttime_gngs[0], uttime_gngs[0].jd)
    nt = len(uttime_gngs)

    # Make initial plan
    gm = GreedyMax(observations, time_slots, sites, verbose=False)
    gm.schedule()
    plan, obstab, targtab = gm.plan, gm.observations, gm.time_slots

    # Print current plan
    # i_day = 0
    sites = gm.sites
    for site in sites:
        # if s == 'gs':
        #     night_length = gs_night_length[i_day]
        # else:
        #     night_length = gn_night_length[i_day]
        print(Site(site).name.upper())
        obs_order, i_start, i_end = get_order(plan=plan[site])
        # print(obs_order, i_start, i_end)
        sum_score = 0.0
        sum_metric = 0.0
        time_used = 0.0 * u.hr
        for i in range(len(obs_order)):
            if obs_order[i] >= 0:
                print('{:18} {} {} {:8.4f}'.format(short_observation_id(otab_gngs['obs_id'][obs_order[i]]),
                                                   uttime_gngs[i_start[i]].strftime('%H:%M'),
                                                   uttime_gngs[i_end[i]].strftime('%H:%M'),
                                                   np.max(abs(targtab.weights[site][obs_order[i]][i_start[i]:i_end[i]]))
                                                   )
                      )
                # print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
                sum_score += np.sum(abs(targtab.weights[site][obs_order[i]][i_start[i]:i_end[i] + 1]))
                # sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i] + 1])
                time_used += (i_end[i] - i_start[i] + 1) * dt
        print('Sum score = {:7.2f}'.format(sum_score))
        print('Sum score/time step = {:7.2f}'.format(sum_score / nt))
        # print('Sum metric = {:7.2f}'.format(sum_metric))
        # print('Sum metric/time step = {:7.2f}'.format(sum_metric / nt))
        print('Time scheduled = {:5.2f}'.format(time_used))
        # print('Fraction of night scheduled = {:5.2f}'.format((time_used / night_length).value))
        print('')

