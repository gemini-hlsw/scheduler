#!/usr/bin/env python3
"""
The main executable.
"""

from greedy_max import *
from resource_mock import Resource

import pickle
from astropy.table import Table
from astropy.visualization import time_support
time_support()


#from schedule import SchedulingGroup, Observation
#import odb

tabdir = './data/'

if __name__ == '__main__':
    # Multi-site example

    # Target table
    #ttab_gngs = Table.read(tabdir + 'targtab_metvisha_gngs_20201123.fits')
    #print(ttab_gngs.colnames)
    # Observation table
    #otab_gngs = Table.read(tabdir + 'obstab_gngs_20201123.fits')
    # Time array
    #timestab_gngs = Table.read(tabdir + 'timetab_gngs_20201123.fits')
    timestab_gngs = Table.read(tabdir + 'timetab_gs_2019-02-02.fits')
    #Group table
    f0 = open(f'{tabdir}/grptab_plan','rb')
    grptab_gngs = pickle.load(f0)
    f1 = open(f'{tabdir}/obstab_plan','rb')
    otab_gngs = pickle.load(f1)
    f2 = open(f'{tabdir}/tartab_plan','rb')
    ttab_gngs = pickle.load(f2)
    f0.close()
    f1.close()
    f2.close()

    #print(grptab_gngs.colnames)
    #print(otab_gngs.colnames)
    #print(ttab_gngs.colnames)
    
    night_date = '2019-01-03'
    sites = list(sites_from_column_names(ttab_gngs.colnames))
    dt = time_slot_length(timestab_gngs['time'])
    # Resource API mock 
    resource = Resource('/resource_mock/data')
    resource.connect()
    
    fpu = resource.night_info('fpu', sites, night_date)
    fpur = resource.night_info('fpur', sites, night_date)
    grat = resource.night_info('grat', sites, night_date)
    instruments = resource.night_info('instr', sites, night_date)
    lgs = resource.night_info('LGS', sites, night_date)
    modes = resource.night_info('mode', sites, night_date)
    ifus = {'FPU':None, 'FPUr':None}
    ifus['FPU'] = resource.night_info('fpu-ifu', sites, night_date)
    ifus['FPUr'] = resource.night_info('fpur-ifu', sites, night_date)
    fpu2b = resource.fpu_to_barcode

    # Load observation and groups
    obs_ids = [row['obs_id'] for row in otab_gngs]
    grp_ids = [row['prog_ref'] for row in grptab_gngs]

    #bands = {obs_id: Band(otab_gngs[obs_id == obs_id]['band']) for obs_id in obs_ids}
    bands = {obs_id: None for obs_id in obs_ids} # NOTE: temporary bypass

    if 'inst' in otab_gngs.colnames or 'disperser' in otab_gngs.colnames:
        instruments = {obs_id: otab_gngs['inst'][idx] for idx, obs_id in enumerate(obs_ids)} 
        dispersers = {obs_id: otab_gngs['disperser'][idx] for idx, obs_id in enumerate(obs_ids)}
    else:
        # NOTE: Why is this optional? 
        instruments = {obs_id: None for idx, obs_id in enumerate(obs_ids)} 
        dispersers = {obs_id: None for idx, obs_id in enumerate(obs_ids)}

    # Time management for observation  
    obs_times = {obs_id: int(np.ceil(otab_gngs['obs_time'].quantity[idx] / dt.to(u.h).value)) for idx, obs_id in enumerate(obs_ids)}
    tot_times = {obs_id: int(np.ceil(otab_gngs['tot_time'].quantity[idx] / dt.to(u.h).value)) for idx, obs_id in enumerate(obs_ids)}
    categories = {obs_id: Category(otab_gngs['obsclass'][idx].lower()) for idx, obs_id in enumerate(obs_ids)}

    units = [] 
    for grp_id, group in enumerate(grptab_gngs):
        obs_idxs = group['oidx']
        can_be_split = group['split']
        
        standard_time = int(np.ceil(group['pstdt'].to(u.h).value/ dt.to(u.h).value))
        site = Site.GS if sum(ttab_gngs['weight_gs'][grp_id]) > 0 else Site.GN
       
        
        if len(obs_idxs) > 1: #group 
            
            observations = []
            calibrations = []
            for obs in obs_idxs:
                obs_id = obs_ids[obs]
                acq_value = int(np.ceil(10./60.) / dt.to(u.h).value) if dispersers[obs_id] == 'mirror' else int(np.ceil(15. / 60.)/dt.to(u.h).value)
                
                new_obs = Observation(obs, obs_id, bands[obs_id], 
                                categories[obs_id], obs_times[obs_id],
                                tot_times[obs_id], instruments[obs_id],
                                dispersers[obs_id], acq_value)
                if categories[obs_id] == Category.Science or categories[obs_id] == Category.ProgramCalibration:
                    observations.append(new_obs)
                else:
                    calibrations.append(new_obs)

        else: #single observation 
            obs = group['oidx'][0]
            obs_id = obs_ids[obs]
            acq_value = int(np.ceil(10./60.) / dt.to(u.h).value) if dispersers[obs_id] == 'mirror' else int(np.ceil(15. / 60.)/dt.to(u.h).value)
            
            new_obs = Observation(obs, obs_id, bands[obs_id], 
                                  categories[obs_id],obs_times[obs_id],
                                  tot_times[obs_id], instruments[obs_id],
                                  dispersers[obs_id], acq_value)
            observations = [new_obs] if new_obs == Category.Science or Category.ProgramCalibration else []
            calibrations = [new_obs] if new_obs == Category.PartnerCalibration  else []

        units.append(SchedulingUnit(grp_id, site, observations, calibrations, 
                                    can_be_split, standard_time))

    airmass_gs = {idx: otab_gngs['airmass_gs'][idx] for idx, obs_id in enumerate(obs_ids)}
    # airmass_gn = {idx: otab_gngs['airmass_gn'][idx] for idx, obs_id in enumerate(obs_ids)}
    weights_gs = {idx: ttab_gngs['weight_gs'][idx] for idx, obs_id in enumerate(grp_ids)}
    #weights_gn = {idx: ttab_gngs['weight_gn'][idx] for idx, obs_id in enumerate(grp_ids)}
    weights_gn = None
    airmass_gn = None
    weights = {Site.GS: weights_gs, Site.GN: weights_gn}
    airmass = {Site.GS: airmass_gs, Site.GN: airmass_gn}
   
    total_time_slots = len(ttab_gngs['weight_' + Site(sites[0]).name.lower()][0])
    #TODO: Calculate nt by site
    time_slots = TimeSlots(dt, weights, airmass, total_time_slots, fpu, fpur, grat, 
                            instruments, lgs, modes, fpu2b, ifus)

    # Convert to UT
    uttime_gngs = Time(timestab_gngs['time'], format='iso')
    #print(uttime_gngs[0], uttime_gngs[0].jd)
    nt = len(uttime_gngs) # NOTE: What is this for?

    # Make initial plan
    gm = GreedyMax(units, time_slots, sites)
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

