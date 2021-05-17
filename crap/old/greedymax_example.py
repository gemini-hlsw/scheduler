#!/usr/bin/env python

# An example of running the Greedy-Max scheduler algorithm
# Bryan Miller
# 2020 Oct 11

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from matplotlib import dates

import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.visualization import time_support
time_support()

import greedymax as gm

tabdir = './data/'

# Original, single site example

# Target table
ttab = Table.read(tabdir + 'targtab_metvisha_20201109.fits')

# Observation table
otab = Table.read(tabdir + 'obstab_20201109.fits')

# Time array
timestab = Table.read(tabdir + 'timetab_20201109.fits')
# print(timestab['time'][0])
dt = gm.deltat(timestab['time'])
print(dt)

# Convert to UT
uttime = Time(timestab['time'], format='iso')
print(uttime[0], uttime[0].jd)
print(timestab)
nt = len(uttime)

# Make initial plan
plan, targtab = gm.schedule_night(otab, ttab, dt, verbose=True)

# Optimize(?) plan
plan_opt = gm.optimize(plan, targets=targtab, verbose=False, summary=True)

# Print current plan
obs_order, i_start, i_end = gm._get_order(plan=plan)
# print(obs_order, i_start, i_end)
sum_score = 0.0
sum_metric = 0.0
time_used = 0.0 * u.hr
for i in range(len(obs_order)):
    if obs_order[i] >= 0:
        print('{:18} {} {} {:8.4f}'.format(gm.shortid(otab['obs_id'][obs_order[i]]),
             uttime[i_start[i]].strftime('%H:%M'),
             uttime[i_end[i]].strftime('%H:%M'),
             np.max(abs(targtab['weight'][obs_order[i]][i_start[i]:i_end[i]]))))
#                 print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
        sum_score += np.sum(abs(targtab['weight'][obs_order[i]][i_start[i]:i_end[i]+1]))
        sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i]+1])
        time_used += (i_end[i] - i_start[i] + 1) * dt
print('Sum score = {:7.2f}'.format(sum_score))
print('Sum score/time step = {:7.2f}'.format(sum_score / len(uttime)))
print('Sum metric = {:7.2f}'.format(sum_metric))
print('Sum metric/time step = {:7.2f}'.format(sum_metric / len(uttime)))
print('Time scheduled = {:5.2f}'.format(time_used))
print('Fraction of night scheduled = {:5.2f}'.format(time_used / (nt * dt)))

# Plot current plan
ax = plt.gca()
date_hhmm = dates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_hhmm)
# for label in ax.get_xticklabels():
#     label.set_rotation(20)
#     label.set_horizontalalignment('right')

for i in range(len(obs_order)):
    if obs_order[i] >= 0:
        ax.plot(uttime, abs(targtab['weight'][obs_order[i]]))
        # https://stackoverflow.com/questions/36699155/how-to-get-color-of-most-recent-plotted-line-in-pythons-plt
        colour = plt.gca().lines[-1].get_color()
        ax.plot(uttime[i_start[i]:i_end[i]+1], abs(targtab['weight'][obs_order[i]][i_start[i]:i_end[i]+1]),
                linewidth=4, color=colour, label=gm.shortid(otab['obs_id'][obs_order[i]]))
ax.set_ylabel('Score')
ax.set_xlabel('Time [UT]')
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

# Example of astropy.Table creation

# # arrays of data
# id = ['Obs1', 'Obs2', 'Obs3', 'Obs4', 'Obs5']
#
# scores = np.array([1.5, 3.5, 2.5, 8.5, 4.5])
#
# # List of column names
# columns = ['id', 'weight']
#
# # Make table
# targets = Table([id, scores], names=columns)
#
# print(targets)


# Multi-site example

# Target table
ttab_gngs = Table.read(tabdir + 'targtab_metvisha_gngs_20201123.fits')

# Observation table
otab_gngs = Table.read(tabdir + 'obstab_gngs_20201123.fits')
# Remove inst and disperser columns so that greedy-max doesn't add time for partner cals
# otab_gngs.remove_columns(['inst', 'disperser'])
print(otab_gngs.colnames)

# Time array
timestab_gngs = Table.read(tabdir + 'timetab_gngs_20201123.fits')
# print(timestab['time'][0])
dt = gm.deltat(timestab_gngs['time'])
print(dt.to(u.min))

# Convert to UT
uttime_gngs = Time(timestab_gngs['time'], format='iso')
print(uttime_gngs[0], uttime_gngs[0].jd)
# print(timestab_gngs)
nt = len(uttime_gngs)

# print(ttab_gngs['id'][0])
# for n in range(nt):
#     print('{:>4d} {} {:7.4f} {:7.4f}'.format(n, uttime_gngs[n], ttab_gngs['weight_gs'][0][n], ttab_gngs['weight_gn'][0][n]))

# Make initial plan
plan, obstab, targtab = gm.schedule_night_multi(otab_gngs, ttab_gngs, dt, verbose=True)

# Print current plan
# i_day = 0
sites = list(plan.copy().keys())
for s in sites:
    # if s == 'gs':
    #     night_length = gs_night_length[i_day]
    # else:
    #     night_length = gn_night_length[i_day]
    print(s.upper())
    obs_order, i_start, i_end = gm._get_order(plan=plan[s])
    # print(obs_order, i_start, i_end)
    sum_score = 0.0
    sum_metric = 0.0
    time_used = 0.0 * u.hr
    for i in range(len(obs_order)):
        if obs_order[i] >= 0:
            print('{:18} {} {} {:8.4f}'.format(gm.shortid(otab_gngs['obs_id'][obs_order[i]]),
                                               uttime_gngs[i_start[i]].strftime('%H:%M'),
                                               uttime_gngs[i_end[i]].strftime('%H:%M'),
                                               np.max(abs(targtab['weight_' + s][obs_order[i]][i_start[i]:i_end[i]]))))
            #                 print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
            sum_score += np.sum(abs(targtab['weight_' + s][obs_order[i]][i_start[i]:i_end[i] + 1]))
            sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i] + 1])
            time_used += (i_end[i] - i_start[i] + 1) * dt
    print('Sum score = {:7.2f}'.format(sum_score))
    print('Sum score/time step = {:7.2f}'.format(sum_score / nt))
    print('Sum metric = {:7.2f}'.format(sum_metric))
    print('Sum metric/time step = {:7.2f}'.format(sum_metric / nt))
    print('Time scheduled = {:5.2f}'.format(time_used))
    # print('Fraction of night scheduled = {:5.2f}'.format((time_used / night_length).value))
    print('')

# Plot plans
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

date_hhmm = dates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(date_hhmm)
ax2.xaxis.set_major_formatter(date_hhmm)

# for label in ax.get_xticklabels():
#     label.set_rotation(20)
#     label.set_horizontalalignment('right')

# inight = 0
s = 'gs'
obs_order, i_start, i_end = gm._get_order(plan=plan[s])
for i in range(len(obs_order)):
    if obs_order[i] >= 0:
        p = ax1.plot(uttime_gngs, abs(targtab['weight_'+s][obs_order[i]]))
        colour = p[-1].get_color()
        ax1.plot(uttime_gngs[i_start[i]:i_end[i]+1], abs(targtab['weight_'+s][obs_order[i]][i_start[i]:i_end[i]+1]),
                linewidth=4, color=colour, label=gm.shortid(obstab['obs_id'][obs_order[i]]))

ax1.legend()

s = 'gn'
obs_order, i_start, i_end = gm._get_order(plan=plan[s])
for i in range(len(obs_order)):
    if obs_order[i] >= 0:
        p = ax2.plot(uttime_gngs, abs(targtab['weight_'+s][obs_order[i]]))
        colour = p[-1].get_color()
        ax2.plot(uttime_gngs[i_start[i]:i_end[i]+1], abs(targtab['weight_'+s][obs_order[i]][i_start[i]:i_end[i]+1]),
                linewidth=4, color=colour, label=gm.shortid(obstab['obs_id'][obs_order[i]]))

ax2.legend()

# ymin, ymax = ax1.get_ylim()
# ax1.vlines(gs_twi_mor12[0], 0, ymax, color='black', linestyle='--')
# ymin, ymax = ax2.get_ylim()
# ax2.vlines(gn_twi_eve12[0], 0, ymax, color='black', linestyle='--')

ax1.set_xlabel('Time [UT]')
ax2.set_xlabel('Time [UT]')

ax1.set_ylabel('Score')
ax1.set_title('Gemini South')
ax2.set_title('Gemini North')

# plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()
