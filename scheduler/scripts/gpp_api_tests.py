# The entire document was commented out, it should be uncommented when pyexplore get replaced
# TODO change pyexplore to other api query
# #!/usr/bin/env python3
# # Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# # For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

# import os, sys

# # Requires https://github.com/andrewwstephens/pyexplore
# # sys.path.append(os.path.join(os.environ['PYEXPLORE']))
# from pyexplore import explore


# if __name__ == '__main__':

#     # Calls for Proposals
#     for cfp in explore.cfp():
#         print(cfp)

#     # List programs
#     programs = explore.programs()
#     progid = None
#     for p in programs:
#         print(f'{p.id}: {p.name}')
#         progid = p.id if progid is None else progid
#     print("")

#     # Information from the first program
#     # print(progid)
#     prog = explore.program(progid)
#     # print(prog)
#     print(f'{progid} {prog.reference}: {prog.name} {prog.pi.orcid_family_name}')

#     # List observations in the program
#     # From program info
#     print("Observations")
#     for o in prog.observations.matches:
#         print(f"\t{o.id} {o.title} {o.index} {o.group_id} {o.group_index}")
#     print("")

#     # From observations query
#     for o in explore.observations(progid, include_deleted=False):
#         print(f"\t{o.id}: {o.title}")
#     print("")

#     # Initial scheduler observation query
#     obs_for_sched = explore.observations_for_scheduler(include_deleted=False)
#     for o in obs_for_sched:
#         print(f'{o.id}: {o.title} {o.active_status} {o.status}')

#     # List targets
#     targets = explore.targets(progid, include_deleted=False)
#     print("Targets")
#     for i, t in enumerate(targets):
#         print(f'\t{i} {t.id}: {t.name}')

#     print(targets[0])
#     print(targets[4])
#     t = targets[2]
#     print(
#         f'{t.id}: {t.name:20} {t.sidereal.ra.hms} {t.sidereal.dec.dms} {t.sidereal.proper_motion.ra.milliarcseconds_per_year} '
#         f'{t.sidereal.proper_motion.dec.milliarcseconds_per_year}')

#     # Observation in group
#     print(f"\nObservation in group")
#     obs_mayall5_grp = explore.observation(prog.observations.matches[0].id)
#     print(obs_mayall5_grp.id, obs_mayall5_grp.title, obs_mayall5_grp.status)
#     print(obs_mayall5_grp.group_id, obs_mayall5_grp.group_index)

#     print(f"\nGroup Information")
#     grp = explore.group(obs_mayall5_grp.group_id)
#     print(grp.id, grp.parent_id, grp.name, grp.minimum_required, grp.ordered)
#     print(grp.minimum_interval, grp.maximum_interval)
#     # print(grp.elements)
#     for element in grp.elements:
#         print(f"\t{element.parent_group_id}  {element.group}  {element.observation.id}" )

#     # Sequence
#     sequence = explore.sequence(obs_mayall5_grp.id, include_acquisition=True)
#     print(f"Sequence for {obs_mayall5_grp.id}")
#     for step in sequence:
#         print(step['atom'], step['class'], step['type'], step['exposure'])
#     print(sequence)

#     print(f"\nAtom information")
#     explore.sequence_atoms(obs_mayall5_grp.id, include_acquisition=True)

#     # Constraints
#     print(f"\nWeather Constraints")
#     for c in explore.get_constraints(progid):
#         print(c)

