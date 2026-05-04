import numpy as np
import numpy.typing as npt
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import requests
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time, TimeDelta
import astropy.units as u

from lucupy.minimodel import NightIndex, Observation, Target

from scheduler.services.logger_factory import create_logger
from scheduler.services.visibility.snapshot import VisibilitySnapshot, TargetSnapshot

_logger = create_logger(__name__, with_id=False)


class SightClient:
    """Client for the Sight visibility pre-calculation service."""

    def __init__(self, api_url: str):
        self.api_url = api_url

    def get_visible_observations(
        self,
        observations: list,
        night_date: date,
    ) -> List[str]:
        """Filter observations to only those visible on a given night.

        Args:
            observations: List of Observation objects (lucupy minimodel).
            night_date: The night to check visibility for.

        Returns:
            List of observation ID strings that are visible on the given night.
        """
        obs_payload = []
        for obs in observations:
            base = obs.base_target()
            entry = {
                "observation_id": obs.id.id,
                "target_name": base.name if base else "",
                "site_id": obs.site.name,
            }
            #if obs.constraints:

            #    entry["constraints"] = {
            #        "iq": obs.constraints.conditions.iq.name if obs.constraints.conditions else None,
            #        "cc": obs.constraints.conditions.cc.name if obs.constraints.conditions else None,
            #        "sb": obs.constraints.conditions.sb.name if obs.constraints.conditions else None,
            #        "wv": obs.constraints.conditions.wv.name if obs.constraints.conditions else None,
            #        "elevation_type": obs.constraints.elevation_type.name if obs.constraints.elevation_type else None,
            #        "elevation_min": obs.constraints.elevation_min,
            #        "elevation_max": obs.constraints.elevation_max,
            #        "timing_windows": [
            #            {"start": tw.start.isoformat(), "end": tw.end.isoformat()}
            #            for tw in (obs.constraints.timing_windows or [])
            #        ],
            #    }
            obs_payload.append(entry)

        response = requests.post(
            f"{self.api_url}/visibility/visible_observations",
            json={
                "observations": obs_payload,
                "night_date": night_date.isoformat(),
            },
            timeout=300,
        )
        response.raise_for_status()
        visible_response = response.json().get("visible_observations", [])

        return [ o.get("observation_id") for o in visible_response ]

    def get_cumulative_visibility(
        self,
        observation_ids: List[str],
        start_date: date,
        end_date: date,
    ) -> dict:
        """Fetch cumulative remaining visibility minutes for observations.

        Args:
            observation_ids: List of observation ID strings.
            start_date: Start of the date range.
            end_date: End of the date range.

        Returns:
            Dict with structure:
                {
                    'observations': {
                        obs_id: {
                            'targets': {
                                target_name: {
                                    'site': str,
                                    'cumulative_remaining_minutes': float,
                                    'nights_with_visibility': int,
                                }
                            }
                        }
                    }
                }
        """
        response = requests.post(
            f"{self.api_url}/visibility/cumulative",
            json={
                "observation_ids": observation_ids,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def get_precalculated_visibility_bulk(
        self,
        observation_ids: List[str],
        start_date: date,
        end_date: date,
    ) -> dict:
        """Retrieve bulk pre-calculated visibility from the Sight API.

        Returns dict with structure:
            {
                'observations': {
                    obs_id: {
                        'targets': {
                            target_name: {
                                'nights': {
                                    "2018-08-15": {
                                        "night_date": "2018-08-15",
                                        "site": "GS",
                                        "remaining_minutes": 126,
                                        "visible_ranges": [[547, 672], ...],
                                    }
                                }
                            }
                        }
                    }
                }
            }
        """
        response = requests.post(
            f"{self.api_url}/visibility/precalculated/bulk",
            json={
                "observation_ids": observation_ids,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def get_target_snapshots_bulk(
        self,
        target_names: List[str],
        site_ids: List[str],
        start_date: date,
        end_date: date,
        batch_size: int = 100,
    ) -> dict:
        """Retrieve bulk target snapshot data from the Sight API in batches.

        Splits the target list into chunks of ``batch_size`` and issues one
        POST per chunk, merging the results into a single dict.

        Args:
            target_names: List of target names to fetch snapshots for.
            site_ids: List of site identifiers (e.g. ["GN", "GS"]).
            start_date: Start date of the range.
            end_date: End date of the range.
            batch_size: Maximum number of targets per request (default 100).

        Returns:
            {
                target_name: {
                    "nights": {
                        "GN_2025-02-15": {
                            "night_date": "2025-02-15",
                            "site": "GN",
                            "night_duration_minutes": 653,
                            "alt": [float, ...],
                            "airmass": [float, ...],
                        },
                        ...
                    }
                },
                ...
            }
        """
        merged: dict = {}
        total_batches = (len(target_names) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch = target_names[batch_start:batch_end]

            _logger.info(
                f'Fetching target snapshots batch {batch_idx + 1}/{total_batches} '
                f'({len(batch)} targets)...'
            )

            response = requests.post(
                f"{self.api_url}/stage1/greedymax/bulk",
                json={
                    "target_names": batch,
                    "site_ids": site_ids,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                timeout=300,
            )
            response.raise_for_status()
            merged.update(response.json())

        return merged

    @staticmethod
    def _date_str_to_jd_str(date_str: str) -> str:
        """Convert a date string like '2018-08-15' to a julian day string like '2458345'."""
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        return str(int(Time(dt).jd))

    def parse_visibility_for_obs(
        self,
        obs_vis_data: dict,
    ) -> Dict[str, VisibilitySnapshot]:
        """Parse endpoint visibility data into VisibilitySnapshot dict keyed by julian day string.

        Sight returns night keys as date strings (e.g. '2018-08-15').
        The rest of the scheduler expects julian day strings (e.g. '2458345').

        Args:
            obs_vis_data: The per-observation data from the endpoint, containing
                         'targets' -> target_name -> 'nights' -> date_str -> vis data
        """
        vis_snapshots: Dict[str, VisibilitySnapshot] = {}

        for target_name, target_data in obs_vis_data.get('targets', {}).items():
            for date_str, night_data in target_data.get('nights', {}).items():
                jd_str = self._date_str_to_jd_str(date_str)
                vis_snapshots[jd_str] = VisibilitySnapshot.from_dict(night_data)

        return vis_snapshots

    @staticmethod
    def parse_target_snapshot_for_night(night_data: dict) -> TargetSnapshot:
        """Parse greedymax endpoint target snapshot data into a TargetSnapshot.

        The greedymax endpoint only returns 'alt' and 'airmass' arrays.
        All other fields (coord, az, hourangle, par_ang) are set to None.

        Args:
            night_data: Dict with 'alt' and 'airmass' arrays, plus metadata
                        fields 'night_date', 'site', 'night_duration_minutes'.
        """
        alt = Angle(night_data['alt'], unit=u.rad)
        az = Angle(night_data['az'], unit=u.rad)

        coord = SkyCoord(ra=night_data['ra'] * u.deg, dec=night_data['dec'] * u.deg, frame='icrs')
        hour_angle = Angle(night_data['hourangle'], unit=u.rad)
        airmass = np.array(night_data['airmass'])

        return TargetSnapshot(
            coord=coord,
            alt=alt,
            az=az,
            hourangle=hour_angle,
            airmass=airmass,
            par_ang=None,
        )
