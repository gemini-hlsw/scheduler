from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Site, Target, NightEvent, TargetNightData, VisibilityData
from scheduler.services.sight.database.repositories import (
    SiteRepository,
    NightEventRepository,
    TargetRepository,
    TargetNightDataRepository,
    VisibilityDataRepository
)
from scheduler.services.sight.calculations.night_events import calculate_night_events_for_night
from scheduler.services.sight.calculations.stage2 import calculate_visibility, ObservationConstraints as Stage2Constraints
from scheduler.services.sight.calculations.arrays import unpack_array

from .models import (
    ObservationRequest,
    VisibilityResult,
    CalculationResponse,
    mask_to_ranges,
    TargetCreate,
    TargetResponse,
    BulkTargetCreateResponse,
    BulkVisibilityResponse,
    PrecalculatedVisibilityResult,
    
)
from .constants import SITE_KEY_TO_ID, SITE_ID_TO_KEY


class Calculator:
    """
    Main service for visibility calculations.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.site_repo = SiteRepository(session)
        self.night_repo = NightEventRepository(session)
        self.target_repo = TargetRepository(session)
        self.target_data_repo = TargetNightDataRepository(session)
        self.visibility_repo = VisibilityDataRepository(session)
        
        self._sites_cache: dict[int, Site] | None = None
    
    async def get_sites(self) -> dict[int, Site]:
        """Get all sites, cached in memory."""
        if self._sites_cache is None:
            self._sites_cache = await self.site_repo.get_all_as_dict()
        return self._sites_cache
    
    async def calculate_visibility(
        self,
        requests: list[ObservationRequest],
        night_date: date,
    ) -> CalculationResponse:
        """
        Calculate visibility for a list of observation requests.
        """
        if not requests:
            return CalculationResponse(results=[], night_date=night_date)
        
        # Convert site keys to IDs
        site_ids = set(SITE_KEY_TO_ID[r.site_id] for r in requests)
        target_names = set(r.target_name for r in requests)
        
        # Load targets by name
        targets_by_name = {}
        for name in target_names:
            target = await self.target_repo.get_by_name(name)
            if target:
                targets_by_name[name] = target
        
        target_ids = set(t.id for t in targets_by_name.values())
        
        await self._ensure_night_events(site_ids, night_date)
        await self._ensure_stage1_data(target_ids, site_ids, night_date)
        
        night_events = {
            ne.site_id: ne 
            for ne in await self.night_repo.get_for_multiple_sites(
                list(site_ids), night_date
            )
        }
        
        target_data = {}
        for site_id in site_ids:
            data_list = await self.target_data_repo.get_for_targets_on_night(
                list(target_ids), site_id, night_date
            )
            for data in data_list:
                target_data[(data.target_id, data.site_id)] = data

        results = []
        for request in requests:
            target = targets_by_name.get(request.target_name)
            if target is None:
                continue
            
            site_id = SITE_KEY_TO_ID[request.site_id]
            night_event = night_events.get(site_id)
            stage1_data = target_data.get((target.id, site_id))
            
            if night_event is None or stage1_data is None:
                continue
            
            result = self._calculate_stage2(
                request, night_event, stage1_data, target.name
            )
            results.append(result)
        
        return CalculationResponse(results=results, night_date=night_date)
    
    async def _ensure_night_events(
        self,
        site_ids: set[int],
        night_date: date,
    ) -> None:
        """Ensure night events exist for all required sites."""
        sites = await self.get_sites()
        
        for site_id in site_ids:
            exists = await self.night_repo.exists(site_id, night_date)
            if not exists:
                site = sites[site_id]
                await self._calculate_and_store_night_events(site, night_date)
    
    async def _calculate_and_store_night_events(
        self,
        site: Site,
        night_date: date,
    ) -> NightEvent:
        """Calculate night events for a site and store them."""
        arrays = calculate_night_events_for_night(site, night_date)
        
        night_event = await self.night_repo.create(
            site_id=site.id,
            night_date=night_date,
            night_duration_minutes=arrays.night_duration_minutes,
            sunset=arrays.sunset,
            sunrise=arrays.sunrise,
            night_start=arrays.night_start,
            night_end=arrays.night_end,
            midnight=arrays.midnight,
            twilight_evening_12=arrays.twilight_evening_12,
            twilight_morning_12=arrays.twilight_morning_12,
            moonrise=arrays.moonrise,
            moonset=arrays.moonset,
            moon_dist=arrays.moon_dist,
            sun_alt=arrays.sun_alt,
            sun_az=arrays.sun_az,
            sun_par_ang=arrays.sun_par_ang,
            moon_alt=arrays.moon_alt,
            moon_az=arrays.moon_az,
            moon_par_ang=arrays.moon_par_ang,
            moon_ra=arrays.moon_ra,
            moon_dec=arrays.moon_dec,
            sun_moon_ang=arrays.sun_moon_ang,
            local_sidereal_times=arrays.local_sidereal_times,
        )
        
        return night_event
    
    async def _ensure_stage1_data(
        self,
        target_ids: set[int],
        site_ids: set[int],
        night_date: date,
    ) -> None:
        """Ensure Stage 1 data exists for all required targets at all sites."""
        targets = {t.id: t for t in await self.target_repo.get_many(list(target_ids))}
        
        for target_id in target_ids:
            target = targets.get(target_id)
            if target is None:
                continue
            
            for site_id in site_ids:
                data = await self.target_data_repo.get_by_target_site_night(
                    target_id, site_id, night_date
                )
                
                if data is None or data.is_stale(target):
                    await self._calculate_and_store_stage1(
                        target, site_id, night_date
                    )
    
    async def _calculate_and_store_stage1(
        self,
        target: Target,
        site_id: int,
        night_date: date,
    ) -> TargetNightData:
        """Calculate Stage 1 data for a target and store it."""
        night_event = await self.night_repo.get_by_site_and_night(site_id, night_date)
        if night_event is None:
            raise ValueError(f"Night event not found for site {site_id} on {night_date}")
        
        sites = await self.get_sites()
        site = sites[site_id]
        
        from calculations.stage1 import calculate_stage1
        
        arrays = calculate_stage1(target, site, night_event)
        
        data = await self.target_data_repo.upsert(
            target_id=target.id,
            site_id=site_id,
            night_date=night_date,
            night_duration_minutes=arrays.night_duration_minutes,
            ra=arrays.ra,
            dec=arrays.dec,
            alt=arrays.alt,
            az=arrays.az,
            hourangle=arrays.hourangle,
            airmass=arrays.airmass,
            par_ang=arrays.par_ang,
            target_updated_at=target.updated_at,
        )
        
        return data
    
    def _calculate_stage2(
        self,
        request: ObservationRequest,
        night_event: NightEvent,
        stage1_data: TargetNightData,
        target_name: str,
    ) -> VisibilityResult:
        """Calculate Stage 2 visibility for a single observation."""
        stage2_constraints = Stage2Constraints.model_validate(
            request.constraints.model_dump()
        )
        
        result = calculate_visibility(
            alt_bytes=stage1_data.alt,
            az_bytes=stage1_data.az,
            airmass_bytes=stage1_data.airmass,
            hourangle_bytes=stage1_data.hourangle,
            ra_bytes=stage1_data.ra,
            dec_bytes=stage1_data.dec,
            sun_alt_bytes=night_event.sun_alt,
            moon_alt_bytes=night_event.moon_alt,
            moon_ra_bytes=night_event.moon_ra,
            moon_dec_bytes=night_event.moon_dec,
            sun_moon_ang_bytes=night_event.sun_moon_ang,
            moon_dist=night_event.moon_dist,
            night_start=night_event.night_start,
            night_duration_minutes=stage1_data.night_duration_minutes,
            constraints=stage2_constraints,
        )
        
        return VisibilityResult(
            observation_id=request.observation_id,
            target_name=target_name,
            site=request.site_id,
            night_date=night_event.night_date,
            remaining_minutes=result.remaining_minutes,
            visible_ranges=mask_to_ranges(result.visibility_mask),
        )

    async def get_visible_observations(
        self,
        requests: list[ObservationRequest],
        night_date: date,
    ) -> list[VisibilityResult]:
        """
        Return only observations that are visible on the given night.

        First checks for pre-calculated Stage 2 data in the DB.
        Only calculates from scratch for observations without stored results.
        """
        if not requests:
            return []

        # Check for pre-calculated visibility
        obs_ids = [r.observation_id for r in requests]
        precalculated = await self.visibility_repo.get_by_observation_ids_on_night(
            obs_ids, night_date
        )
        precalc_by_obs_id = {d.observation_id: d for d in precalculated}
        # Load target names for precalculated results
        precalc_target_ids = set(d.target_id for d in precalculated)
        targets_by_id = {}
        if precalc_target_ids:
            targets_by_id = {
                t.id: t for t in await self.target_repo.get_many(list(precalc_target_ids))
            }

        visible: list[VisibilityResult] = []
        missing_requests: list[ObservationRequest] = []

        for request in requests:
            data = precalc_by_obs_id.get(request.observation_id)
            if data is not None:
                # Use pre-calculated result
                if data.remaining_minutes > 0:
                    target = targets_by_id.get(data.target_id)
                    target_name = target.name if target else request.target_name
                    visible.append(VisibilityResult(
                        observation_id=data.observation_id,
                        target_name=target_name,
                        site=SITE_ID_TO_KEY[data.site_id],
                        night_date=data.night_date,
                        remaining_minutes=data.remaining_minutes,
                        visible_ranges=data.visible_ranges,
                    ))
            else:
                missing_requests.append(request)

        # Calculate from scratch only for observations without stored results
        if missing_requests:
            response = await self.calculate_visibility(
                requests=missing_requests,
                night_date=night_date,
            )
            visible.extend(r for r in response.results if r.remaining_minutes > 0)

        return visible

    async def create_targets_bulk(
        self,
        targets: list[TargetCreate],
        start_date: date,
        end_date: date,
    ) -> BulkTargetCreateResponse:
        """
        Create multiple targets and pre-compute Stage 1 data.
        """
        created_targets = []
        errors = []
        
        # Get all site IDs
        sites = await self.get_sites()
        site_ids = set(sites.keys())
        
        for target_data in targets:
            try:
                # Check if target already exists
                existing = await self.target_repo.get_by_name(target_data.name)
                if existing:
                    errors.append(f"Target '{target_data.name}' already exists")
                    continue
                
                target = await self.target_repo.create(
                    name=target_data.name,
                    is_sidereal=target_data.is_sidereal,
                    base_ra=target_data.base_ra,
                    base_dec=target_data.base_dec,
                    pm_ra=target_data.pm_ra,
                    pm_dec=target_data.pm_dec,
                    epoch=target_data.epoch,
                    horizons_id=target_data.horizons_id,
                    tag=target_data.tag,
                )
                
                # Compute Stage 1 for date range
                current_date = start_date
                while current_date <= end_date:
                    # Ensure night events exist
                    await self._ensure_night_events(site_ids, current_date)
                    
                    # Compute Stage 1 for this target
                    for site_id in site_ids:
                        await self._calculate_and_store_stage1(target, site_id, current_date)
                    
                    current_date += timedelta(days=1)
                
                created_targets.append(TargetResponse.model_validate(target))
                
            except Exception as e:
                errors.append(f"Failed to create '{target_data.name}': {str(e)}")
        
        return BulkTargetCreateResponse(
            created=len(created_targets),
            failed=len(errors),
            targets=created_targets,
            errors=errors,
        )
    
    async def calculate_visibility_bulk(
        self,
        requests: list[ObservationRequest],
        start_date: date,
        end_date: date,
    ) -> BulkVisibilityResponse:
        """
        Calculate visibility for observations across a date range.
        """
        results_by_date: dict[str, list[VisibilityResult]] = {}
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            
            response = await self.calculate_visibility(
                requests=requests,
                night_date=current_date,
            )
            
            results_by_date[date_str] = response.results
            current_date += timedelta(days=1)
        
        total_nights = (end_date - start_date).days + 1
        
        return BulkVisibilityResponse(
            results=results_by_date,
            start_date=start_date,
            end_date=end_date,
            total_nights=total_nights,
        )

    async def precompute_stage1(
        self,
        start_date: date,
        end_date: date,
        target_names: list[str] | None = None,
        site_ids: list[str] | None = None,
    ) -> dict:
        """
        Pre-compute Stage 1 data for existing targets across a date range.
        
        If target_names is None, computes for all targets.
        If site_ids is None, computes for all sites.
        """
        # Get targets
        if target_names:
            targets = []
            for name in target_names:
                target = await self.target_repo.get_by_name(name)
                if target:
                    targets.append(target)
        else:
            targets = await self.target_repo.get_all()
        
        if not targets:
            return {
                "targets": 0,
                "sites": 0,
                "nights": 0,
                "total_computations": 0,
            }
        
        # Get sites
        if site_ids:
            site_id_ints = set(SITE_KEY_TO_ID[s] for s in site_ids)
        else:
            site_id_ints = set(SITE_KEY_TO_ID.values())
        
        target_ids = set(t.id for t in targets)
        total = 0
        nights = 0
        
        current_date = start_date
        while current_date <= end_date:
            # Ensure night events exist
            await self._ensure_night_events(site_id_ints, current_date)
            
            # Compute Stage 1 for all targets
            await self._ensure_stage1_data(target_ids, site_id_ints, current_date)
            
            total += len(targets) * len(site_id_ints)
            nights += 1
            current_date += timedelta(days=1)
        
        return {
            "targets": len(targets),
            "sites": len(site_id_ints),
            "nights": nights,
            "total_computations": total,
        }

    async def store_visibility(
        self,
        requests: list[ObservationRequest],
        start_date: date,
        end_date: date,
    ) -> dict:
        """
        Calculate and store visibility for observations across a date range.
        """
        stored = 0

        # Load targets by name
        target_names = set(r.target_name for r in requests)
        targets_by_name = {}
        for name in target_names:
            target = await self.target_repo.get_by_name(name)
            if target:
                targets_by_name[name] = target

        current_date = start_date
        while current_date <= end_date:
            # Calculate visibility for this night
            response = await self.calculate_visibility(
                requests=requests,
                night_date=current_date,
            )
            
            # Store each result
            for result in response.results:
                target = targets_by_name.get(result.target_name)
                if not target:
                    continue
                
                site_id = SITE_KEY_TO_ID[result.site]
                
                # Find the original request to get constraints
                original_request = next(
                    (r for r in requests if r.observation_id == result.observation_id),
                    None
                )
                constraints_dict = original_request.constraints.model_dump(mode="json") if original_request else {}
                
                await self.visibility_repo.upsert(
                    observation_id=result.observation_id,
                    target_id=target.id,
                    site_id=site_id,
                    night_date=result.night_date,
                    remaining_minutes=result.remaining_minutes,
                    visible_ranges=result.visible_ranges,
                    constraints=constraints_dict
                )
                stored += 1
            
            current_date += timedelta(days=1)

        nights = (end_date - start_date).days + 1

        return {
            "stored": stored,
            "nights": nights,
        }

    async def get_precalculated_visibility(
        self,
        start_date: date,
        end_date: date,
        observation_id: str | None = None,
        target_name: str | None = None,
    ) -> list[PrecalculatedVisibilityResult]:
        """
        Retrieve pre-calculated visibility from the database.
        """
        results = []

        if observation_id and target_name:
            # Get by both
            target = await self.target_repo.get_by_name(target_name)
            if not target:
                return []
            
            data_list = await self.visibility_repo.get_by_observation_and_target(
                observation_id=observation_id,
                target_id=target.id,
                start_date=start_date,
                end_date=end_date,
            )
        elif observation_id:
            # Get by observation only
            data_list = await self.visibility_repo.get_by_observation(
                observation_id=observation_id,
                start_date=start_date,
                end_date=end_date,
            )
        elif target_name:
            # Get by target only
            target = await self.target_repo.get_by_name(target_name)
            if not target:
                return []
            
            data_list = await self.visibility_repo.get_by_target(
                target_id=target.id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return []

        # Load target names for results
        target_ids = set(d.target_id for d in data_list)
        targets = {t.id: t for t in await self.target_repo.get_many(list(target_ids))}

        for data in data_list:
            target = targets.get(data.target_id)
            if not target:
                continue
            
            results.append(PrecalculatedVisibilityResult(
                observation_id=data.observation_id,
                target_name=target.name,
                site=SITE_ID_TO_KEY[data.site_id],
                night_date=data.night_date,
                remaining_minutes=data.remaining_minutes,
                visible_ranges=data.visible_ranges
            ))

        return results
    
    async def get_precalculated_visibility_bulk(
        self,
        observation_ids: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, dict]:
        """
        Retrieve pre-calculated visibility grouped by observation and target.
        """
        from sqlalchemy import select, and_
        
        if not observation_ids:
            return {}
        
        # Query all visibility data for these observations
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id.in_(observation_ids),
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(
            VisibilityData.observation_id,
            VisibilityData.night_date,
        )
        
        result = await self.session.execute(stmt)
        data_list = result.scalars().all()
        
        if not data_list:
            return {}
        
        # Load all target names
        all_target_ids = set(d.target_id for d in data_list)
        targets = {t.id: t for t in await self.target_repo.get_many(list(all_target_ids))}
        
        # Build nested structure
        observations = {}
        
        for data in data_list:
            obs_id = data.observation_id
            target = targets.get(data.target_id)
            if not target:
                continue
            
            target_name = target.name
            date_str = data.night_date.isoformat()
            
            # Initialize nested dicts
            if obs_id not in observations:
                observations[obs_id] = {"targets": {}}
            
            if target_name not in observations[obs_id]["targets"]:
                observations[obs_id]["targets"][target_name] = {"nights": {}}
            
            # Add visibility data
            observations[obs_id]["targets"][target_name]["nights"][date_str] = {
                "night_date": data.night_date,
                "site": SITE_ID_TO_KEY[data.site_id],
                "remaining_minutes": data.remaining_minutes,
                "visible_ranges": data.visible_ranges,
            }
        
        return observations

    async def get_cumulative_remaining_visibility(
        self,
        observation_ids: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, dict]:
        """
        Sum remaining_minutes across all nights in the date range,
        grouped by observation and target.
        """
        from sqlalchemy import select, and_

        if not observation_ids:
            return {}

        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id.in_(observation_ids),
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(
            VisibilityData.observation_id,
        )

        result = await self.session.execute(stmt)
        data_list = result.scalars().all()

        if not data_list:
            return {}

        # Load target names
        all_target_ids = set(d.target_id for d in data_list)
        targets = {t.id: t for t in await self.target_repo.get_many(list(all_target_ids))}

        # Accumulate per observation/target
        observations: dict[str, dict[str, dict]] = {}

        for data in data_list:
            target = targets.get(data.target_id)
            if not target:
                continue

            obs_id = data.observation_id
            target_name = target.name

            if obs_id not in observations:
                observations[obs_id] = {}

            if target_name not in observations[obs_id]:
                observations[obs_id][target_name] = {
                    "site": SITE_ID_TO_KEY[data.site_id],
                    "cumulative_remaining_minutes": 0,
                    "nights_with_visibility": 0,
                }

            entry = observations[obs_id][target_name]
            entry["cumulative_remaining_minutes"] += data.remaining_minutes
            if data.remaining_minutes > 0:
                entry["nights_with_visibility"] += 1

        # Wrap into expected shape
        return {
            obs_id: {"targets": tgts}
            for obs_id, tgts in observations.items()
        }

    async def get_stage1_bulk(
    self,
    target_names: list[str],
    site_ids: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, dict]:
        """
        Retrieve Stage 1 data in bulk.
        
        Returns:
            {
                target_name: {
                    "nights": {
                        "GN_2025-02-15": {
                            "night_date": "2025-02-15",
                            "site": "GN",
                            "night_duration_minutes": 653,
                            "ra": [...],
                            "dec": [...],
                            "alt": [...],
                            "az": [...],
                            "hourangle": [...],
                            "airmass": [...],
                            "par_ang": [...]
                        },
                        ...
                    }
                },
                ...
            }
        """
        from sqlalchemy import select, and_
        
        if not target_names or not site_ids:
            return {}
        
        # Get targets by name
        targets_by_name = {}
        for name in target_names:
            target = await self.target_repo.get_by_name(name)
            if target:
                targets_by_name[name] = target
        
        if not targets_by_name:
            return {}
        
        target_ids = [t.id for t in targets_by_name.values()]
        site_id_ints = [SITE_KEY_TO_ID[s] for s in site_ids]
        
        # Query all Stage 1 data
        stmt = select(TargetNightData).where(
            and_(
                TargetNightData.target_id.in_(target_ids),
                TargetNightData.site_id.in_(site_id_ints),
                TargetNightData.night_date >= start_date,
                TargetNightData.night_date <= end_date,
            )
        ).order_by(
            TargetNightData.target_id,
            TargetNightData.site_id,
            TargetNightData.night_date,
        )
        
        result = await self.session.execute(stmt)
        data_list = result.scalars().all()
        
        if not data_list:
            return {}
        
        # Build target_id -> name mapping
        id_to_name = {t.id: name for name, t in targets_by_name.items()}
        
        # Build nested structure
        targets = {}
        
        for data in data_list:
            target_name = id_to_name.get(data.target_id)
            if not target_name:
                continue
            
            site_key = SITE_ID_TO_KEY[data.site_id]
            date_str = data.night_date.isoformat()
            key = f"{site_key}_{date_str}"
            
            n = data.night_duration_minutes
            
            # Initialize nested dicts
            if target_name not in targets:
                targets[target_name] = {"nights": {}}
            
            # Unpack arrays
            targets[target_name]["nights"][key] = {
                "night_date": data.night_date,
                "site": site_key,
                "night_duration_minutes": n,
                "ra": unpack_array(data.ra, n).tolist(),
                "dec": unpack_array(data.dec, n).tolist(),
                "alt": unpack_array(data.alt, n).tolist(),
                "az": unpack_array(data.az, n).tolist(),
                "hourangle": unpack_array(data.hourangle, n).tolist(),
                "airmass": unpack_array(data.airmass, n).tolist(),
                "par_ang": unpack_array(data.par_ang, n).tolist() if data.par_ang else None,
            }
        
        return targets

    async def get_stage1_greedymax_bulk(
    self,
    target_names: list[str],
    site_ids: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, dict]:
        """
        Retrieve minimal Stage 1 data (alt + airmass only) in bulk.
        
        Returns:
            {
                target_name: {
                    "nights": {
                        "GN_2025-02-15": {
                            "night_date": "2025-02-15",
                            "site": "GN",
                            "night_duration_minutes": 653,
                            "alt": [...],
                            "airmass": [...]
                        },
                        ...
                    }
                },
                ...
            }
        """
        from sqlalchemy import select, and_
        
        if not target_names or not site_ids:
            return {}
        
        # Get targets by name
        targets_by_name = {}
        for name in target_names:
            target = await self.target_repo.get_by_name(name)
            if target:
                targets_by_name[name] = target
        
        if not targets_by_name:
            return {}
        
        target_ids = [t.id for t in targets_by_name.values()]
        site_id_ints = [SITE_KEY_TO_ID[s] for s in site_ids]
        
        # Query only needed columns
        stmt = select(
            TargetNightData.target_id,
            TargetNightData.site_id,
            TargetNightData.night_date,
            TargetNightData.night_duration_minutes,
            TargetNightData.ra,
            TargetNightData.dec,
            TargetNightData.alt,
            TargetNightData.az,
            TargetNightData.airmass,
            TargetNightData.hourangle,
        ).where(
            and_(
                TargetNightData.target_id.in_(target_ids),
                TargetNightData.site_id.in_(site_id_ints),
                TargetNightData.night_date >= start_date,
                TargetNightData.night_date <= end_date,
            )
        ).order_by(
            TargetNightData.target_id,
            TargetNightData.site_id,
            TargetNightData.night_date,
        )
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        if not rows:
            return {}
        
        # Build target_id -> name mapping
        id_to_name = {t.id: name for name, t in targets_by_name.items()}
        
        # Build nested structure
        targets = {}
        
        for row in rows:
            target_name = id_to_name.get(row.target_id)
            if not target_name:
                continue
            
            site_key = SITE_ID_TO_KEY[row.site_id]
            date_str = row.night_date.isoformat()
            key = f"{site_key}_{date_str}"
            
            n = row.night_duration_minutes
            
            # Initialize nested dicts
            if target_name not in targets:
                targets[target_name] = {"nights": {}}
            
            # Unpack only alt, airmass, and hourangle
            targets[target_name]["nights"][key] = {
                "night_date": row.night_date,
                "site": site_key,
                "night_duration_minutes": n,
                "ra": unpack_array(row.ra, n).tolist(),
                "dec": unpack_array(row.dec, n).tolist(),
                "alt": unpack_array(row.alt, n).tolist(),
                "az": unpack_array(row.az, n).tolist(),
                "airmass": unpack_array(row.airmass, n).tolist(),
                "hourangle": unpack_array(row.hourangle, n).tolist(),
            }
        
        return targets

async def get_calculator(session: AsyncSession) -> Calculator:
    """Factory function for FastAPI dependency injection."""
    return Calculator(session)
