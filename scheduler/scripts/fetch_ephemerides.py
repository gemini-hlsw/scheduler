"""
Script to download a series of visibility calculations
"""
import asyncio
import os
import pickle
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import aioboto3
from astropy.coordinates import SkyCoord
from astropy.time import Time
from botocore.exceptions import BotoCoreError, ClientError
from gpp_client.api import WhereTarget, WhereOrderTargetId, WhereOrderProgramId, WhereProgram
from lucupy.minimodel import Site, NonsiderealTarget, TargetType, TargetTag, TargetName

from scheduler.services.ephemeris import EphemerisCalculator

from scheduler.clients.scheduler_gpp_client import gpp_client_instance
from threading import Lock

from urllib.parse import urlparse

# Parse CloudCube env vars (set automatically by Heroku add-on)
_cube_url = urlparse(os.environ["CLOUDCUBE_URL"])
S3_BUCKET = _cube_url.netloc.split(".")[0]
S3_KEY_PREFIX = _cube_url.path.lstrip("/") + "/ephemerides/"

eph_calculator = EphemerisCalculator()

target_coords: dict[str, dict[str, dict[datetime, SkyCoord]]] = {}
_coords_lock = Lock()

def fetch_ephemerides(
    site: Site, target: NonsiderealTarget, start: datetime, end: datetime
) -> tuple[str, str, datetime, SkyCoord]:
    """Fetch ephemerides for a single target and return (target_name, site_name, start, coord)."""
    start_time = Time(start.isoformat())
    end_time = Time(end.isoformat())

    sky_coord = eph_calculator.calculate_coordinates(
        site, target, start_time, end_time
    )
    site_name = site.name if hasattr(site, "name") else str(site)
    return target.name, site_name, start, sky_coord


def fetch_batch_ephemerides(
    site: Site,
    targets: list[NonsiderealTarget],
    start: datetime,
    end: datetime,
    batch_size: int = 5,
    max_workers: int = 4,
) -> dict[str, dict[str, dict[datetime, SkyCoord]]]:
    """
    Fetch ephemerides for a list of targets, batched by target count.

    Args:
        site:        Observing site.
        targets:     Full list of nonsidereal targets to process.
        start:       Start datetime for ephemeris range.
        end:         End datetime for ephemeris range.
        batch_size:  Number of targets to submit per worker batch.
        max_workers: Maximum number of concurrent threads.

    Returns:
        Snapshot of target_coords populated during this run.
    """
    # Split targets into batches of `batch_size`
    batches: list[list[NonsiderealTarget]] = [
        targets[i : i + batch_size] for i in range(0, len(targets), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for batch in batches:
            for target in batch:
                future = executor.submit(fetch_ephemerides, site, target, start, end)
                futures[future] = target.name

        for future in as_completed(futures):
            target_name = futures[future]
            try:
                t_name, site_name, dt, sky_coord = future.result()
                with _coords_lock:
                    target_coords \
                        .setdefault(t_name, {}) \
                        .setdefault(site_name, {})[dt] = sky_coord
            except Exception as exc:
                print(f"[ERROR] Failed to fetch ephemerides for '{target_name}': {exc}")

    with _coords_lock:
        return dict(target_coords)


async def pickle_and_upload_to_s3(
    coords: dict[str, dict[str, dict[datetime, SkyCoord]]],
    site: Site,
    start: datetime,
    end: datetime,
    bucket: str = S3_BUCKET,
    key_prefix: str = S3_KEY_PREFIX,
) -> str:
    """
    Serialize `coords` to a pickle byte stream and upload it to S3.

    The S3 key is auto-generated from site + date range:
        <key_prefix><site>_<start>_<end>.pkl
    e.g. ephemerides/GS_20260301_20260315.pkl

    Args:
        coords:     Dict of target name -> SkyCoord to persist.
        site:       Observing site (used in the key name).
        start:      Start datetime of the ephemeris range.
        end:        End datetime of the ephemeris range.
        bucket:     S3 bucket name.
        key_prefix: Key prefix / folder inside the bucket.

    Returns:
        The full S3 URI of the uploaded object (s3://<bucket>/<key>).

    Raises:
        RuntimeError: If the upload fails.
    """
    site_name = site.name if hasattr(site, "name") else str(site)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    key = f"{key_prefix}{site_name}_{start_str}_{end_str}.pkl"

    # Serialize to an in-memory buffer — no temp files needed.
    buffer = io.BytesIO()
    pickle.dump(coords, buffer)
    buffer.seek(0)

    session = aioboto3.Session(
        aws_access_key_id=os.environ["CLOUDCUBE_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CLOUDCUBE_SECRET_ACCESS_KEY"],
    )

    try:
        async with session.client("s3") as s3:
            await s3.upload_fileobj(buffer, bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"Failed to upload ephemerides to s3://{bucket}/{key}: {exc}"
        ) from exc

    s3_uri = f"s3://{bucket}/{key}"
    print(f"[S3] Uploaded {len(coords)} target(s) -> {s3_uri}")
    return s3_uri

async def get_nonsidereal_targets():

    client = gpp_client_instance.client
    response = await client._client.get_scheduler_all_programs_id()
    program_ids = [p.id for p in response.programs.matches]

    response = await client.target.get_all(
        where=WhereTarget(program=WhereProgram(id=WhereOrderProgramId(in_=program_ids)))
    )
    return [
        NonsiderealTarget(
            name=t["name"],
            type=t["nonsidereal"]["type"],
            des=t["nonsidereal"]["des"],
            tag=t["nonsidereal"]["keyType"],
            magnitudes=frozenset()
        )
        for t in response["matches"] if t["nonsidereal"]
    ]

async def main():

    # Params to run

    site = Site.GS
    start = datetime(2026, 3, 1)
    end = datetime(2026, 3, 2)

    #targets = await get_nonsidereal_targets()
    #print(f"Found {len(targets)} nonsidereal targets.")

    targets = [ NonsiderealTarget(
        name=TargetName("Beer"),
        des="1971 UC1",
        type=TargetType.BASE,
        magnitudes=frozenset(),
        tag=TargetTag.ASTEROID,
    )]

    results = fetch_batch_ephemerides(
        site=site,
        targets=targets,
        start=start,
        end=end,
        batch_size=5,
        max_workers=4,
    )

    print(f"Stored ephemerides for {len(results)} targets.")
    for t_name, sites in results.items():
        for site_name, datetimes in sites.items():
            print(f"  {t_name} @ {site_name}: {len(datetimes)} datetime(s)")

    await pickle_and_upload_to_s3(results, site, start, end)


if __name__ == "__main__":

    asyncio.run(main())
