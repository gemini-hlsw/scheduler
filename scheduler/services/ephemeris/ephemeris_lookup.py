# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import io
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import final
from urllib.parse import urlparse

import aioboto3
from astropy.coordinates import SkyCoord
from botocore.exceptions import BotoCoreError, ClientError
from lucupy.minimodel import Site
from lucupy.meta import Singleton

from scheduler.services.logger_factory import create_logger

__all__ = [
    'EphemerisLookup',
]

logger = create_logger(__name__)


@final
class EphemerisLookup(metaclass=Singleton):
    """
    In-memory lookup table for pre-computed ephemeris coordinates.

    Data is loaded from a pickle (either from S3 or a local file) with structure:
        dict[str, dict[str, dict[datetime, SkyCoord]]]
             target_name -> site_name -> datetime -> SkyCoord
    """

    def __init__(self) -> None:
        self._table: dict[str, dict[str, dict[datetime, SkyCoord]]] = {}

    async def load_from_s3(self, site: Site, start: datetime, end: datetime) -> None:
        """Download a pickle from S3 (CloudCube) and populate the lookup table.

        The S3 key is derived using the same convention as
        ``fetch_ephemerides.pickle_and_upload_to_s3``:
            <key_prefix><site>_<start>_<end>.pkl

        If the file is not found or the download fails, a warning is logged
        and the table is left unchanged.
        """
        cube_url = urlparse(os.environ["CLOUDCUBE_URL"])
        bucket = cube_url.netloc.split(".")[0]
        key_prefix = cube_url.path.lstrip("/") + "/ephemerides/"

        site_name = site.name if hasattr(site, "name") else str(site)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        key = f"{key_prefix}{site_name}_{start_str}_{end_str}.pkl"

        session = aioboto3.Session(
            aws_access_key_id=os.environ["CLOUDCUBE_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["CLOUDCUBE_SECRET_ACCESS_KEY"],
        )

        try:
            async with session.client("s3") as s3:
                buffer = io.BytesIO()
                await s3.download_fileobj(bucket, key, buffer)
                buffer.seek(0)
        except (BotoCoreError, ClientError) as exc:
            logger.warning(f"Could not download ephemerides from s3://{bucket}/{key}: {exc}")
            return

        data: dict[str, dict[str, dict[datetime, SkyCoord]]] = pickle.load(buffer)
        self._merge(data)
        logger.info(f"Loaded {len(data)} target(s) from s3://{bucket}/{key}")

    async def load_from_file(self, path: Path | str) -> None:
        """Load a local pickle file and populate the lookup table.

        If the file does not exist, a warning is logged and the table is
        left unchanged.
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Pickle file not found: {path}")
            return
        with path.open("rb") as f:
            data: dict[str, dict[str, dict[datetime, SkyCoord]]] = pickle.load(f)
        self._merge(data)
        logger.info(f"Loaded {len(data)} target(s) from {path}")

    def lookup(self, target_name: str, site_name: str, dt: datetime) -> SkyCoord:
        """Return the SkyCoord for a given target, site, and datetime.

        Raises:
            KeyError: if the combination is not found in the table.
        """
        try:
            return self._table[target_name][site_name][dt]
        except KeyError:
            raise KeyError(
                f"No ephemeris found for target='{target_name}', "
                f"site='{site_name}', datetime={dt}"
            )

    @property
    def targets(self) -> list[str]:
        """Return the list of target names currently loaded."""
        return list(self._table.keys())

    def _merge(self, data: dict[str, dict[str, dict[datetime, SkyCoord]]]) -> None:
        """Merge *data* into the existing table (additive)."""
        for target, sites in data.items():
            for site, times in sites.items():
                self._table.setdefault(target, {}).setdefault(site, {}).update(times)
