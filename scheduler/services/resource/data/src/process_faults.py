import re
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from definitions import ROOT_DIR

from lucupy.minimodel import Site

# NOTE: This processing changes the fault dates to the DATE NIGHTS of when the faults occurred.
# NOTE: We don't deal with faults yet because of the GS faults issue as per email discussion.
pattern = r'FR-(\d+)\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([\d.]+)\s+\[([^\]]+)\]'

processed = 0
file = Path(ROOT_DIR) / 'scheduler' / 'services' / 'resource' / 'data' / 'GN_Faults.txt'
site = Site.GN
with file.open() as f:
    for line in f:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        match = re.match(pattern, line)
        if match is None:
            print(f'ERROR: could not parse: {line}')
            sys.exit(0)

        fr_id, local_datetime_str, duration_str, description = match.groups()
        local_datetime = datetime.strptime(local_datetime_str, '%Y-%m-%d %H:%M:%S')
        local_datetime = local_datetime.replace(tzinfo=site.timezone)
        duration = timedelta(hours=float(duration_str))

        # Determine the night of the fault report from the local datetime.
        # If it is before noon, it belongs to the previous night.
        if local_datetime.time() < time(hour=12):
            night_date = local_datetime.date() - timedelta(days=1)
        else:
            night_date = local_datetime.date()
        end_datetime = local_datetime + duration
        print(f'{night_date.strftime("%Y-%m-%d")}\t{local_datetime.strftime("%H:%M")}\t'
              f'{end_datetime.strftime("%H:%M")}\t[FR-{fr_id}: {description}]')
        processed += 1

print(f'Processed {processed} entries.')
