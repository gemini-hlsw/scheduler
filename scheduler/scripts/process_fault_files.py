# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

# This file is likely obsolete.

from pathlib import Path
import logging
import re

from lucupy.minimodel import ALL_SITES

from definitions import ROOT_DIR
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__, level=logging.INFO)


def process_fault_file(input_file_path: Path, output_file_path: Path) -> None:
    """
    Process one of the fault files provided by science into a standardized fault file format with structure:
    FR-###### YYYY-MM-DD HH:MM:SS [Description]
    where the entries (with the exception of the date-time entry) are separated by tabs.
    """
    # Define a regular expression pattern to match lines starting with "FR" and capture the relevant data.
    _pattern = r'FR (\d+)\s+(\d{4}) (\d{2}) (\d{2}) (\d{2}:\d{2}:\d{2}) (?:\d{2}:\d{2})\s+(\d+\.\d+)\s+(.+)'

    _logger.info('+++ Beginning fault file processing +++')
    _logger.info(f'Input file: {input_file_path.relative_to(ROOT_DIR)}')
    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            total_lines = 0
            entry_lines = 0
            for line in input_file:
                total_lines += 1
                match = re.match(_pattern, line)
                if match:
                    entry_lines += 1
                    fr_number, year, month, day, time, value, description = match.groups()
                    description = description.strip().replace('\t', ' ')
                    formatted_dt = f'{year}-{month}-{day} {time}'
                    formatted_line = f'FR-{fr_number}\t{formatted_dt}\t{value}\t[{description}]\n'
                    output_file.write(formatted_line)
            _logger.info(f'Output file: {output_file_path.relative_to(ROOT_DIR)}')
            _logger.info(f'Total lines: {total_lines}, processed lines: {entry_lines}, '
                         f'Discarded lines: {total_lines - entry_lines}')
            _logger.info('+++ Processing successful. +++')
    except Exception as ex:
        _logger.error(f'Failure: {ex}')
        _logger.error('--- Processing failed. ---')


def process_eng_task_file(input_file_path: Path, output_file_path: Path) -> None:
    """
    Process one of the engineering task files provided by science into a standardized file format with structure:
    YYYY-MM-DD start-time end-time [description]
    where start-time and end-time can be:
     1. local time values at the given site; or
     2. twi, indicating evening twilight for start-time and morning twilight for end-time.
    """
    # Define a regular expression pattern to match lines starting with "FR" and capture the relevant data.
    _pattern = r'(\d{4}-\d{2}-\d{2})\s+(\d+:\d{2}|twi)\s+(\d+:\d{2}|twi)\s+(\[.*\])'

    _logger.info('+++ Beginning engineering task file processing +++')
    _logger.info(f'Input file: {input_file_path.relative_to(ROOT_DIR)}')

    try:
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            total_lines = 0
            entry_lines = 0
            for line in input_file:
                total_lines += 1
                match = re.match(_pattern, line)
                if match:
                    entry_lines += 1
                    date, start_time, end_time, description = match.groups()
                    if len(start_time) == 4:
                        start_time = f'0{start_time}'
                    if len(end_time) == 4:
                        end_time = f'0{end_time}'
                    description = description.replace('\t', ' ')
                    formatted_line = f'{date}\t{start_time}\t{end_time}\t{description}\n'
                    output_file.write(formatted_line)
            _logger.info(f'Output file: {output_file_path.relative_to(ROOT_DIR)}')
            _logger.info(f'Total lines: {total_lines}, processed lines: {entry_lines}, '
                         f'Discarded lines: {total_lines - entry_lines}')
            _logger.info('+++ Processing successful. +++')
    except Exception as ex:
        _logger.error(f'Failure: {ex}')
        _logger.error('--- Processing failed. ---')


def main():
    resource_path = Path(ROOT_DIR) / 'scheduler' / 'services' / 'resource' / 'data'
    fault_files = {f'Faults_All{site.name}.txt': f'{site.name}_faults.txt' for site in ALL_SITES}
    for input_file_path, output_file_path in fault_files.items():
        process_fault_file(resource_path / input_file_path, resource_path / output_file_path)

    eng_files = {f'ClosedDome{site.name}_Eng.txt': f'{site.name}_engtasks.txt' for site in ALL_SITES}
    for input_file_path, output_file_path in eng_files.items():
        process_eng_task_file(resource_path / input_file_path, resource_path / output_file_path)


if __name__ == "__main__":
    main()
