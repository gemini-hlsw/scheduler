# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import datetime
import re
from pathlib import Path

from definitions import ROOT_DIR

year = datetime.datetime.now().year
year_range_pattern = r'Copyright \(c\) (\d{4})-(\d{4})'
new_copyright = f'# Copyright (c) 2016-{year} Association of Universities for Research in Astronomy, Inc. (AURA)\n'
second_line = '# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause\n\n'


def update_file(file_path):
    updated_lines = []
    found_copyright = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        match = re.search(year_range_pattern, line)
        if match:
            found_copyright = True
            old_year = int(match.group(2))
            if old_year < year:
                print(f'Updating copyright year in file {file_path} from {old_year} to {year}')
                updated_line = re.sub(year_range_pattern, f'Copyright (c) 2016-{year}', line)
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    if not found_copyright:
        print(f'Adding copyright to file: {file_path}')
        updated_lines.insert(0, new_copyright)
        updated_lines.insert(1, second_line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)


def update_copyright(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                update_file(os.path.join(root, filename))


if __name__ == '__main__':
    update_copyright(Path(ROOT_DIR) / 'scheduler')
    update_file(Path(ROOT_DIR) / 'definitions.py')
