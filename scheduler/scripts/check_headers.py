# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from pathlib import Path

from definitions import ROOT_DIR

# Checks the headers of all .py files to make sure that they contain the above Copyright message.


HEADER = '# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)\n# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause'

def pre_adder(file):
    with open(file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(HEADER + '\n' + content)


if __name__ == '__main__':
    COPYRIGHT = '# Copyright'

    bad_files = []
    for path in Path(ROOT_DIR).rglob('*.py'):
        with open(path, 'r') as f:
            if not f.readline().startswith(COPYRIGHT):
                bad_files.append(path)

    if bad_files:
        print('*** FILES MISSING COPYRIGHT ***')
        for bad_file in bad_files:
            print(bad_file)
            pre_adder(bad_file)
        exit(1)
    else:
        print('All files copyrighted.')
        exit(0)
