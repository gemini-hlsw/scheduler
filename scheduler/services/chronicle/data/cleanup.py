# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime


def main():
    new_lines = []

    with open('EngTasksGS.txt', 'r') as f:
        old_lines = f.readlines()

    for line in old_lines:
        lines = line.split('\t')
        print(lines)
        if lines[1].strip() == '0' or lines[1].strip() == '0.0':
            continue
        date = lines[0]
        first_date_str = date.split('-')[0]
        date_obj = datetime.strptime(first_date_str, "%Y %b%d")
        formatted_date_str = date_obj.strftime("%Y-%m-%-d")

        start, end = lines[2].split('-')
        msg = lines[4].strip() if len(lines) == 5 else lines[3].strip()
        new_lines.append(f'{formatted_date_str}\t{start}\t{end}\t[{msg}]\n')

    # Write the file out again
    with open('EngTasksGS.txt', 'w') as file:
        file.writelines(new_lines)


if __name__ == '__main__':
    main()
