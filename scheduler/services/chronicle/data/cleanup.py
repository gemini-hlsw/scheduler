from datetime import datetime

new_lines = []

with open('WLGS2.txt', 'r') as f:
    old_lines = f.readlines()

for line in old_lines:
    lines = line.split('\t')
    print(lines)
    if lines[1].strip() == '0' or lines[1].strip() == '0.0':
        continue
    date = lines[0]
    first_date_str, second_date_str = date.split('-')

    date_obj_first = datetime.strptime(first_date_str, "%Y %b%d")
    formatted_date_str = date_obj_first.strftime("%Y-%m-%-d")

    start, end = lines[2].split('-')
    # msg = lines[4].strip() if len(lines) == 5 else lines[3].strip()
    new_lines.append(f'{formatted_date_str}\t{start}\t{end}\n')

# Write the file out again
with open('WLGS2.txt', 'w') as file:
    file.writelines(new_lines)