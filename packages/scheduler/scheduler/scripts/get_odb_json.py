#!/usr/bin/env python3
# coding: utf-8

# Code for downloading json files from the ODB
# Bryan Miller
# 2025-12-23 updated

# Example for extracting json files for the scheduler
# python get_odb_json.py -s sched -p ./json/20251223  @~/python/scheduler/scheduler/data/program_ids.redis.txt
# cd ./json/20251223
# zip -X programs.zip *.json

import argparse
from odb_json import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("programid", help="Program IDs, comma separated or prefix with '@' for list file")
    parser.add_argument("-p","--path", help="Path for json files", default="None")
    parser.add_argument("-s","--server", help="Server", type=str, choices=["gs", "gn", "sched"])
    parser.add_argument("-t", "--test", help="Test server?", action="store_true", default=False)
    parser.add_argument("-g", "--gz", help="Gzip output?", action="store_true", default=False)
    parser.add_argument("-o", "--overwrite", help="Overwrite?", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", help="Verbose output?", action="store_true", default=False)
    args = parser.parse_args()

    progids = []
    if (args.programid[0] == "@") and (os.path.isfile(args.programid[1:])):
        l_file = open(args.programid[1:])
        for line in l_file:
            line = line.rstrip("\n")
            if line[0] != "#":
                progids.append(line.strip(' '))
        l_file.close()
    else:
        progids = args.programid.replace(" ","").split(",")

    # print(progids)
    for progid in progids:
        program = odb_json(progid, server=args.server, path=args.path, test=args.test, overwrite=args.overwrite,
                           gz=args.gz, verbose=args.verbose)
