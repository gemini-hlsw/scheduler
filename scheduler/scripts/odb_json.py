#!/usr/bin/env python

import os
import json
import gzip
import requests


def odb_json(progid, server='sched', path='None', overwrite=False, test=False, gz=False, verbose=False):
    """
    Download json of ODB program information

    Parameters
        progid:  Program ID of program to extract
        server: one of sched, gs, gn
        path:    Path for json files
        overwrite: Overwrite any existing json files?
        test: use test ODB server
        gz: gzip the resulting json disk files
        verbose: Verbose output?

    Return
        json_result:   JSON query result as a dictionary
    """

    json_result = None

    if progid == "":
        print('odb_json: program id not given.')
        raise ValueError('Program id not given.')

    teststr = 'test' if test else ''

    server_url = ''
    match server:
        case 'sched':
            server_url = 'http://gnodbscheduler.hi.gemini.edu'
        case 'gs':
            server_url = 'http://gsodb' + teststr + '.cl.gemini.edu'
        case 'gn':
            server_url = 'http://gnodb' + teststr + '.hi.gemini.edu'
        case _:
            print('Server not supported, must be "sched", "gs", or "gn".')
            return json_result

    file = progid + '.json'
    if gz:
        file += '.gz'
    if not overwrite and os.path.exists(os.path.join(path, file)):
        if gz:
            with gzip.open(os.path.join(path, file), 'r') as fin:
                json_bytes = fin.read()
            json_str = json_bytes.decode('utf-8')
        else:
            with open(os.path.join(path, file), 'r') as fin:
                json_str = fin.read()

        if verbose:
            print(f'Reading {file}')
        json_result = json.loads(json_str)
    else:
        if verbose:
            print(f'Querying {progid} from {server_url}')
        response = requests.get(server_url + ':8442/programexport?id=' + progid)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            print('odb_json: request failed: {}'.format(response.text))
            raise exc
        else:
            json_result = response.json()
            if (overwrite or not os.path.exists(os.path.join(path, file))) and path != 'None':
                json_str = json.dumps(json_result, indent=2)
                if gz:
                    with gzip.open(os.path.join(path, file), 'wb') as fout:
                        fout.write(json_str.encode('utf-8'))
                else:
                    with open(os.path.join(path, file), 'w') as fout:
                        fout.write(json_str)

        if verbose:
            print(f'\t {response.url}')
        # print(response.text)

    return json_result
