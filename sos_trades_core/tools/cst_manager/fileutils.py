'''
Copyright 2022 Airbus SAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import json
import os
import sys 
from pathlib2 import Path

def list_path(path, check):
    selection = [(name, os.path.join(path, name)) for name in os.listdir(path) if check(os.path.join(path, name))]
    return selection

def readjson(file_fullpath):
    # Load input file into data dictionary
    try:
        with open(file_fullpath, 'r') as fid:
            return json.load(fid)
    except Exception as error:
        print('ERROR: Cannot decode filename=', str(file_fullpath))
        print(error)
        sys.exit(1)

def writejson(file_fullpath, json_data):
    with open(file_fullpath, 'w') as fid:
        return json.dump(json_data, fid, indent=4)

def from_json_or_file(data, fixpath):
    if type(data) == str or type(data) == str:
        fixed_path = fixpath(data)
        return readjson(fixed_path)
    else:
        return data

def relative_path(folder_name, path):
    if not os.path.isfile(path):
        fixedpath = os.path.join(folder_name, path)
        # navigate one level up the tree to see if it can be found there instead
        if not os.path.isfile(fixedpath):
            fixedpath = os.path.join(os.path.dirname(os.path.abspath(folder_name)), path)
    else:
        fixedpath = path

    return fixedpath

def get_relative_path(target, origin):
    """
    Return a Path object containing the path from origin to target. This function is limited and can handle only simple case (target and origin have a common part). 
    """
    if isinstance(target, Path):
        a = target
    else:
        a = Path(target)
    if isinstance(origin, Path):
        b = origin
    else:
        b = Path(origin)
    
    for pa in a.parents:
        for pb in b.parents:
            if pa == pb: 
                break
        if pa == pb: 
            break
    common_path = pa
    rb =b.relative_to(common_path)
    ra = a.relative_to(common_path)
    b_to_a = Path( '../' * len(rb.parents)) / ra
    return b_to_a
