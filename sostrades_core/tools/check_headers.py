'''
Copyright 2023 Capgemini

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

import git
import os
import re
from enum import Enum

verbose = False

LICENCE = """Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

CARTOUCHE_BASE = """'''
{}
'''
"""

CAP_COPYRIGHT_2023 = "Copyright 2023 Capgemini"
CAP_COPYRIGHT = "Copyright 2024 Capgemini"
CAP_MODIFIED_COPYRIGHT = "Modifications on {} Copyright 2024 Capgemini"
AIRBUS_COPYRIGHT = "Copyright 2022 Airbus SAS"

CARTOUCHE_CAP_AIRBUS = CARTOUCHE_BASE.format(f"{AIRBUS_COPYRIGHT}\n{CAP_MODIFIED_COPYRIGHT}\n{LICENCE}")
CARTOUCHE_CAP = CARTOUCHE_BASE.format(f"{CAP_COPYRIGHT}\n{LICENCE}")

# Define a regular expression to match the cartouche only at the beginning
cartouche_pattern = r"^'''(.*?)'''(\n|\Z)"

cartouche_modified_pattern = r"Modifications on (.+) Copyright 202(.) Capgemini"

class FileChange(Enum):
    NOTSET = 0
    ADDED = 1
    MODIFIED = 2

class HeaderError:

    def __init__(self, concerned_file, type_of_change,error_details,expected_header,current_header = "None"):
        self.concerned_file = concerned_file
        self.type_of_change = type_of_change
        self.expected_header = expected_header
        self.current_header = current_header
        self.error_details = error_details

    def __str__(self):
     return f"-------------------\n\
Header Error on {self.type_of_change} file : {self.concerned_file}\n\
Details : {self.error_details}\n\
Expected header is :\n{ self.expected_header}\n\
but header is\n{self.current_header}\n\
-------------------\n"

def check_header_for_added_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)

    if cartouche_match:
        if (cartouche_match.group(0).startswith(f"'''\n{CAP_COPYRIGHT}") or cartouche_match.group(0).startswith(f"'''\n{CAP_COPYRIGHT_2023}")) and cartouche_match.group(0).__contains__(LICENCE):
            #OK
            if verbose :
                print(f"Cartouche OK for path {file_path}")
        else:
            # Unexpected cartouche
            if verbose :
                print(f"ERROR Unexpected cartouche for path {file_path}", cartouche_match)
            return HeaderError(file_path,FileChange.ADDED, "Unexpected header",CARTOUCHE_CAP,cartouche_match.group(0))
            
    else:
        # No cartouche
        if verbose :
            print(f"No Cartouche Error for path {file_path}")
        return HeaderError(file_path,FileChange.ADDED, "No header",CARTOUCHE_CAP)


def check_header_for_modified_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)
    if cartouche_match:
        cartouche_modified_match = re.search(pattern=cartouche_modified_pattern, string=str(cartouche_match.group(0)), flags=re.DOTALL)


        if cartouche_match.group(0).startswith(f"'''\n{AIRBUS_COPYRIGHT}") and cartouche_match.group(0).__contains__(LICENCE) and cartouche_modified_match :
            #OK
            if verbose :
               print(f"Cartouche OK for path {file_path}")   
        else:
            # Unexpected cartouche
            if verbose :
                print(f"ERROR Unexpected cartouche for path {file_path}", cartouche_match)
            return HeaderError(file_path,FileChange.MODIFIED, "Unexpected header",CARTOUCHE_CAP_AIRBUS,cartouche_match.group(0))

    else:
        # No cartouche, add it
        if verbose :
            print(f"No Cartouche Error for path {file_path}")
        return HeaderError(file_path,FileChange.MODIFIED, "No header",CARTOUCHE_CAP_AIRBUS)


def check_headers(ignored_exts,ignored_file,airbus_rev_commit):
    
    HeaderErrorList = []

    repo_dir = "."
    
    # Initialize a Git Repo object wity current directory
    repo = git.Repo(repo_dir)

    # fetch all
    for remote in repo.remotes:
        remote.fetch()
    
    #diff between airbus version  and current checkout version (dev) -> all modifications and addition between Airbus version published on OS-C and the newly published
    commit_dev = repo.head.commit
    commit_airbus = repo.commit(airbus_rev_commit)
    
    diff_index = commit_airbus.diff(commit_dev)

    for diff_item in diff_index.iter_change_type('A'):
        # Added
        item_path = diff_item.a_path
     
        if item_path not in ignored_file:
            if verbose:
                print("A", item_path)

            file_path = os.path.join(repo_dir, item_path)

            # Need to add Cap Header for python file
            if item_path.endswith(".py"):
                error = check_header_for_added_file(file_path)
                if error:
                    HeaderErrorList.append(error)
                
            elif item_path.split(".")[-1].lower() in ignored_exts:
                # Do nothing for pkl, markdown, csv, ...
                pass
            else:
                # Need to check not handled file too
                error = check_header_for_added_file(file_path)
                if error:
                    HeaderErrorList.append(error)
                
                print("UNHANDLED", file_path)


    for diff_item in diff_index.iter_change_type('D'):
        # Deleted
        item_path = diff_item.a_path
        if item_path not in ignored_file:
            if verbose:
                print("D", item_path)    

    for diff_item in diff_index.iter_change_type('R'):
        # Renamed
        item_path = diff_item.a_path
        item_path_r = diff_item.b_path

        if item_path_r not in ignored_file:
            if verbose:
                print("R", item_path, '->', item_path_r)

    for diff_item in diff_index.iter_change_type('M'):
        # Modified
        item_path = diff_item.b_path

        if item_path not in ignored_file:
            if verbose:
                print("M", item_path)

            file_path = os.path.join(repo_dir, item_path)

            # Need to add Cap Modified Header section for python file
            if item_path.endswith(".py"):
                error = check_header_for_modified_file(file_path)
                if error:
                    HeaderErrorList.append(error)

            elif item_path.split(".")[-1].lower() in ignored_exts:
                # Do nothing for pkl, markdown, csv, ...
                pass
            elif item_path.endswith(".ts") or item_path.endswith(".html") or item_path.endswith(".scss"):
                # Skip modified ts
                pass
            else:
                # Need to check not handled file too
                error = check_header_for_modified_file(file_path)
                if error:
                    HeaderErrorList.append(error)
                print("UNHANDLED", file_path)
        
    for diff_item in diff_index.iter_change_type('T'):
        # Changed in the type path
        item_path = diff_item.a_path

        if item_path not in ignored_file:
        
            if verbose:
                print("T", item_path)

    if len(HeaderErrorList) > 0:
        errors = ""
        for he in HeaderErrorList:
            errors += str(he)
        errors +=f"\nFound {len(HeaderErrorList)} header error(s)"
        raise Exception(f'{errors}')
        
               