'''
Copyright 2024 Capgemini

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

from datetime import datetime, timezone
import time
from time import mktime
import git
import os
import re
from enum import Enum

verbose = False

repo_dir = "."

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

CARTOUCHE_CAP_AIRBUS = CARTOUCHE_BASE.format(
    f"{AIRBUS_COPYRIGHT}\n{CAP_MODIFIED_COPYRIGHT}\n{LICENCE}"
)
CARTOUCHE_CAP = CARTOUCHE_BASE.format(f"{CAP_COPYRIGHT}\n{LICENCE}")

# Define a regular expression to match the cartouche only at the beginning
cartouche_pattern = r"^'''(.*?)'''(\n|\Z)"

cartouche_modified_pattern = r"Modifications on (.+) Copyright 202(.) Capgemini"


class FileChange(Enum):
    """ 
    Enum class for the type of change detected used by HeaderError class
    """
    NOTSET = 0
    ADDED = 1
    MODIFIED = 2


class HeaderError:
    """ 
    Class that represents an header error for logging 
    """
    def __init__(
        self,
        concerned_file,
        type_of_change,
        error_details,
        expected_header,
        current_header="None",
    ):
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
    """
    Check if the header inside the file is matching the added file header requirement if not return an HeaderError

    :param file_path: path to the file where to check the header status
    :type file_path: str
    """

    with open(file_path, "r") as file:
        content = file.read()

    cartouche_match = re.search(
        pattern=cartouche_pattern, string=content, flags=re.DOTALL
    )

    if cartouche_match:
        if (
            cartouche_match.group(0).startswith(f"'''\n{CAP_COPYRIGHT}")
            or cartouche_match.group(0).startswith(f"'''\n{CAP_COPYRIGHT_2023}")
        ) and cartouche_match.group(0).__contains__(LICENCE):
            # OK
            if verbose:
                print(f"Cartouche OK for path {file_path}")
        else:
            # Unexpected cartouche
            if verbose:
                print(
                    f"ERROR Unexpected cartouche for path {file_path}", cartouche_match
                )
            return HeaderError(
                file_path,
                FileChange.ADDED,
                "Unexpected header",
                CARTOUCHE_CAP,
                cartouche_match.group(0),
            )

    else:
        # No cartouche
        if verbose:
            print(f"No Cartouche Error for path {file_path}")
        return HeaderError(file_path, FileChange.ADDED, "No header", CARTOUCHE_CAP)


def check_header_for_modified_file(file_path) -> HeaderError:
    """
    Check if the header inside the file is matching the modified file header requirement if not return an HeaderError

    :param file_path: path to the file where to check the header status
    :type file_path: str
    """

    with open(file_path, "r") as file:
        content = file.read()

    cartouche_match = re.search(
        pattern=cartouche_pattern, string=content, flags=re.DOTALL
    )
    if cartouche_match:
        cartouche_modified_match = re.search(
            pattern=cartouche_modified_pattern,
            string=str(cartouche_match.group(0)),
            flags=re.DOTALL,
        )

        if (
            cartouche_match.group(0).startswith(f"'''\n{AIRBUS_COPYRIGHT}")
            and cartouche_match.group(0).__contains__(LICENCE)
            and cartouche_modified_match
        ):
            # OK
            if verbose:
                print(f"Cartouche OK for path {file_path}")
        else:
            # Unexpected cartouche
            if verbose:
                print(
                    f"ERROR Unexpected cartouche for path {file_path}", cartouche_match
                )
            return HeaderError(
                file_path,
                FileChange.MODIFIED,
                "Unexpected header",
                CARTOUCHE_CAP_AIRBUS,
                cartouche_match.group(0),
            )

    else:
        # No cartouche, add it
        if verbose:
            print(f"No Cartouche Error for path {file_path}")
        return HeaderError(
            file_path, FileChange.MODIFIED, "No header", CARTOUCHE_CAP_AIRBUS
        )


def get_first_commit_time(full_file_path: str) -> datetime:
    """
    Check if the header inside the file is matching the added file header requirement if not return an HeaderError

    :param file_path: path to the file where to check the header status
    :type file_path: str
    """
    g = git.Git(repo_dir)

    logs = g.log("--follow",full_file_path)
    #logs = g.log(full_file_path)

    # isolate : "Date:   Wed Nov 22 11:08:54 2023" like  note that it is the time string by default easy to convert
    # Parse logs Getting Date keep the last one that is the older
    pattern = r"Date:   [A-Z][a-z]{2} [A-Z][a-z]{2} [0-9]{1,2} [0-9]{1,2}:[0-9]{2}:[0-9]{2} [q0-9]{4}"
    dates_list = re.compile(pattern).findall(logs)

    # print(fullFilePath + " : " + str(dates_list))
    date_str = re.sub("Date:   ", "", dates_list[len(dates_list) - 1].strip())

    d = time.strptime(date_str)

    return datetime.fromtimestamp(mktime(d), tz=timezone.utc)


def has_been_commited_from_airbus(full_file_path: str, refcommit: git.Commit) -> bool:
    # True if committed from Airbus
    return refcommit.committed_datetime > get_first_commit_time(full_file_path)


def check_headers(ignored_exts, ignored_file, airbus_rev_commit: str):
    header_error_list = []

    # Initialize a Git Repo object wity current directory
    repo = git.Repo(repo_dir)

    # fetch all
    for remote in repo.remotes:
        remote.fetch()

    # diff between airbus version  and current checkout version (dev) -> all modifications and addition between Airbus version published on OS-C and the newly published
    commit_dev = repo.head.commit
    commit_airbus = repo.commit(airbus_rev_commit)

    print("Airbus date : " + str(commit_airbus.committed_datetime))

    diff_index = commit_airbus.diff(commit_dev)

    for diff_item in diff_index.iter_change_type("A"):
        # Added
        item_path = diff_item.a_path

        if item_path not in ignored_file:
            if verbose:
                print("A", item_path)

            file_path = os.path.join(repo_dir, item_path)

            # Need to add Cap Header for python file
            if item_path.endswith(".py"):
                error = check_header_for_added_file(file_path)
                #if has_been_commited_from_airbus(file_path, commit_airbus):
                    # print(
                    #     "ERROR detected has added but committed by Airbus : "
                    #     + file_path
                    # )
                    #error = check_header_for_modified_file(file_path)
                if error:
                    header_error_list.append(error)

            elif item_path.split(".")[-1].lower() in ignored_exts:
                # Do nothing for pkl, markdown, csv, ...
                pass
            else:
                # Need to check not handled file too
                error = check_header_for_added_file(file_path)
                #if has_been_commited_from_airbus(file_path, commit_airbus):
                #    error = check_header_for_modified_file(file_path)
                if error:
                    header_error_list.append(error)

                print("UNHANDLED", file_path)

        else:
            if verbose:
                print("Ignored added file", item_path)

    for diff_item in diff_index.iter_change_type("D"):
        # Deleted
        item_path = diff_item.a_path
        if item_path not in ignored_file:
            if verbose:
                print("D", item_path)

    for diff_item in diff_index.iter_change_type("R"):
        # Renamed
        item_path = diff_item.a_path
        item_path_r = diff_item.b_path

        if item_path_r not in ignored_file:
            if verbose:
                print("R", item_path, "->", item_path_r)

    for diff_item in diff_index.iter_change_type("M"):
        # Modified
        item_path = diff_item.b_path

        if item_path not in ignored_file:
            if verbose:
                print("M", item_path)

            file_path = os.path.join(repo_dir, item_path)

            # Need to add Cap Modified Header section for python file
            if item_path.endswith(".py"):
                error = check_header_for_modified_file(file_path)
                if not has_been_commited_from_airbus(file_path, commit_airbus):
                    error = check_header_for_added_file(
                        file_path
                    )  # if not commited from Airbus then it is an added file detected by error by diff has a modified file
                if error:
                    header_error_list.append(error)

            elif item_path.split(".")[-1].lower() in ignored_exts:
                # Do nothing for pkl, markdown, csv, ...
                pass
            elif (
                item_path.endswith(".ts")
                or item_path.endswith(".html")
                or item_path.endswith(".scss")
            ):
                # Skip modified ts
                pass
            else:
                # Need to check not handled file too
                error = check_header_for_modified_file(file_path)
                if not has_been_commited_from_airbus(file_path, commit_airbus):
                    error = check_header_for_added_file(
                        file_path
                    )  # if not commited from Airbus then it is an added file detected by error by diff has a modified file
                if error:
                    header_error_list.append(error)
                print("UNHANDLED", file_path)
        else:
            if verbose:
                print("Ignored modified file", item_path)

    for diff_item in diff_index.iter_change_type("T"):
        # Changed in the type path
        item_path = diff_item.a_path

        if item_path not in ignored_file:
            if verbose:
                print("T", item_path)

    if len(header_error_list) > 0:
        errors = ""
        for he in header_error_list:
            errors += str(he)
        errors += f"\nFound {len(header_error_list)} header error(s)"
        raise Exception(f"{errors}")
