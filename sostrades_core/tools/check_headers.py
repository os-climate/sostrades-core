'''
Copyright 2025 Capgemini

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
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from enum import Enum
from time import mktime

import git

VERBOSE = False

BRANCH = "develop"

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
{}'''
"""

CAP_COPYRIGHT = "Copyright {} Capgemini".format(datetime.today().strftime("%Y/%m/%d"))
CAP_MODIFIED_COPYRIGHT = "Modifications on {} " + "Copyright {} Capgemini".format(datetime.today().strftime("%Y"))
AIRBUS_COPYRIGHT = "Copyright 2022 Airbus SAS"

CARTOUCHE_CAP_AIRBUS = CARTOUCHE_BASE.format(f"{AIRBUS_COPYRIGHT}\n{CAP_MODIFIED_COPYRIGHT}\n\n{LICENCE}")
CARTOUCHE_CAP = CARTOUCHE_BASE.format(f"{CAP_COPYRIGHT}\n\n{LICENCE}")

# Define a regular expression to match the cartouche only at the beginning
cartouche_pattern = r"^'''(.*?)'''(\n|\Z)"
# cartouche_pattern = r"^(?:'''|\"\"\")(.*?)(?:'''|\"\"\")(\n|\Z)"
cap_copyright_pattern = "Copyright 202(.) Capgemini"
cartouche_modified_pattern = r"Modifications on (.+) Copyright 202(.) Capgemini"


class FileChange(Enum):
    """Enum class for the type of change detected used by HeaderError class"""

    NOTSET = 0
    ADDED = 1
    MODIFIED = 2


class HeaderError:
    """Class that represents an header error for logging"""

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
Expected header is :\n{self.expected_header}\n\
but header is\n{self.current_header}\n\
-------------------\n"


class HeaderTools:
    def __init__(self):
        self.verbose = VERBOSE

    def set_verbose_mode(self, val):
        self.verbose = val

    def check_header_for_added_file(self, file_path):
        """
        Check if the header inside the file is matching the added file header requirement if not return an HeaderError

        :param file_path: path to the file where to check the header status
        :type file_path: str
        """
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)

        if cartouche_match:
            first_line = cartouche_match.group(0).split("\n")[1]
            if re.search(pattern=cap_copyright_pattern, string=first_line) and LICENCE in cartouche_match.group(0):
                # OK
                if self.verbose:
                    return None
                return None
            # Unexpected cartouche
            if self.verbose:
                pass
            return HeaderError(
                file_path,
                FileChange.ADDED,
                "Unexpected header",
                CARTOUCHE_CAP,
                cartouche_match.group(0),
            )

        # No cartouche
        if self.verbose:
            pass
        return HeaderError(file_path, FileChange.ADDED, "No header", CARTOUCHE_CAP)

    def check_header_for_modified_file(self, file_path) -> HeaderError:
        """
        Check if the header inside the file is matching the modified file header requirement if not return an HeaderError

        :param file_path: path to the file where to check the header status
        :type file_path: str
        """
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)
        if cartouche_match:
            cartouche_modified_match = re.search(
                pattern=cartouche_modified_pattern,
                string=str(cartouche_match.group(0)),
                flags=re.DOTALL,
            )

            if (
                cartouche_match.group(0).startswith(f"'''\n{AIRBUS_COPYRIGHT}")
                and LICENCE in cartouche_match.group(0)
                and cartouche_modified_match
            ):
                # OK
                if self.verbose:
                    return None
                return None
            # Unexpected cartouche
            if self.verbose:
                pass
            return HeaderError(
                file_path,
                FileChange.MODIFIED,
                "Unexpected header",
                CARTOUCHE_CAP_AIRBUS,
                cartouche_match.group(0),
            )

        # No cartouche, add it
        if self.verbose:
            pass
        return HeaderError(file_path, FileChange.MODIFIED, "No header", CARTOUCHE_CAP_AIRBUS)

    def parse_and_replace_add_cartouche(self, file_path, new_cartouche):
        # Read the content of the file
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        # Search for the cartouche at the start of the content
        cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)

        if cartouche_match:
            first_line = cartouche_match.group(0).split("\n")[1]
            if re.search(pattern=cap_copyright_pattern, string=first_line) and LICENCE in cartouche_match.group(0):
                # OK
                if self.verbose:
                    pass
            else:
                # Unexpected cartouche
                # Than modify it

                new_content = re.sub(
                    pattern=cartouche_pattern,
                    repl=new_cartouche,
                    string=content,
                    flags=re.DOTALL,
                )
                self.write_back(file_path, new_content)

        else:
            # No cartouche
            if self.verbose:
                pass
            new_content = new_cartouche + content
            self.write_back(file_path, new_content)

    def parse_and_replace_modified_cartouche(self, file_path, new_cartouche):
        # Read the content of the file
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        # Search for the cartouche at the start of the content
        cartouche_match = re.search(pattern=cartouche_pattern, string=content, flags=re.DOTALL)
        if cartouche_match:
            cartouche_modified_match = re.search(
                pattern=cartouche_modified_pattern,
                string=str(cartouche_match.group(0)),
                flags=re.DOTALL,
            )

            if (
                cartouche_match.group(0).startswith(f"'''\n{AIRBUS_COPYRIGHT}")
                and LICENCE in cartouche_match.group(0)
                and cartouche_modified_match
            ):
                # OK
                if self.verbose:
                    pass
            else:
                # Unexpected cartouche
                if self.verbose:
                    pass
                new_content = re.sub(
                    pattern=cartouche_pattern,
                    repl=new_cartouche,
                    string=content,
                    flags=re.DOTALL,
                )
                self.write_back(file_path, new_content)

        else:
            # No cartouche, add it
            if self.verbose:
                pass
            new_content = new_cartouche + content
            self.write_back(file_path, new_content)

    def write_back(self, file: str, new_content: str):
        """Write back method"""
        # Write the modified content back to the file
        with open(file, "w", encoding="utf-8") as file:
            file.write(new_content)

    def get_first_commit_time(self, git_git, full_file_path: str) -> datetime:
        """
        Check if the header inside the file is matching the added file header requirement if not return an HeaderError

        :param file_path: path to the file where to check the header status
        :type file_path: str
        """
        logs = git_git.log("--follow", os.path.join(os.getcwd(), full_file_path))

        # isolate : "Date:   Wed Nov 22 11:08:54 2023" like  note that it is the time string by default easy to convert
        # Parse logs Getting Date keep the last one that is the older
        pattern = r"Date:   [A-Z][a-z]{2} [A-Z][a-z]{2} [0-9]{1,2} [0-9]{1,2}:[0-9]{2}:[0-9]{2} [q0-9]{4}"
        dates_list = re.compile(pattern).findall(logs)

        # print(fullFilePath + " : " + str(dates_list))
        date_str = re.sub("Date:   ", "", dates_list[len(dates_list) - 1].strip())

        d = time.strptime(date_str)

        return datetime.fromtimestamp(mktime(d), tz=timezone.utc)

    def has_been_commited_from_airbus(self, git_git, full_file_path: str, refcommit: git.Commit) -> bool:
        # True if committed from Airbus

        commit_date = self.get_first_commit_time(git_git, full_file_path)
        if commit_date is not None:
            return refcommit.committed_datetime > commit_date
        msg = "Commit_date should not be None"
        raise ValueError(msg)

    def write_headers_if_needed_in_repo(self, ignored_exts, ignored_files, sha, repo_dir):

        # Initialize a Git Repo object
        git_repo = git.Repo(repo_dir)
        git_git = git.Git(repo_dir)

        commit_dev = git_repo.commit(BRANCH)
        commit_origin_dev = git_repo.commit(sha)
        diff_index = commit_origin_dev.diff(commit_dev)

        for diff_item in diff_index.iter_change_type("A"):
            # Added
            item_path = diff_item.a_path

            if item_path not in ignored_files:
                if self.verbose:
                    pass

                file_path = os.path.join(repo_dir, item_path)

                # Need to add Cap Header for python file
                if item_path.endswith(".py"):
                    self.parse_and_replace_add_cartouche(file_path, CARTOUCHE_CAP)
                elif item_path.split(".")[-1].lower() in ignored_exts:
                    # Do nothing for pkl, markdown, csv, ...
                    pass
                else:
                    pass

        for diff_item in diff_index.iter_change_type("D"):
            # Deleted
            if item_path not in ignored_files:
                item_path = diff_item.a_path
                if self.verbose:
                    pass

        for diff_item in diff_index.iter_change_type("R"):
            # Renamed
            item_path = diff_item.a_path
            if item_path not in ignored_files:
                item_path_r = diff_item.b_path
                if self.verbose:
                    pass

        for diff_item in diff_index.iter_change_type("M"):
            # Modified
            item_path = diff_item.b_path
            if item_path not in ignored_files:
                if self.verbose:
                    pass

                file_path = os.path.join(repo_dir, item_path)

                # Get the commit that modified this file
                modifications_dates = [
                    modification_commit.committed_datetime.date()
                    for modification_commit in git_repo.iter_commits(BRANCH, paths=item_path)
                    if modification_commit.committed_datetime >= commit_origin_dev.committed_datetime
                ]
                # Add today as we are modifying the file
                # modifications_dates += [datetime.datetime.now()]
                dates = sorted(
                    {modification_commit.strftime("%Y/%m/%d") for modification_commit in modifications_dates}
                )

                date_str = dates[0] if len(dates) == 1 else f"{dates[0]}-{dates[-1]}"

                # Need to add Cap Modified Header section for python file
                if item_path.endswith(".py"):
                    self.parse_and_replace_modified_cartouche(
                        file_path,
                        CARTOUCHE_BASE.format(
                            f"{AIRBUS_COPYRIGHT}\n{CAP_MODIFIED_COPYRIGHT.format(date_str)}\n\n{LICENCE}"
                        ),
                    )
                    if not self.has_been_commited_from_airbus(git_git, file_path, commit_origin_dev):
                        self.parse_and_replace_add_cartouche(file_path, CARTOUCHE_CAP)
                elif item_path.split(".")[-1].lower() in ignored_exts:
                    # Do nothing for pkl, markdown, csv, ...
                    pass
                elif item_path.endswith((".ts", ".html", ".scss")):
                    # Skip modified ts
                    pass
                else:
                    self.parse_and_replace_modified_cartouche(
                        file_path,
                        CARTOUCHE_BASE.format(
                            f"{AIRBUS_COPYRIGHT}\n{CAP_MODIFIED_COPYRIGHT.format(date_str)}\n\n{LICENCE}"
                        ),
                    )
                    if not self.has_been_commited_from_airbus(git_git, file_path, commit_origin_dev):
                        self.parse_and_replace_add_cartouche(file_path, CARTOUCHE_CAP)

        for diff_item in diff_index.iter_change_type("T"):
            # Changed in the type path
            item_path = diff_item.a_path
            if item_path not in ignored_files and self.verbose:
                pass

    def check_headers(self, ignored_exts, ignored_file, airbus_rev_commit: str):
        header_error_list = []

        # Initialize a Git Repo object wity current directory
        repo = git.Repo(repo_dir)
        git_git = git.Git(repo_dir)

        # fetch all
        for remote in repo.remotes:
            remote.fetch()

        # diff between airbus version  and current checkout version (dev) -> all modifications and addition between Airbus version published on OS-C and the newly published
        commit_dev = repo.head.commit
        commit_airbus = repo.commit(airbus_rev_commit)

        diff_index = commit_airbus.diff(commit_dev)

        for diff_item in diff_index.iter_change_type("A"):
            # Added
            item_path = diff_item.a_path

            if item_path not in ignored_file:
                if self.verbose:
                    pass

                file_path = os.path.join(repo_dir, item_path)

                # Need to add Cap Header for python file
                if item_path.endswith(".py"):
                    error = self.check_header_for_added_file(file_path)
                    # if has_been_commited_from_airbus(file_path, commit_airbus):
                    # print(
                    #     "ERROR detected has added but committed by Airbus : "
                    #     + file_path
                    # )
                    # error = check_header_for_modified_file(file_path)
                    if error:
                        header_error_list.append(error)

                elif item_path.split(".")[-1].lower() in ignored_exts:
                    # Do nothing for pkl, markdown, csv, ...
                    pass
                else:
                    # Need to check not handled file too
                    error = self.check_header_for_added_file(file_path)
                    # if has_been_commited_from_airbus(file_path, commit_airbus):
                    #    error = check_header_for_modified_file(file_path)
                    if error:
                        header_error_list.append(error)

            else:
                if self.verbose:
                    pass

        for diff_item in diff_index.iter_change_type("D"):
            # Deleted
            item_path = diff_item.a_path
            if item_path not in ignored_file and self.verbose:
                pass

        for diff_item in diff_index.iter_change_type("R"):
            # Renamed
            item_path = diff_item.a_path
            item_path_r = diff_item.b_path

            if item_path_r not in ignored_file and self.verbose:
                pass

        for diff_item in diff_index.iter_change_type("M"):
            # Modified
            item_path = diff_item.b_path

            if item_path not in ignored_file:
                if self.verbose:
                    pass

                file_path = os.path.join(repo_dir, item_path)

                # Need to add Cap Modified Header section for python file
                if item_path.endswith(".py"):
                    error = self.check_header_for_modified_file(file_path)
                    if not self.has_been_commited_from_airbus(git_git, file_path, commit_airbus):
                        error = self.check_header_for_added_file(
                            file_path
                        )  # if not commited from Airbus then it is an added file detected by error by diff has a modified file
                    if error:
                        header_error_list.append(error)

                elif item_path.split(".")[-1].lower() in ignored_exts:
                    # Do nothing for pkl, markdown, csv, ...
                    pass
                elif item_path.endswith((".ts", ".html", ".scss")):
                    # Skip modified ts
                    pass
                else:
                    # Need to check not handled file too
                    error = self.check_header_for_modified_file(file_path)
                    if not self.has_been_commited_from_airbus(git_git, file_path, commit_airbus):
                        error = self.check_header_for_added_file(
                            file_path
                        )  # if not commited from Airbus then it is an added file detected by error by diff has a modified file
                    if error:
                        header_error_list.append(error)
            else:
                if self.verbose:
                    pass

        for diff_item in diff_index.iter_change_type("T"):
            # Changed in the type path
            item_path = diff_item.a_path

            if item_path not in ignored_file and self.verbose:
                pass

        if len(header_error_list) > 0:
            errors = ""
            for he in header_error_list:
                errors += str(he)
            errors += f"\nFound {len(header_error_list)} header error(s)"
            msg = f"{errors}"
            raise Exception(msg)
