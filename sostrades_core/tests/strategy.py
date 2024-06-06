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
import glob
import os
import sys
import tempfile
from typing import Union

import pytest

"""
TEST STRATEGY MODULE

How to use ?

- open command prompt
- go into the repo you want 'cd <repo>'
- run 'python strategy.py'

-> this will run all l0 tests in repo.


- run 'python strategy.py l1' for l1 tests (not usecases)
- run 'python strategy.py l2' for l1 tests (not usecases)
- run 'python strategy.py uc' for testing usecases

"""

# Create a temporary file


def run_tests_l0_l1_l2(main_folder, file_pattern):
    """run the tests in main folder corresponding to file_pattern"""
    initial_path = os.path.abspath(os.path.curdir)
    sub_test_folder = 'tests' if main_folder != "sos_trades_api" else os.path.join('tests', 'controllers')
    test_folder = os.path.join(main_folder, sub_test_folder)
    os.chdir(test_folder)
    # Print the arguments
    print(rf"STARTING TESTS {test_folder}\{file_pattern}")

    test_files = glob.glob(file_pattern)
    if len(test_files) == 0:
        print(f"No test file discovered corresponding to pattern {file_pattern}")
        return 0
    try:
        # Use subprocess to run the pytest command on the specified file
        exitcode = pytest.main(['-W', 'ignore', '-v', '--durations=5', '--durations-min=2.0'] + test_files)
    except Exception as e:
        # Handle any errors or exceptions here
        print(f"Error while running pytest on {test_files}: {e}")
        exitcode = -1

    os.chdir(initial_path)
    return exitcode


def run_generated_usecase_test(temp_file_path):
    """run the tests in main folder corresponding to file_pattern"""
    try:
        # Use subprocess to run the pytest command on the specified file
        exitcode = pytest.main(['-W', 'ignore','--verbose', '--durations=5', '--durations-min=2.0'] + [temp_file_path])
    except Exception as e:
        # Handle any errors or exceptions here
        print(f"Error while running pytest on {temp_file_path}: {e}")
        exitcode = -1

    return exitcode


def gather_usecases(mainfolder: str, processes_reponame: str):
    """gather all usecases of a repo"""
    processes_root_path = os.path.join(mainfolder, processes_reponame)

    def recursive_gathering(folder_path) -> list[str]:
        usecases_path = []
        items_in_directory = os.listdir(folder_path)
        folders = [item for item in items_in_directory if os.path.isdir(os.path.join(folder_path, item))]
        for folder in folders:
            sub_folder_path = os.path.join(folder_path, folder)
            usecases_path += recursive_gathering(sub_folder_path)

        usecases_path += glob.glob(os.path.join(folder_path, "usecase*.py"))
        return usecases_path

    usecases = recursive_gathering(processes_root_path)
    return usecases


def generate_script_for_usecases_test(usecases):
    """generates code for testing usecases"""
    script = "import unittest\n\n\nclass TestUsecases(unittest.TestCase):\n\tpass\n"

    def get_test_function_for_usecase(usecase_path: str, test_number: int):
        path_for_import = usecase_path.replace(".py",'').replace("\\",'.').replace("/",'.')
        test_name = f"test_{test_number:03d}_"+ "_".join(usecase_path.replace(".py",'').replace("\\",'.').replace("/",'.').split(".")[-2:])
        test_function_code = f"\n\tdef {test_name}(self):\n" \
                             f"\t\tfrom {path_for_import} import Study\n" \
                             f"\t\tusecase = Study()\n" \
                             f"\t\tusecase.test()\n"
        return test_function_code
    for i, usecase_path in enumerate(usecases):
        script += get_test_function_for_usecase(usecase_path, i)

    return script


def delete_generated_file(path):
    if os.path.exists(path):
        os.remove(path)


def test_usecases(mainfolder: str, processes_reponame: str):
    """test usecases"""
    usecases = gather_usecases(mainfolder, processes_reponame)
    script_test = generate_script_for_usecases_test(usecases)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(script_test)
    initial_dir = os.path.abspath(os.curdir)
    temp_file_folder_path = os.path.dirname(temp_file_path)
    os.chdir(temp_file_folder_path)

    try:
        exitcode = run_generated_usecase_test(temp_file_path)
        delete_generated_file(temp_file_path)
        os.chdir(initial_dir)
        return exitcode
    except KeyboardInterrupt:
        delete_generated_file(temp_file_path)
        os.chdir(initial_dir)
        sys.exit(2)


allowed_test_types = ['l0', 'l1', 'l2', 'uc', '--pattern']
l_tests_pattern_mapping = {
    "l0": "l0_*.py",
    "l1": "l1_*.py",
    "l2": "l2_*.py",
}


def gather_arguments(test_type_default_value: str):
    """gathers arguments for running tests"""
    testtype = test_type_default_value
    file_pattern = ''
    if len(sys.argv) >= 2:
        testtype = sys.argv[1]
    if testtype not in allowed_test_types:
        print(f"Test type argument is wrong. must be in {allowed_test_types}")
    if testtype == '--pattern':
        if len(sys.argv) >= 3:
            file_pattern = sys.argv[2]
        else:
            print('please indicate the specific pattern')
            sys.exit(-1)
    if testtype != 'uc' and testtype != '--pattern':
        file_pattern = l_tests_pattern_mapping[testtype]

    return testtype, file_pattern


def test_strategy(main_folder_default_value: str, processes_folder: Union[str, None]):
    sys.modules.pop('logging')
    test_type, file_pattern = gather_arguments(test_type_default_value="l0")

    if test_type != "uc":
        exit_code = run_tests_l0_l1_l2(main_folder_default_value, file_pattern)
    elif processes_folder is not None:
        exit_code = test_usecases(main_folder_default_value, processes_folder)
    else:
        print('No processes repo specified, exiting test.')
        exit_code = 0

    print(f"EXIT CODE {exit_code}")
    sys.exit(exit_code)
