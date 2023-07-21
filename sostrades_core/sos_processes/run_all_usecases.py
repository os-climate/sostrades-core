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
'''
mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
'''
from sostrades_core.sos_processes.processes_factory import SoSProcessFactory
from importlib import import_module
from os.path import dirname, isdir
from os import listdir, makedirs
from tempfile import gettempdir


class UseCaseFailedException(Exception):
    """Exception if a usecase fails.

    Attributes:
        message -- explanation of the error
    """
    error_list = []

    def __init__(self, error_list):
        self.error_list = error_list
        super().__init__(self.error_list)

    def __str__(self):
        return '\n' + '\n'.join(self.error_list)


def get_all_usecases(processes_repo):
    '''
    return all usecases of a repository
    '''
    process_factory = SoSProcessFactory(additional_repository_list=[
                                        processes_repo], search_python_path=False)
    process_dict = process_factory.get_processes_dict()
    usecase_dict = []
    for repository in process_dict:
        for process in process_dict[repository]:
            imported_module = import_module('.'.join([repository, process]))
            process_directory = dirname(imported_module.__file__)
            # Run all usecases
            for usecase_py in listdir(process_directory):
                if usecase_py.startswith('usecase'):
                    usecase = usecase_py.replace('.py', '')
                    usecase_dict.append(
                        '.'.join([repository, process, usecase]))
    return usecase_dict


def run_all_usecases(processes_repo):
    '''
    Run all usecases of a specific repository
    Raise an exception if a use doesn't run
    '''
    all_usecase_passed = True
    error_list = []
    # Retrieve all processes for this repository only
    process_factory = SoSProcessFactory(additional_repository_list=[
                                        processes_repo], search_python_path=False)
    process_dict = process_factory.get_processes_dict()
    # Set dir to dump reference
    dump_dir = f'{ gettempdir() }/references'
    if not isdir(dump_dir):
        makedirs(dump_dir, exist_ok=True)
    for repository in process_dict:
        for process in process_dict[repository]:

            imported_module = import_module('.'.join([repository, process]))

            if imported_module is not None and imported_module.__file__ is not None:
                process_directory = dirname(imported_module.__file__)
                # Run all usecases
                for usecase_py in listdir(process_directory):
                    if usecase_py.startswith('usecase'):
                        try:
                            usecase = usecase_py.replace('.py', '')
                            imported_module = import_module(
                                '.'.join([repository, process, usecase]))
                            imported_usecase = getattr(
                                imported_module, 'Study')()
                            imported_usecase.set_dump_directory(
                                dump_dir)
                            imported_usecase.load_data()
                            imported_usecase.run(dump_study=True,
                                                 for_test=True)
                        except Exception as e:
                            all_usecase_passed = False
                            error_list.append(
                                f'An error occured while running the usecase {repository}.{process}.{usecase}: {e}')
            else:
                print(
                    f"Process {'.'.join([repository, process])} skipped. Check presence of __init__.py in the folder.")

    if all_usecase_passed is False:
        raise UseCaseFailedException(error_list)


def run_optim_usecases(processes_repo):
    '''
    Run all usecases of a specific repository
    Raise an exception if a use doesn't run
    '''
    all_usecase_passed = True
    error_list = []
    # Retrieve all processes for this repository only
    process_factory = SoSProcessFactory(additional_repository_list=[
                                        processes_repo], search_python_path=False)
    process_dict = process_factory.get_processes_dict()
    # Set dir to dump reference
    dump_dir = f'{ gettempdir() }/references'
    if not isdir(dump_dir):
        makedirs(dump_dir, exist_ok=True)
    for repository in process_dict:
        for process in process_dict[repository]:

            imported_module = import_module('.'.join([repository, process]))

            if imported_module is not None and imported_module.__file__ is not None:
                process_directory = dirname(imported_module.__file__)
                # Run all usecases
                for usecase_py in listdir(process_directory):
                    if usecase_py.startswith('usecase'):
                        try:
                            usecase = usecase_py.replace('.py', '')
                            imported_module = import_module(
                                '.'.join([repository, process, usecase]))
                            imported_usecase = getattr(
                                imported_module, 'Study')()
                            if not imported_usecase.run_usecase:
                                print('Optim usecases detected... Launch run.')
                                imported_usecase.set_run_usecase(True)
                                imported_usecase.set_dump_directory(
                                    dump_dir)
                                imported_usecase.load_data()
                                imported_usecase.run(dump_study=True,
                                                     for_test=True)

                        except Exception as e:
                            all_usecase_passed = False
                            error_list.append(
                                f'An error occured while running the usecase {repository}.{process}.{usecase}: {e}')
            else:
                print(
                    f"Process {'.'.join([repository, process])} skipped. Check presence of __init__.py in the folder.")

    if all_usecase_passed is False:
        raise UseCaseFailedException(error_list)
