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

from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.sos_processes.processes_factory import SoSProcessFactory
from importlib import import_module
from os.path import dirname, isdir
from os import listdir, makedirs, environ
from logging import DEBUG

from copy import deepcopy
from tempfile import gettempdir
import traceback
from gemseo.utils.compare_data_manager_tooling import compare_dict, \
    delete_keys_from_dict
from multiprocessing import Process, Queue
from queue import Empty
import time
import platform

PROCESS_IN_PARALLEL = 5


def manage_process_queue(process_list, message_queue):
    """ Regarding a given process list (already start)
    manage sub process lifecycle and queue message passing

    :params: process_list, list of processses to manage
    :type: multiprocessing.Process

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue


    :return: [str, str], [test status, error message list]
    """
    liveprocs = list(process_list)
    global_output_error = ''
    global_test_passed = True
    while liveprocs:
        try:
            while 1:
                messages = message_queue.get(False)
                test_passed = messages[0]
                output_error = messages[1]

                if not test_passed:
                    global_test_passed = False

                if len(output_error) > 0:
                    global_output_error += output_error
        except Empty:
            pass

        time.sleep(0.5)  # Give tasks a chance to put more data in
        if not message_queue.empty():
            continue
        liveprocs = [p for p in liveprocs if p.is_alive()]

    return global_test_passed, global_output_error


def manage_process_launch(process_list, message_queue):
    """ Regarding a given process list (not started)
    manage sub process launching

    :params: process_list, list of processses to manage
    :type: multiprocessing.Process

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue


    :return: (str, str) , test status and error message list
    """

    global_output_error = ''
    global_test_passed = True

    while len(process_list) > 0:
        candidate_process = []

        if len(process_list) > PROCESS_IN_PARALLEL:
            candidate_process = [process_list.pop()
                                 for index in range(PROCESS_IN_PARALLEL)]
        else:
            candidate_process = process_list
            process_list = []

        for process in candidate_process:
            process.start()

        result_test_passed, result_output_error = manage_process_queue(
            candidate_process, message_queue)

        global_test_passed = global_test_passed and result_test_passed

        if len(result_output_error) > 0:
            global_output_error += result_output_error

    return global_test_passed, global_output_error


def processed_multiple_configure(usecase, message_queue):
    """ Management of usecase regarding a multiple configure methods

    :params: usecase, usecase name (as module name) to manage
    :type: str

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue

    """
    test_passed = True
    configured = True
    output_error = ''
    try:
        dm_data_dict_1, dm_data_dict_2 = multiple_configure(usecase)
    except Exception as e:
        test_passed = False
        configured = False
        output_error += f'Error while Configuring twice {usecase}:\n {e}'
        output_error += '\n---------------------------------------------------------\n'
    if configured:
        try:
            dict_error = {}
            compare_dict(dm_data_dict_1, dm_data_dict_2, '', dict_error)
            if dict_error != {}:
                test_passed = False
                for error in dict_error:
                    output_error += f'Error while Configuring twice {usecase}:\n'
                    output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                    output_error += '\n---------------------------------------------------------\n'
        except Exception as e:
            traceback.print_exc()
            test_passed = False
            output_error += f'Error while comparing data_dicts of {usecase}:\n {e}'
            output_error += '\n---------------------------------------------------------\n'

    message_queue.put([test_passed, output_error])


def configure_twice_all_usecases_and_compare_dm(processes_repo):
    """ Management of usecase regarding a multiple configure methods

    :params: processes_repo, list of usecase module name to manage
    :type: str[]

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue

    :return: [str, str], [test status, error message list]

    """

    # if platform.system() == 'Windows':
    #     raise OSError(
    #         'This method launch usecase with multiprocessing.It is not intended to be runned under Windows OS regarding the ressources consumptio')

    usecase_dict = get_all_usecases(processes_repo)
    message_queue = Queue()
    process_list = []

    for usecase in usecase_dict:
        process_list.append(
            Process(target=processed_multiple_configure, args=(usecase, message_queue,)))

    return manage_process_launch(process_list, message_queue)


def processed_run_twice_all_usecases(usecase, message_queue, force_run=False):
    """ Management of usecase regarding a multiple run methods

    :params: usecase, usecase name (as module name) to manage
    :type: str

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue

    """
    test_passed = True
    runned = True
    output_error = ''
    try:
        dm_data_dict_1, dm_data_dict_2 = multiple_run(usecase, force_run)
    except Exception as e:
        test_passed = False
        runned = False
        output_error += f'Error while Running twice {usecase}:\n {e}'
        output_error += '\n---------------------------------------------------------\n'
    if runned:
        try:
            dict_error = {}
            if 'optim' in usecase or 'mda' in usecase or 'opt' in usecase:
                # dict shouldn't be equal between optim iterations
                print(f'----- SKIPPED DICT COMPARISON FOR {usecase} -----')
            else:
                # remove unwanted elements from dm comparison
                #   - residuals_history because iterations are different
                #   - type metadata because second run does not perform type conversion so metadata is None
                unwanted_keys, keys_to_none = [], {}
                for key, value in dm_data_dict_1.items():
                    if 'residuals_history' in key:
                        unwanted_keys += [key]
                    if type(value) == dict:
                        if 'type_metadata' in value.keys():
                            keys_to_none[key] = 'type_metadata'
                for key, value in keys_to_none.items():
                    dm_data_dict_1[key][value] = None
                [dm_data_dict_1.pop(key) for key in unwanted_keys]
                [dm_data_dict_2.pop(key) for key in unwanted_keys]
                compare_dict(dm_data_dict_1,
                             dm_data_dict_2, '', dict_error)
            if dict_error != {}:
                test_passed = False
                for error in dict_error:
                    output_error += f'Error while Running twice {usecase}:\n'
                    output_error += f'Mismatch in {error}: {dict_error.get(error)}'
                    output_error += '\n---------------------------------------------------------\n'
        except Exception as e:
            traceback.print_exc()
            test_passed = False
            output_error += f'Error while comparing data_dicts of {usecase}:\n {e}'
            output_error += '\n---------------------------------------------------------\n'

    message_queue.put([test_passed, output_error])


def run_twice_all_usecases_and_compare_dm(processes_repo, force_run=False):
    """ Management of usecase regarding a multiple configure methods

    :params: processes_repo, list of usecase module name to manage
    :type: str[]

    :params: message_queue, message queue associted to process and subprocesses
    :type: multiprocessing.Queue

    :return: [str, str], [test status, error message list]

    """
    # if platform.system() == 'Windows':
    #     raise OSError(
    #         'This method launch usecase with multiprocessing.It is not intended to be runned under Windows OS regarding the resources consumption')

    usecase_dict = get_all_usecases(processes_repo)
    message_queue = Queue()
    process_list = []

    for usecase in usecase_dict:
        process_list.append(
            Process(target=processed_run_twice_all_usecases, args=(usecase, message_queue, force_run,)))

    return manage_process_launch(process_list, message_queue)


def get_all_usecases(processes_repo):
    '''
        Retrieve all usecases in a repository
        :params: processes_repo, repository where to find processes
        :type: String

        :returns: List of strings: ['usecase_1','usecase_2']
    '''

    process_factory = SoSProcessFactory(additional_repository_list=[
        processes_repo], search_python_path=False)
    process_list = process_factory.get_processes_dict()
    usecase_list = []
    for repository in process_list:
        for process in process_list[repository]:
            try:
                process_module = '.'.join([repository, process])
                imported_module = import_module(process_module)
                if imported_module is not None and imported_module.__file__ is not None:
                    process_directory = dirname(imported_module.__file__)
                    # Run all usecases
                    for usecase_py in listdir(process_directory):
                        if usecase_py.startswith('usecase'):
                            usecase = usecase_py.replace('.py', '')
                            usecase_list.append('.'.join([repository, process, usecase]))
            except Exception as error:
                print(f'An error occurs when trying to load {process_module}\n{error}')
    return usecase_list


def multiple_configure(usecase):
    '''
        Configure twice a usecase and return the two treeviews from the configure
        :params: usecase, usecase to configure twice
        :type: String
        :returns: Two dm as dictionary
    '''

    print(f'----- CONFIGURE TWICE A USECASE {usecase} -----')
    # Instanciate Study
    imported_module = import_module(usecase)
    uc = getattr(imported_module, 'Study')()
    # First step : Dump data to a temp folder
    uc.set_dump_directory(gettempdir())
    uc.load_data()
    dump_dir = uc.dump_directory
    uc.dump_data(dump_dir)

    # Set repo_name, proc_name, study_name to create BaseStudyManager
    repo_name = uc.repository_name
    proc_name = uc.process_name
    study_name = uc.study_name

    print("---- FIRST CONFIGURE ----")
    # First run : Load Data in a new BaseStudyManager and run study
    study_1 = BaseStudyManager(repo_name, proc_name, study_name)
    study_1.load_data(from_path=dump_dir)
    study_1.execution_engine.configure()
    # Deepcopy dm
    dm_dict_1 = deepcopy(study_1.execution_engine.get_anonimated_data_dict())
    study_1.dump_data(dump_dir)

    # Second run : Load Data in a new BaseStudyManager and run study
    print("---- SECOND CONFIGURE ----")
    study_2 = BaseStudyManager(repo_name, proc_name, study_name)
    study_2.load_data(from_path=dump_dir)
    study_2.execution_engine.configure()
    # Deepcopy dm
    dm_dict_2 = deepcopy(study_2.execution_engine.get_anonimated_data_dict())

    delete_keys_from_dict(dm_dict_1), delete_keys_from_dict(dm_dict_2)
    return dm_dict_1, dm_dict_2


def multiple_run(usecase, force_run=False):
    '''
        Run twice a usecase and return the two treeviews from the runs
        :params: usecase, usecase to run twice
        :type: String
        :returns: Two treeview as dictionary
    '''
    print(f'----- RUN TWICE A USECASE {usecase} -----')
    # Instanciate Study
    imported_module = import_module(usecase)
    uc = getattr(imported_module, 'Study')()

    # ----------------------------------------------------
    # First step : Dump data to a temp folder

    # Dump data can be set using en environment variable (mainly for devops test purpose
    # So check that environment variable before using de default location

    # Default variable location
    base_dir = f'{gettempdir()}/references'

    if environ.get('SOS_TRADES_REFERENCES_SPECIFIC_FOLDER') is not None:
        base_dir = environ['SOS_TRADES_REFERENCES_SPECIFIC_FOLDER']

    if not isdir(base_dir):
        makedirs(base_dir, exist_ok=True)

    print(f'Reference location for use case {usecase} is {base_dir}')

    uc.set_dump_directory(base_dir)
    uc.load_data()
    dump_dir = uc.dump_directory
    uc.dump_data(dump_dir)
    uc.dump_disciplines_data(dump_dir)

    if uc.run_usecase() or force_run == True:
        # Set repo_name, proc_name, study_name to create BaseStudyManager
        repo_name = uc.repository_name
        proc_name = uc.process_name
        study_name = uc.study_name

        print("---- FIRST RUN ----")
        # First run : Load Data in a new BaseStudyManager and run study
        try:
            study_1 = BaseStudyManager(repo_name, proc_name, study_name)
            study_1.load_data(from_path=dump_dir)
            study_1.set_dump_directory(base_dir)
            study_1.run(logger_level=DEBUG, dump_study=True, for_test=True)
            # Deepcopy dm
            dm_dict_1 = deepcopy(
                study_1.execution_engine.get_anonimated_data_dict())
            study_1.dump_data(dump_dir)
        except Exception as e:
            raise Exception(f'Error during first run: {e}')

        # Second run : Load Data in a new BaseStudyManager and run study
        print("---- SECOND RUN ----")
        try:
            study_2 = BaseStudyManager(repo_name, proc_name, study_name)
            study_2.load_data(from_path=dump_dir)
            study_2.run(logger_level=DEBUG)
            # Deepcopy dm
            dm_dict_2 = deepcopy(
                study_2.execution_engine.get_anonimated_data_dict())
        except Exception as e:
            raise Exception(f'Error during second run: {e}')

        # Delete ns ref from the two DMs
        delete_keys_from_dict(dm_dict_1), delete_keys_from_dict(dm_dict_2)
        return dm_dict_1, dm_dict_2
    else:
        print(f'{usecase} is configured not to run, skipping double run.')
        return {}, {}
