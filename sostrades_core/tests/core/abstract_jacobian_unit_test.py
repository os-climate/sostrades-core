'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/02 Copyright 2023 Capgemini

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
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from abc import ABC, abstractmethod

import inspect

import glob
from importlib import import_module
from os.path import join, basename, dirname
import os
import logging
from multiprocessing import Process
from gemseo.core.discipline import MDODiscipline

PROCESS_IN_PARALLEL = 5


class AbstractJacobianUnittest(unittest.TestCase, ABC):
    """
    unit test jacobian management implement
    """

    DUMP_JACOBIAN = False
    PICKLE_DIRECTORY = 'jacobian_pkls'

    def generate_analytic_gradient_pickle(self, test_names=[]):
        """ Main method to launch associated jacobian test and force dump of jacobian pickle
        """
        local_logger = logging.getLogger(__name__)
        jacobian_test_entries = self.analytic_grad_entry()

        for entry in jacobian_test_entries:
            is_in_list = False
            if len(test_names) > 0:
                for test_name in test_names:
                    if test_name in str(entry):
                        is_in_list = True
            else:
                is_in_list = True
            if not is_in_list:
                continue
            try:
                local_logger.info(
                    f'Jacobian launched on {str(entry)}')
                self.setUp()
                AbstractJacobianUnittest.DUMP_JACOBIAN = True
                entry()
            except Exception as ex:
                local_logger.exception(
                    f'Jacobian fail on {str(entry)}')

    @abstractmethod
    def analytic_grad_entry(self):
        """ Method to overload with jacobian test in order to be dump with the automated script
        """
        raise TypeError('test_analytic_gradient must be overloaded')

    def check_jacobian(self, location, filename, discipline, local_data, inputs, outputs,
                       step=1e-15, derr_approx='complex_step',
                       input_column=None, output_column=None, threshold=1e-8, parallel=False,
                       n_processes=5, linearization_mode='auto', directory=PICKLE_DIRECTORY):
        """ Method that encapsulate check_jacobian call in order to witch between loading and dumping mode
        """

        if n_processes > MDODiscipline.N_CPUS:
            n_processes = MDODiscipline.N_CPUS

        local_logger = logging.getLogger(__name__)

        file_path = join(location, directory,
                         filename)

        if AbstractJacobianUnittest.DUMP_JACOBIAN:
            local_logger.info(
                f'Jacobian dump mode enable on {join(location, filename)}')
            check_flag = discipline.check_jacobian(step=step, inputs=inputs, input_data=local_data,
                                                   outputs=outputs, derr_approx=derr_approx,
                                                   dump_jac_path=file_path, input_column=input_column, output_column=output_column, parallel=parallel,
                                                   n_processes=n_processes, linearization_mode=linearization_mode)
        else:
            check_flag = discipline.check_jacobian(step=step, inputs=inputs, input_data=local_data,
                                                   outputs=outputs, derr_approx=derr_approx,
                                                   load_jac_path=file_path, input_column=input_column, output_column=output_column, parallel=parallel,
                                                   n_processes=n_processes, linearization_mode=linearization_mode)

        self.assertTrue(check_flag, msg=f"Wrong gradient in {discipline.name}")

    @staticmethod
    def launch_all_pickle_generation(root_module, file_regex='l1*.py', directories=[PICKLE_DIRECTORY], test_names=[]):
        """ Static method that look for jacobian test to generate associated pickle (in the given folder)
            and then push newly generated files into git repository
        """

        root_dir = dirname(root_module.__file__)
        local_logger = logging.getLogger(__name__)
        local_logger.info(
            f'Looking for L1 tests into {root_dir}')

        l1_list = glob.glob(
            join(root_dir, file_regex))
        local_logger.info(
            f'found files {l1_list}')

        process_list = []

        for file in l1_list:
            file_module = basename(file).replace('.py', '')
            module_name = f'{root_module.__name__}.{file_module}'

            try:
                a = import_module(module_name)
                for name, obj in inspect.getmembers(a):
                    if inspect.isclass(obj) and issubclass(obj, AbstractJacobianUnittest) and name != AbstractJacobianUnittest.__name__:
                        local_logger.info(
                            f'Execute jacobian dump on {module_name}')
                        inst = obj()
                        process_list.append(
                            Process(target=inst.generate_analytic_gradient_pickle(test_names=test_names)))
            except Exception as ex:
                local_logger.error(f'Error on module : {module_name}\n{ex}')

        if len(process_list) > 0:

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

                for entry in candidate_process:
                    entry.join()

            for directory in directories:
                os.system(f'git add ./{directory}/*.pkl')
            os.system(f'git commit -m "regeneration of jacobian pickles for {file_regex}"')
            os.system('git pull')
            os.system('git push')
        else:
            local_logger.warning('Nothing run so nothing to commit/push')