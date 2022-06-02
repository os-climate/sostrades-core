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
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal
import pprint
import numpy as np
import pandas as pd
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from copy import deepcopy
from tempfile import gettempdir


class TestMultiScenarioOfDoeEval(unittest.TestCase):
    """
    MultiScenario and doe_eval processes test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''

        self.study_name = 'MyStudy'
        self.ns = f'{self.study_name}'
        self.sc_name = "DoE_Eval"

        self.exec_eng = ExecutionEngine(self.ns)
        self.factory = self.exec_eng.factory

    def setup_usecase(self, restricted=True):
        """
        Define a set of data inputs
        """
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        ######### Numerical values   ####
        x = 2.0
        y = 3.0

        ax2 = 4.0
        by2 = 5.0
        cx = 6.0
        dy = 7.0
        exy = 12.0

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                          'lower_bnd': [-5., -5.],
                          'upper_bnd': [+5., +5.],
                          #'enable_variable': [True, True],
                          # 'activated_elem': [[True], [True]]
                          }
        my_doe_algo = "lhs"
        n_samples = 4

        dspace_xy = pd.DataFrame(dspace_dict_xy)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        if restricted == False:
            values_dict[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
            values_dict[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            values_dict[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'Empty'

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_xy
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_z
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace_xy

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        values_dict[f'{self.study_name}.DoE_Eval.Hessian.x'] = x
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.y'] = y

        values_dict[f'{self.study_name}.DoE_Eval.Hessian.ax2'] = ax2
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.by2'] = by2
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.cx'] = cx
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.dy'] = dy
        values_dict[f'{self.study_name}.DoE_Eval.Hessian.exy'] = exy

        return [values_dict]

    def setup_usecase_2(self, restricted=True):
        """
        Define a set of data inputs
        """
        # SubProcess selection values
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        ######### Numerical values   ####

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y'],
                          'lower_bnd': [-5., -5.],
                          'upper_bnd': [+5., +5.],
                          #'enable_variable': [True, True],
                          # 'activated_elem': [[True], [True]]
                          }
        my_doe_algo = "lhs"
        n_samples = 4

        dspace_xy = pd.DataFrame(dspace_dict_xy)

        ######### Fill the dictionary for dm   ####
        values_dict = {}
        if restricted == False:
            values_dict[f'{self.study_name}.DoE_Eval.repo_of_sub_processes'] = repo
            values_dict[f'{self.study_name}.DoE_Eval.sub_process_folder_name'] = mod_id
            values_dict[f'{self.study_name}.DoE_Eval.usecase_of_sub_process'] = 'usecase'

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_xy
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_z
        values_dict[f'{self.study_name}.DoE_Eval.design_space'] = dspace_xy

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo
        values_dict[f'{self.study_name}.DoE_Eval.algo_options'] = {
            'n_samples': n_samples}

        return [values_dict]

    def test_01_build_doe_eval_from_python_and_disc(self):
        '''
        Test the creation of the doe and nested disciplines from a python sos_processes with an input wrapped discipline
        It is then used (fill data and execute)
        '''
        print('test_01_build_doe_eval_from_python_and_disc')
        mod_path = 'sos_trades_core.sos_wrapping.test_discs.disc_hessian.DiscHessian'
        disc_name = 'Hessian'
        disc_builder = self.exec_eng.factory.get_builder_from_module(
            disc_name, mod_path)
        builder_list = [disc_builder]

        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        # doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
        #    'DoE_Eval', 'build_doe_eval', builder_list)
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_02_build_doe_eval_from_python_and_proc(self):
        '''
        Test the creation of the doe and nested disciplines from a python sos_processes with an input process for discipline selection
        It is then used (fill data and execute)
        '''
        print('test_02_build_doe_eval_from_python_and_proc')
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'

        builder_list = self.exec_eng.factory.get_builder_from_process(
            repo=repo, mod_id=mod_id)
        self.exec_eng.ns_manager.add_ns(
            'ns_doe_eval', f'{self.exec_eng.study_name}.DoE_Eval')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_03_build_doe_eval_with_empty_disc(self):
        '''
        Test the creation of the doe without nested disciplines and directly from DoE_eval class
        '''
        print('test_03_build_doe_eval_with_empty_disc')
        builder_list = []

        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        # doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
        #    'DoE_Eval', 'build_doe_eval', builder_list)
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        print(self.exec_eng.display_treeview_nodes())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_04_build_doe_eval_with_nested_proc_selection(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_04_build_doe_eval_with_nested_proc_selection')
        # setup an empty doe
        builder_list = []
        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'build_doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # provide a process (with disciplines) to the set doe
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # provide inputs to the set doe with disciplines
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_05_build_doe_eval_with_nested_proc_selection_through_process_driver(self):
        '''
        Test the creation of the doe without nested disciplines directly from DoE_eval class : 
        through_process_test_driver_build_doe_eval_empty.
        And then its update with with an input process for discipline selection.
        It is then used (fill data and execute)
        '''
        print('test_05_build_doe_eval_with_nested_proc_selection_through_process_driver')
        # setup an empty doe and configure
        driver_repo = 'sos_trades_core.sos_processes.test'
        driver_mod_id = 'test_driver_build_doe_eval_empty'
        doe_eval_builder = self.exec_eng.factory.get_builder_from_process(
            repo=driver_repo, mod_id=driver_mod_id)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # provide a sub process (with disciplines) to the set doe and configure
        repo = 'sos_trades_core.sos_processes.test'
        mod_id = 'test_disc_hessian'
        dict_values = {}
        dict_values['MyStudy.DoE_Eval.repo_of_sub_processes'] = repo
        dict_values['MyStudy.DoE_Eval.sub_process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # provide data inputs to the set doe with disciplines and configure
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())
        # Provide subprocess and provide data input
        values_dict = self.setup_usecase(restricted=False)[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.display_treeview_nodes()
        # run
        print(self.exec_eng.execute())
        print('Inputs')
        print(self.exec_eng.dm.get_data_dict_values())

    def test_06_build_doe_eval_with_nested_proc_selection_through_process_driver(self):

        from os.path import join, dirname
        from sos_trades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        repo = 'sos_trades_core.sos_processes.test'
        mod_id_empty_doe = 'test_driver_build_doe_eval_empty'
        self.study_name = 'Essai'

        study_dump = BaseStudyManager(repo, mod_id_empty_doe, 'Essai')
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        dict_values = self.setup_usecase_2(restricted=False)
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        print(study_dump.ee.dm.get_data_dict_values())
        study_dump.ee.configure()
        print(study_dump.ee.dm.get_data_dict_values())
        # study_dump.run()

        #study_load = BaseStudyManager(repo, mod_id_empty_doe, 'Essai')
        # study_load.load_data(from_path=dump_dir)
        # print(study_load.ee.dm.get_data_dict_values())
        # study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)


if '__main__' == __name__:
    my_test = TestMultiScenarioOfDoeEval()
    my_test.test_06_build_doe_eval_with_nested_proc_selection_through_process_driver()
