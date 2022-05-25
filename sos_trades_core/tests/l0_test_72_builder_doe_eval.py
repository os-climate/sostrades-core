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

    def setup_usecase(self):
        """
        Define a set of data inputs
        """
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

        input_selection_xy = {'selected_input': [True, True, False, False, False, False, False],
                              'full_name': ['DoE_Eval.Hessian.x', 'DoE_Eval.Hessian.y',
                                            'DoE_Eval.Hessian.ax2', 'DoE_Eval.Hessian.by2', 'DoE_Eval.Hessian.cx',
                                            'DoE_Eval.Hessian.dy', 'DoE_Eval.Hessian.exy']}
        input_selection_xy = pd.DataFrame(input_selection_xy)

        output_selection_z = {'selected_output': [True],
                              'full_name': ['DoE_Eval.Hessian.z']}
        output_selection_z = pd.DataFrame(output_selection_z)

        dspace_dict_xy = {'variable': ['x', 'y'],
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
        self.exec_eng.execute()

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
        self.exec_eng.execute()

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
        self.exec_eng.display_treeview_nodes()

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
        dict_values['MyStudy.DoE_Eval.repo_of_processes'] = repo
        dict_values['MyStudy.DoE_Eval.process_folder_name'] = mod_id
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # provide inputs to the set doe with disciplines
        values_dict = self.setup_usecase()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        # run
        self.exec_eng.execute()
