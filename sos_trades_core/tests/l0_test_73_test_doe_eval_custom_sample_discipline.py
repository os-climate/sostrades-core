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
import itertools


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

    def setup_usecase_1(self):
        """
        Define a set of data inputs
        """
        input_selection_ABC = {'selected_input': [True, True, True],
                               'full_name': ['stat_A', 'stat_B', 'stat_C']}
        input_selection_ABC = pd.DataFrame(input_selection_ABC)
        output_selection_sum_stat = {'selected_output': [True],
                                     'full_name': ['sum_stat']}
        output_selection_sum_stat = pd.DataFrame(output_selection_sum_stat)
        my_doe_algo = 'CustomDOE'
        my_dict_of_vec = {}
        my_dict_of_vec['stat_A'] = [2, 7]
        my_dict_of_vec['stat_B'] = [2]
        my_dict_of_vec['stat_C'] = [3, 4, 8]
        vect_list = [my_dict_of_vec[elem] for elem in my_dict_of_vec.keys()]
        my_sample = list(itertools.product(*vect_list))
        my_res = np.array(my_sample)
        custom_samples_df = pd.DataFrame(my_res, columns=my_dict_of_vec.keys())

        ######### Fill the dictionary for dm   ####
        values_dict = {}

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_ABC
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_sum_stat
        values_dict[f'{self.study_name}.DoE_Eval.custom_samples_df'] = custom_samples_df

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo

        values_dict[f'{self.study_name}.stat_A'] = 2.
        values_dict[f'{self.study_name}.stat_B'] = 2.
        values_dict[f'{self.study_name}.stat_C'] = 3.

        return [values_dict]

    def setup_usecase_2(self):
        """
        Define a set of data inputs
        """
        input_selection_ABC = {'selected_input': [True, True, True],
                               'full_name': ['stat_A', 'stat_B', 'stat_C']}
        input_selection_ABC = pd.DataFrame(input_selection_ABC)
        output_selection_sum_stat = {'selected_output': [True],
                                     'full_name': ['sum_stat']}
        output_selection_sum_stat = pd.DataFrame(output_selection_sum_stat)
        my_doe_algo = 'CustomDOE'
        my_dict_of_vec = {}
        my_dict_of_vec['stat_A'] = [2, 7]
        my_dict_of_vec['stat_B'] = [2]
        my_dict_of_vec['stat_C'] = [3, 4, 8]

        ######### Fill the dictionary for dm   ####
        values_dict = {}

        values_dict[f'{self.study_name}.Combvec.my_dict_of_vec'] = my_dict_of_vec

        values_dict[f'{self.study_name}.DoE_Eval.eval_inputs'] = input_selection_ABC
        values_dict[f'{self.study_name}.DoE_Eval.eval_outputs'] = output_selection_sum_stat

        values_dict[f'{self.study_name}.DoE_Eval.sampling_algo'] = my_doe_algo

        values_dict[f'{self.study_name}.stat_A'] = 2.
        values_dict[f'{self.study_name}.stat_B'] = 2.
        values_dict[f'{self.study_name}.stat_C'] = 3.

        return [values_dict]

    def test_01_deo_sum_stat(self):
        '''
        Test the creation of the custom doe_eval on a sumstat discipline
        '''
        print('test_01_deo_sum_stat')
        mod_path = 'sos_trades_core.sos_wrapping.test_discs.sum_stat.Sumstat'
        disc_name = 'Sumstat'
        disc_builder = self.exec_eng.factory.get_builder_from_module(
            disc_name, mod_path)
        builder_list = [disc_builder]
        self.exec_eng.ns_manager.add_ns('ns_doe_eval', 'MyStudy.DoE_Eval')
        self.exec_eng.ns_manager.add_ns('ns_sum_stat', 'MyStudy')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            'DoE_Eval', 'doe_eval', builder_list)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        values_dict = self.setup_usecase_1()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.display_treeview_nodes(True)
        # print(self.exec_eng.dm.get_data_dict_values())
        # run
        self.exec_eng.execute()
        print(self.exec_eng.dm.get_data(
            f'{self.study_name}.DoE_Eval.custom_samples_df')['value'])
        print(self.exec_eng.dm.get_data(
            f'{self.study_name}.sum_stat_dict')['value'])

    def test_02_combvect_chained_with_deo_sum_stat(self):
        '''
        Test of the creation of the coupling of a combvect discipline
         (that generates a custom sample) and a custom doe_eval on a sumstat discipline
        '''
        print('test_02_combvect_chained_with_deo_sum_stat')
        disc_name_samples_gene = 'Combvec'
        mod_path_samples_gene = 'sos_trades_core.sos_wrapping.test_discs.combvec.Combvec'
        sc_name = 'DoE_Eval'
        disc_name = 'Sumstat'
        mod_path = 'sos_trades_core.sos_wrapping.test_discs.sum_stat.Sumstat'
        disc_builder_samples_gene = self.exec_eng.factory.get_builder_from_module(
            disc_name_samples_gene, mod_path_samples_gene)
        self.exec_eng.ns_manager.add_ns(
            'ns_doe_eval', f'{self.exec_eng.study_name}.{sc_name}')
        disc_builder = self.exec_eng.factory.get_builder_from_module(
            disc_name, mod_path)
        builder_list = [disc_builder]
        self.exec_eng.ns_manager.add_ns(
            'ns_sum_stat', f'{self.exec_eng.study_name}')
        doe_eval_builder = self.exec_eng.factory.create_evaluator_builder(
            f'{sc_name}', 'doe_eval', builder_list)
        doe_eval_builder_from_combvec = [
            disc_builder_samples_gene, doe_eval_builder]
        self.exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder_from_combvec)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        values_dict = self.setup_usecase_2()[0]
        self.exec_eng.load_study_from_input_dict(values_dict)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.display_treeview_nodes(True)
        # print(self.exec_eng.dm.get_data_dict_values())
        # run
        self.exec_eng.execute()
        print(self.exec_eng.dm.get_data(
            f'{self.study_name}.Combvec.my_dict_of_vec')['value'])
        print(self.exec_eng.dm.get_data(
            f'{self.study_name}.DoE_Eval.custom_samples_df')['value'])
        print(self.exec_eng.dm.get_data(
            f'{self.study_name}.sum_stat_dict')['value'])
