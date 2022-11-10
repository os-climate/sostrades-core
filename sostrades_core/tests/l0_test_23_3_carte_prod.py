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
import logging
from logging import Handler
from time import time

from pandas._testing import assert_frame_equal

from gemseo.algos.doe.doe_factory import DOEFactory

"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for doe scenario
"""

import unittest
from numpy import array
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sample_generators.doe_sample_generator import DoeSampleGenerator
import os
from os.path import dirname, join


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestCartesianProduct(unittest.TestCase):

    def setUp(self):
        self.repo = 'sostrades_core.sos_processes.test'
        self.study_name = 'cp'
        self.ns = f'{self.study_name}'

        dict_of_list_values = {
            'x': [0., 3., 4., 5., 7.],
            'y_1': [1.0, 2.0],
            'z': [[-10., 0.], [-5., 4.], [10, 10]]
        }
        list_of_values_x_z = [[], dict_of_list_values['x'],
                              [], [], dict_of_list_values['z']]

        input_selection_cp_x_z = {'selected_input': [False, True, False, False, True],
                                  'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                'y_2',
                                                'z'],
                                  'list_of_values': list_of_values_x_z
                                  }
        self.input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

        list_of_values_x_y_1_z = [[], dict_of_list_values['x'],
                                  dict_of_list_values['y_1'], [], dict_of_list_values['z']]

        input_selection_cp_x_y_1_z = {'selected_input': [False, True, True, False, True],
                                      'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                                    'y_2',
                                                    'z'],
                                      'list_of_values': list_of_values_x_y_1_z
                                      }
        self.input_selection_cp_x_y_1_z = pd.DataFrame(
            input_selection_cp_x_y_1_z)

    def test_1_cartesian_product_execution(self):
        """
        This is a test of the cartesian product wrapper
        """

        exec_eng = ExecutionEngine(self.study_name)

        # mod_list = 'sostrades_core.execution_engine.disciplines_wrappers.doe_wrapper.DoeWrapper'
        # doe_builder = exec_eng.factory.get_builder_from_module('DoE', mod_list)
        # exec_eng.ns_manager.add_ns('ns_doe1', 'doe.DoE')

        proc_name = "test_cartesian_product"
        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        #

        # -- set up disciplines in Scenario
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.CP.sampling_method'] = 'cartesian_product'
        disc_dict[f'{self.ns}.CP.eval_inputs_cp'] = self.input_selection_cp_x_z
        #disc_dict[f'{self.ns}.CP.generated_samples'] = generated_samples

        exec_eng.load_study_from_input_dict(disc_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ cp',
                       f'\t|_ CP']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.display_treeview_nodes(True)

        disc = exec_eng.root_process.proxy_disciplines[0]
        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs_cp')
        print('eval_inputs_cp 2:')
        print(disc_eval_inputs_cp)

        disc_generated_samples = disc.get_sosdisc_inputs(
            'generated_samples')
        print('generated_samples:')
        print(disc_generated_samples)

        exec_eng.execute()

        # disc = exec_eng.dm.get_disciplines_with_name(
        #     'cp.CP')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        disc = exec_eng.root_process.proxy_disciplines[0]

        disc_samples = disc.get_sosdisc_outputs(
            'samples_df')

        print(disc_samples)

        targeted_samples = [
            [0.0, [-10.0, 0.0]],
            [0.0, [-5.0, 4.0]],
            [0.0, [10, 10]],
            [3.0, [-10.0, 0.0]],
            [3.0, [-5.0, 4.0]],
            [3.0, [10, 10]],
            [4.0, [-10.0, 0.0]],
            [4.0, [-5.0, 4.0]],
            [4.0, [10, 10]],
            [5.0, [-10.0, 0.0]],
            [5.0, [-5.0, 4.0]],
            [5.0, [10, 10]],
            [7.0, [-10.0, 0.0]],
            [7.0, [-5.0, 4.0]],
            [7.0, [10, 10]]]

        variable_list = ['x', 'z']
        target_samples_df = pd.DataFrame(
            targeted_samples, columns=variable_list)

    def test_2_cartesian_product_step_by_step_execution(self):
        """
        This is a test of the cartesian product wrapper
        """

        exec_eng = ExecutionEngine(self.study_name)

        # mod_list = 'sostrades_core.execution_engine.disciplines_wrappers.doe_wrapper.DoeWrapper'
        # doe_builder = exec_eng.factory.get_builder_from_module('DoE', mod_list)
        # exec_eng.ns_manager.add_ns('ns_doe1', 'doe.DoE')

        proc_name = "test_cartesian_product"
        doe_builder = exec_eng.factory.get_builder_from_process(repo=self.repo,
                                                                mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            [doe_builder])

        exec_eng.configure()

        #
        # -- set up disciplines in Scenario
        # 1. Input sampling_method
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.CP.sampling_method'] = 'cartesian_product'

        exec_eng.load_study_from_input_dict(disc_dict)

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       '|_ cp',
                       f'\t|_ CP']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == exec_eng.display_treeview_nodes()

        exec_eng.display_treeview_nodes(True)

        # disc = exec_eng.dm.get_disciplines_with_name(
        #     'cp.CP')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp
        disc = exec_eng.root_process.proxy_disciplines[0]

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs_cp')
        print('eval_inputs_cp 1:')
        print(disc_eval_inputs_cp)

        # 2. Input eval_inputs_cp
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.CP.eval_inputs_cp'] = self.input_selection_cp_x_z
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.display_treeview_nodes(True)

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs_cp')
        print('eval_inputs_cp 2:')
        print(disc_eval_inputs_cp)

        disc_generated_samples = disc.get_sosdisc_inputs(
            'generated_samples')
        print('generated_samples:')
        print(disc_generated_samples)

        # 3. Input an updated eval_inputs_cp
        # CP inputs
        disc_dict = {}
        disc_dict[f'{self.ns}.CP.eval_inputs_cp'] = self.input_selection_cp_x_y_1_z
        exec_eng.load_study_from_input_dict(disc_dict)

        exec_eng.display_treeview_nodes(True)

        disc_sampling_method = disc.get_sosdisc_inputs(
            'sampling_method')
        print('sampling__method:')
        print(disc_sampling_method)

        disc_eval_inputs_cp = disc.get_sosdisc_inputs(
            'eval_inputs_cp')
        print('eval_inputs_cp 3:')
        print(disc_eval_inputs_cp)

        disc_generated_samples = disc.get_sosdisc_inputs(
            'generated_samples')
        print('generated_samples:')
        print(disc_generated_samples)

        exec_eng.execute()

        disc_samples = disc.get_sosdisc_outputs(
            'samples_df')

        print(disc_samples)


if '__main__' == __name__:
    cls = TestCartesianProduct()
    cls.setUp()
    cls.test_1_cartesian_product_execution()
