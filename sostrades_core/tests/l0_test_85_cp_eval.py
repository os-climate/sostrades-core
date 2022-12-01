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


class TestSoSDOEScenario(unittest.TestCase):

    def setUp(self):
        self.sampling_generation_mode_cp = 'at_configuration_time'
        # self.sampling_generation_mode_cp = 'at_run_time'

        self.study_name = 'cp'
        self.ns = f'{self.study_name}'
        self.sampling_method_cp = 'cartesian_product'

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

        input_selection_cp_x = {'selected_input': [False, True, False, False, True],
                                'full_name': ['DoEEval.subprocess.Sellar_Problem.local_dv', 'x', 'y_1',
                                              'y_2',
                                              ''],
                                'list_of_values': list_of_values_x_z
                                }
        self.input_selection_cp_x = pd.DataFrame(input_selection_cp_x)

        output_selection_obj = {'selected_output': [False, False, True, False, False],
                                'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj = pd.DataFrame(output_selection_obj)

        output_selection_obj_y1_y2 = {'selected_output': [False, False, True, True, True],
                                      'full_name': ['c_1', 'c_2', 'obj', 'y_1', 'y_2']}
        self.output_selection_obj_y1_y2 = pd.DataFrame(
            output_selection_obj_y1_y2)

        self.repo = 'sostrades_core.sos_processes.test'

    def _test_1_cp_eval_mono_instance(self):
        """ We test that the number of samples generated by the fullfact algorithm is the theoretical expected number
        Pay attention to the fact that an additional sample (the reference one ) is added
        """
        #FIXME: CP + Eval mono-instance case not covered yet...

        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        proc_name = "test_sellar_generator_eval"
        doe_eval_builder = factory.get_builder_from_process(repo=self.repo,
                                                            mod_id=proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(
            doe_eval_builder)

        exec_eng.configure()
        builder_mode_input = {f'{self.ns}.Eval.builder_mode': 'mono_instance'}
        exec_eng.load_study_from_input_dict(builder_mode_input)

        # -- set up disciplines in Scenario
        disc_dict = {}
        # Cartesian Products + Eval mono-instance inputs
        disc_dict[f'{self.ns}.SampleGenerator.sampling_method'] = self.sampling_method_cp
        disc_dict[f'{self.ns}.eval_inputs'] = self.input_selection_cp_x_z
        disc_dict[f'{self.ns}.eval_outputs'] = self.output_selection_obj_y1_y2
        exec_eng.load_study_from_input_dict(disc_dict)

        # Sellar inputs
        local_dv = 10.
        values_dict = {}
        # array([1.])
        values_dict[f'{self.ns}.x'] = array([1.])
        values_dict[f'{self.ns}.y_1'] = array([1.])
        values_dict[f'{self.ns}.y_2'] = array([1.])
        values_dict[f'{self.ns}.z'] = array([1., 1.])
        values_dict[f'{self.ns}.Eval.subprocess.Sellar_Problem.local_dv'] = local_dv
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        exp_tv_list = [f'Nodes representation for Treeview {self.ns}',
                       f'|_ {self.ns}',
                       f'\t|_ SampleGenerator',
                       '\t|_ Eval',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Sellar_Problem',
                       '\t\t\t|_ Sellar_2',
                       '\t\t\t|_ Sellar_1']
        exp_tv_str = '\n'.join(exp_tv_list)
        exec_eng.display_treeview_nodes(True)
        assert exp_tv_str == exec_eng.display_treeview_nodes()
        eval_disc = exec_eng.dm.get_disciplines_with_name(
            f'{self.ns}.Eval')[0].mdo_discipline_wrapp.mdo_discipline.sos_wrapp

        eval_disc_samples = eval_disc.get_sosdisc_outputs(
            'samples_inputs_df')
        print('fqsfsdqfqsf')

