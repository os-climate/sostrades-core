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
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join
import pandas as pd
from numpy import array

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_sample_generator_simple.usecase_without_ref import \
    Study


class TestMultiScenario(unittest.TestCase):
    """
    SoSMultiScenario test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sostrades_core.sos_processes.test'
        self.base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

        # reference var values
        self.x1 = 2.
        self.a1 = 3
        self.constant = 3
        self.power = 2
        self.b1 = 4
        self.b2 = 2
        self.z1 = 1.2
        self.z2 = 1.5

        self.b3 = 1
        self.z3 = 1.8

        self.power1 = 1
        self.constant1 = 4
        self.power2 = 3

        self.scenario_list = scenario_list = ['scenario_1', 'scenario_2', 'scenario_3']
        self.subprocess_inputs_to_check = []

    def setUp_cp(self):
        self.sampling_generation_mode_cp = 'at_configuration_time'
        # self.sampling_generation_mode_cp = 'at_run_time'

        self.study_name_cp = 'cp'
        self.sampling_method_cp = 'cartesian_product'

        dict_of_list_values = {
            'Disc1.b': [self.b1, self.b2],
            'z': [self.z1, self.z2]
        }
        list_of_values_b_z = [[], dict_of_list_values['Disc1.b'],
                              [], [], dict_of_list_values['z']]

        input_selection_cp_b_z = {'selected_input': [False, True, False, False, True],
                                  'full_name': ['', 'Disc1.b', '', '', 'z'],
                                  'list_of_values': list_of_values_b_z
                                  }
        self.input_selection_cp_b_z = pd.DataFrame(input_selection_cp_b_z)

        dict_of_list_values_3 = {
            'Disc1.b': [self.b1, self.b2, self.b3],
            'z': [self.z1, self.z2, self.z3]
        }
        list_of_values_b_z_3 = [[], dict_of_list_values_3['Disc1.b'],
                                [], [], dict_of_list_values_3['z']]

        input_selection_cp_b_z_3 = {'selected_input': [False, True, False, False, True],
                                    'full_name': ['', 'Disc1.b', '', '', 'z'],
                                    'list_of_values': list_of_values_b_z_3
                                    }
        self.input_selection_cp_b_z_3 = pd.DataFrame(input_selection_cp_b_z_3)

        dict_of_list_values_b_z_p = {
            'Disc1.b': [self.b1, self.b2],
            'z': [self.z1, self.z2],
            'Disc3.power': [self.power1, self.power2]
        }
        list_of_values_b_z_p = [[], dict_of_list_values_b_z_p['Disc1.b'],
                                [], [], dict_of_list_values_b_z_p['z'], dict_of_list_values_b_z_p['Disc3.power']]

        input_selection_cp_b_z_p = {'selected_input': [False, True, False, False, True, True],
                                    'full_name': ['', 'Disc1.b', '', '', 'z', 'Disc3.power'],
                                    'list_of_values': list_of_values_b_z_p
                                    }
        self.input_selection_cp_b_z_p = pd.DataFrame(input_selection_cp_b_z_p)

    def setUp_cp_sellar(self):
        self.sampling_generation_mode_cp = 'at_configuration_time'
        # self.sampling_generation_mode_cp = 'at_run_time'

        self.study_name_cp = 'cp'
        self.sampling_method_cp = 'cartesian_product'

        dict_of_list_values = {
            'x': [array([3.]), array([4.])],
            'z': [array([-10., 0.])]
        }

        list_of_values = [[], dict_of_list_values['x'],
                          [], [], dict_of_list_values['z']]

        input_selection_cp_x_z = {'selected_input': [False, True, False, False, True],
                                  'full_name': ['Eval.SellarProblem.local_dv', 'x', 'y_1',
                                                'y_2',
                                                'z'],
                                  'list_of_values': list_of_values
                                  }
        self.input_selection_cp_x_z = pd.DataFrame(input_selection_cp_x_z)

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_standalone_simple_sample_generator(self):
        """
        Checks a standalone simple sample generator that takes 'scenario_names' and 'eval_inputs' and generates an
        empty dataframe with the corresponding lines and columns and all scenarios selected.
        """
        sg_builder = self.exec_eng.factory.create_sample_generator('SampleGenerator')
        self.exec_eng.ns_manager.add_ns('ns_sampling', f'{self.exec_eng.study_name}.SampleGenerator')
        self.exec_eng.factory.set_builders_to_coupling_builder([sg_builder])
        self.exec_eng.configure()

        # setup the driver and the sample generator jointly
        dict_values = {}
        sce_names = ['a', 'b', 'c']
        var_names = ['y_2', 'z']
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        dict_values[f'{self.study_name}.SampleGenerator.scenario_names'] = sce_names
        dict_values[f'{self.study_name}.SampleGenerator.eval_inputs'] = pd.DataFrame({'selected_input': [True, True, False],
                                                                                      'full_name': var_names + ['blabla']})
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(f'{self.study_name}.SampleGenerator.generated_samples')

        self.assertEqual(sce_names, samples_df['scenario_name'].values.tolist())
        self.assertEqual(var_names, samples_df.columns[2:].tolist())
        self.assertEqual([True for _ in sce_names], samples_df['selected_scenario'].values.tolist())

    def test_02_multiscenario_with_sample_generator_input_var(self):
        # # simple 2-disc process
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario':[True],
                                                                                     'scenario_name':['reference']}) # TODO: to be removed when default build reference
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values = {}
        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.SampleGenerator.sampling_generation_mode'), 'at_configuration_time')

        eval_inputs = self.exec_eng.dm.get_value(f'{self.study_name}.multi_scenarios.eval_inputs')

        set_subprocess_inputs = set(eval_inputs['full_name'])
        for var in self.subprocess_inputs_to_check:
            self.assertIn(var, set_subprocess_inputs)

        eval_inputs.loc[eval_inputs['full_name'] == 'Disc1.b', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] == 'z', ['selected_input']] = True

        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['a', 'b', 'c', 'd']

        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs'] = eval_inputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_names'] = scenario_list

        self.exec_eng.load_study_from_input_dict(dict_values)

        for scenario in scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.z'] = self.z1

        # activate some of the scenarios, deactivated by default
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        samples_df['selected_scenario'] = [True, True, False, True]
        samples_df['Disc1.b'] = [self.b1, self.b1, self.b2, self.b2]
        samples_df['z'] = [self.z1, self.z2, self.z1, self.z2]
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        ## flatten_subprocess
        # ms_disc = self.exec_eng.dm.get_disciplines_with_name(
        #     'MyCase.multi_scenarios')[0]
        # ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        # self.assertEqual(ms_sub_disc_names, ['scenario_1',
        #                                      'scenario_2',
        #                                      'scenario_4'])

        self.exec_eng.execute()

        y1, o1 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z1 ** self.power)
        y2, o2 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z2 ** self.power)
        y4, o4 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z2 ** self.power)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.a.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.b.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.d.y'), y4)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.a.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.b.o'), o2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.d.o'), o4)


if '__main__' == __name__:
    cls = TestMultiScenario()
    cls.setUp()
    cls.test_05_dump_and_load_after_execute_with_2_trade_vars()
