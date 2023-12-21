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
        self.subprocess_inputs_to_check = ['Disc1.b', 'z', 'Disc3.constant', 'Disc3.power', 'a', 'x', 'z']

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
        Checks a standalone simple sample generator that takes 'eval_inputs' and generates an
        empty dataframe with the corresponding lines and columns and all scenarios selected.
        """
        sg_builder = self.exec_eng.factory.create_sample_generator('SampleGenerator')
        self.exec_eng.ns_manager.add_ns('ns_sampling', f'{self.exec_eng.study_name}.SampleGenerator')
        self.exec_eng.factory.set_builders_to_coupling_builder([sg_builder])
        self.exec_eng.configure()

        # setup the driver and the sample generator jointly
        dict_values = {}
        sce_names = ['a', 'b', 'c']
        selected_sce = [True, False, True]

        var_names = ['y_2', 'z']
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(f'{self.study_name}.SampleGenerator.samples_df')

        ref_scenario = ['Reference Scenario']
        self.assertEqual(['selected_scenario', 'scenario_name'], samples_df.columns.tolist())
        self.assertEqual(ref_scenario, samples_df['scenario_name'].values.tolist())
        self.assertEqual([True for _ in ref_scenario], samples_df['selected_scenario'].values.tolist())

        # modify samples_df and save
        samples_df = pd.DataFrame({'selected_scenario': selected_sce,
                                   'scenario_name': sce_names})
        dict_values[f'{self.study_name}.SampleGenerator.samples_df'] = samples_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        # modify eval_inputs and save
        dict_values[f'{self.study_name}.SampleGenerator.eval_inputs'] = pd.DataFrame(
            {'selected_input': [True, True, False],
             'full_name': var_names + ['blabla']})
        dict_values[f'{self.study_name}.SampleGenerator.overwrite_samples_df'] = True
        self.exec_eng.load_study_from_input_dict(dict_values)

        # check the columns have been added
        samples_df = self.exec_eng.dm.get_value(f'{self.study_name}.SampleGenerator.samples_df')
        self.assertEqual(var_names, samples_df.columns[2:].tolist())
        #

    def test_02_multiscenario_with_sample_generator_input_var(self):

        # simple 2-disc process
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # setup the driver and the sample generator mode
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        self.exec_eng.load_study_from_input_dict(dict_values)

        # check that simple sample generator forces config time sampling
        dict_values = {}
        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.SampleGenerator.sampling_generation_mode'), 'at_configuration_time')

        # check that eval_inputs possible values has been properly loaded
        eval_inputs = self.exec_eng.dm.get_value(f'{self.study_name}.SampleGenerator.eval_inputs')
        set_subprocess_inputs = set(eval_inputs['full_name'])
        for var in self.subprocess_inputs_to_check:
            self.assertIn(var, set_subprocess_inputs)

        # check that, when selecting some eval_inputs, the columns appear in samples_df
        eval_inputs.loc[eval_inputs['full_name'] == 'Disc1.b', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] == 'z', ['selected_input']] = True

        dict_values[f'{self.study_name}.SampleGenerator.eval_inputs'] = eval_inputs
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        self.assertEqual(list(samples_df.columns), ['selected_scenario',
                                                    'scenario_name',
                                                    'Disc1.b',
                                                    'z'])

        # modify samples_df to actually generate the scenarios
        scenario_list = ['a', 'b', 'c', 'd']
        selected_scenario = [True, True, False, True]
        samples_df = pd.DataFrame({'selected_scenario': selected_scenario,
                                   'scenario_name': scenario_list,
                                   'Disc1.b': [self.b1, self.b1, self.b2, self.b2],
                                   'z': [self.z1, self.z2, self.z1, self.z2],
                                   })
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ SampleGenerator\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ a\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3\n' \
                 '\t\t|_ b\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3\n' \
                 '\t\t|_ d\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())

        # manually configure scenarios reference values
        for scenario, sel in zip(scenario_list, selected_scenario):
            if sel:
                dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
                dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
                dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
                dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
        self.exec_eng.load_study_from_input_dict(dict_values)

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

    def test_03_switch_iotype_samples_df(self):
        """
        Checks if switching io type for samples_df is OK in the DM and the treeview
        """
        # simple 2-disc process
        repo_name = self.repo + ".tests_driver_eval.mono"
        proc_name = 'test_mono_driver_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # setup the driver and the sample generator mode
        dict_values = {}
        dict_values[f'{self.study_name}.Eval.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'doe_algo'
        dict_values[f'{self.study_name}.SampleGenerator.sampling_generation_mode'] = 'at_run_time'
        self.exec_eng.load_study_from_input_dict(dict_values)

        # self.assertEqual(self.exec_eng.dm.get_value(
        #     f'{self.study_name}.SampleGenerator.sampling_generation_mode'), 'at_run_time')

        # only 1 samples_df exists to connect results from sample generator with driver
        self.assertEqual(len(self.exec_eng.dm.get_all_namespaces_from_var_name('samples_df')), 1)
        samples_df_data = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df')

        disc_dependencies = samples_df_data['disciplines_dependencies']
        # Sample generator and Driver have samples_df in their data_io
        self.assertEqual(len(disc_dependencies), 2)
        # samples_df is out since we are in run_mode and sample generator computes samples_df
        self.assertEqual(samples_df_data['io_type'], 'out')

        # switch to simple
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'simple'
        self.exec_eng.load_study_from_input_dict(dict_values)

        # we are at configuration_time
        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.SampleGenerator.sampling_generation_mode'), 'at_configuration_time')

        # only 1 samples_df exists to connect results from sample generator with driver
        self.assertEqual(len(self.exec_eng.dm.get_all_namespaces_from_var_name('samples_df')), 1)

        samples_df_data = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df')

        disc_dependencies = samples_df_data['disciplines_dependencies']
        # Sample generator and Driver have samples_df in their data_in
        self.assertEqual(len(disc_dependencies), 2)

        # the two disc have sampels_df as an input then it should be in in the dm (check the treeview)
        self.assertEqual(samples_df_data['io_type'], 'in')


if '__main__' == __name__:
    cls = TestMultiScenario()
    cls.setUp()
    cls.test_02_multiscenario_with_sample_generator_input_var()
