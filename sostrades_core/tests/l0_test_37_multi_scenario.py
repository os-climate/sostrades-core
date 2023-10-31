'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/02 Copyright 2023 Capgemini

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

    def test_01_multiscenario_with_sample_generator_cp(self):
        # # simple 2-disc process NOT USING nested scatters
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_sample_generator_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # get the sample generator inputs
        self.setUp_cp()
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'cartesian_product'
        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp'] = self.input_selection_cp_b_z
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.samples_df')['scenario_name'].values.tolist(), ['scenario_1',
                                                                                    'scenario_2',
                                                                                    'scenario_3',
                                                                                    'scenario_4'])

        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_4']
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
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.y'), y4)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.o'), o4)

    def test_02_multiscenario_with_sample_generator_cp_sellar(self):
        # # simple 2-disc process NOT USING nested scatters
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_with_sample_option_sellar'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # get the sample generator inputs
        self.setUp_cp_sellar()
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.Eval.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'cartesian_product'
        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values[f'{self.study_name}.Eval.eval_inputs_cp'] = self.input_selection_cp_x_z
        self.exec_eng.load_study_from_input_dict(dict_values)

        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.Sellar_Problem.local_dv'] = 10.
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.y_1'] = array([1.])
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.y_2'] = array([1.])
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

    def test_03_multiscenario_with_sample_generator_cp_sellar_study(self):
        # # simple 2-disc process NOT USING nested scatters

        from os.path import join, dirname
        from sostrades_core.study_manager.base_study_manager import BaseStudyManager
        ref_dir = join(dirname(__file__), 'data')
        dump_dir = join(ref_dir, 'dump_load_cache')

        self.study_name = 'MyStudy'
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_with_sample_option_sellar'

        # get the sample generator inputs
        self.setUp_cp_sellar()

        study_dump = BaseStudyManager(
            repo_name, proc_name, self.study_name)
        study_dump.set_dump_directory(dump_dir)
        study_dump.load_data()

        ################ Start checks ##########################
        self.ns = f'{self.study_name}'

        self.exec_eng = study_dump.ee

        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.Eval.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'cartesian_product'
        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values[f'{self.study_name}.Eval.eval_inputs_cp'] = self.input_selection_cp_x_z
        study_dump.load_data(from_input_dict=dict_values)

        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.Sellar_Problem.local_dv'] = 10.
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.y_1'] = array([1.])
            dict_values[self.study_name + '.Eval.' +
                        scenario + '.y_2'] = array([1.])
        study_dump.load_data(from_input_dict=dict_values)
        study_dump.dump_data(dump_dir)
        study_dump.run()

        ########################
        study_load = BaseStudyManager(
            repo_name, proc_name, self.study_name)
        study_load.load_data(from_path=dump_dir)
        # print(study_load.ee.dm.get_data_dict_values())
        study_load.run()
        from shutil import rmtree
        rmtree(dump_dir)

    def test_04_multi_scenario_from_process_with_basic_config_from_usecase(self):
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_sample_generator_simple'
        builder_process = self.exec_eng.factory.get_builder_from_process(
            repo_name, proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_process)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        usecase = Study(execution_engine=self.exec_eng)
        usecase.study_name = self.namespace
        values_dict = usecase.setup_usecase()

        self.exec_eng.load_study_from_input_dict(values_dict[0])

        scenario_list = ['scenario_1', 'scenario_2', 'scenario_4']
        dict_values = {}
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
        y3, o3 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z1 ** self.power)
        y4, o4 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z2 ** self.power)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.y'), y4)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.o'), o4)

    def test_05_multi_scenario_from_process_with_basic_config_from_usecase_and_with_ref(self):
        # FIXME: there seems to be a problem with reference instance + flatten_subprocess
        from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver_sample_generator_simple.usecase_without_ref import \
            Study

        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_sample_generator_simple'
        builder_process = self.exec_eng.factory.get_builder_from_process(
            repo_name, proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            builder_process)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        usecase = Study(execution_engine=self.exec_eng)
        usecase.study_name = self.namespace
        values_dict = usecase.setup_usecase()

        self.exec_eng.load_study_from_input_dict(values_dict[0])

        # activate some of the scenarios, deactivated by default
        dict_values = {}
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        samples_df['selected_scenario'] = [True, True, False, True]
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = True
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'linked_mode'
        # self.exec_eng.load_study_from_input_dict(dict_values)

        # reference var values
        self.x = 2.
        self.a = 3
        self.b = 8
        self.z = 12
        # configure the Reference scenario
        # Non-trade variables (to propagate)
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.a'] = self.a
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.x'] = self.x
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.Disc3.constant'] = self.constant
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.Disc3.power'] = self.power
        # Trade variables reference (not to propagate)
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.Disc1.b'] = self.b
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.z'] = self.z
        self.exec_eng.load_study_from_input_dict(dict_values)

        # Check trades variables are ok and that non-trade variables have been
        # propagated to other scenarios
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.Disc1.b'), self.b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.Disc1.b'), self.b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_4.Disc1.b'), self.b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.z'), self.z1)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.z'), self.z2)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_4.z'), self.z2)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_4.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.x'), self.x)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.x'), self.x)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_4.x'), self.x)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_4']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.constant'), self.constant)
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.power'), self.power)

        # Since reference values have been propagated, it should be able to
        # already execute.
        self.exec_eng.execute()

        # Change non-trade variable value from reference and check it has
        # induced a reconfiguration and re-propagation
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.Disc3.constant'] = 23
        self.exec_eng.load_study_from_input_dict(dict_values)
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.constant'), 23)

        ## flatten_subprocess
        # ms_disc = self.exec_eng.dm.get_disciplines_with_name(
        #     'MyCase.multi_scenarios')[0]
        # ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        # self.assertEqual(ms_sub_disc_names, [
        #     'scenario_1', 'scenario_2', 'scenario_4', 'ReferenceScenario'])

        # Now, check that, since we are in LINKED_MODE, that the non-trade variables from non-reference scenarios have
        # 'editable' in False.
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_4.a', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_4.x', 'editable'), False)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), False)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), False)

        # Now, change to REFERENCE_MODE to COPY_MODE and check that the non-trade variables from non-reference scenarios have
        # 'editable' in True.
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'copy_mode'
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_4.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_4.x', 'editable'), True)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), True)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), True)

    def test_06_consecutive_configure(self):
        # # simple 2-disc process NOT USING nested scatters
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_sample_generator_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # get the sample generator inputs
        self.setUp_cp()
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'cartesian_product'
        self.exec_eng.load_study_from_input_dict(dict_values)

        # same input selection as first test, all scenarios activated
        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp'] = self.input_selection_cp_b_z
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        scenario_names = ['scenario_1',
                          'scenario_2', 'scenario_3', 'scenario_4']
        scenario_vars = ['Disc1.b', 'z']
        self.assertEqual(
            samples_df['scenario_name'].values.tolist(), scenario_names)
        self.assertEqual(samples_df['Disc1.b'].values.tolist(), [self.b1,
                                                                 self.b1,
                                                                 self.b2,
                                                                 self.b2])
        self.assertEqual(samples_df['z'].values.tolist(), [self.z1,
                                                           self.z2,
                                                           self.z1,
                                                           self.z2])
        samples_df['selected_scenario'] = True
        self.exec_eng.load_study_from_input_dict(dict_values)
        ## flatten_subprocess
        # ms_disc = self.exec_eng.dm.get_disciplines_with_name(
        #     'MyCase.multi_scenarios')[0]
        # ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        # self.assertEqual(ms_sub_disc_names, scenario_names)
        for sc in scenario_names:
            for var in scenario_vars:
                self.assertEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.' + sc + '.' + var),
                                 samples_df[samples_df['scenario_name'] == sc].iloc[0][var])

        # deactivate the eval inputs of the cartesian product and check that
        # the scenarios disappear
        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp']['selected_input'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        gen = self.exec_eng.root_process.proxy_disciplines[2]
        print(gen.mdo_discipline_wrapp.wrapper.samples_gene_df)
        self.assertTrue(samples_df.empty)

        # change the trade variables values
        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp'] = self.input_selection_cp_b_z_3
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        scenario_names = ['scenario_1', 'scenario_2', 'scenario_3',
                          'scenario_4', 'scenario_5', 'scenario_6',
                          'scenario_7', 'scenario_8', 'scenario_9']
        scenario_vars = ['Disc1.b', 'z']
        self.assertEqual(
            samples_df['scenario_name'].values.tolist(), scenario_names)
        self.assertEqual(samples_df['Disc1.b'].values.tolist(), [self.b1,
                                                                 self.b1,
                                                                 self.b1,
                                                                 self.b2,
                                                                 self.b2,
                                                                 self.b2,
                                                                 self.b3,
                                                                 self.b3,
                                                                 self.b3
                                                                 ])
        self.assertEqual(samples_df['z'].values.tolist(), [self.z1,
                                                           self.z2,
                                                           self.z3,
                                                           self.z1,
                                                           self.z2,
                                                           self.z3,
                                                           self.z1,
                                                           self.z2,
                                                           self.z3])
        samples_df['selected_scenario'] = True
        self.exec_eng.load_study_from_input_dict(dict_values)
        ## flatten_subprocess
        # ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        # self.assertEqual(ms_sub_disc_names, scenario_names)
        for sc in scenario_names:
            for var in scenario_vars:
                self.assertEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.' + sc + '.' + var),
                                 samples_df[samples_df['scenario_name'] == sc].iloc[0][var])

        # change the trade variables themselves
        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp'] = self.input_selection_cp_b_z_p
        self.exec_eng.load_study_from_input_dict(dict_values)
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        scenario_names = ['scenario_1', 'scenario_2', 'scenario_3',
                          'scenario_4', 'scenario_5', 'scenario_6',
                          'scenario_7', 'scenario_8']
        scenario_vars = ['Disc1.b', 'z', 'Disc3.power']
        self.assertEqual(samples_df['Disc1.b'].values.tolist(), [self.b1,
                                                                 self.b1,
                                                                 self.b1,
                                                                 self.b1,
                                                                 self.b2,
                                                                 self.b2,
                                                                 self.b2,
                                                                 self.b2
                                                                 ])
        self.assertEqual(samples_df['z'].values.tolist(), [self.z1,
                                                           self.z1,
                                                           self.z2,
                                                           self.z2,
                                                           self.z1,
                                                           self.z1,
                                                           self.z2,
                                                           self.z2,
                                                           ])

        self.assertEqual(samples_df['Disc3.power'].values.tolist(), [self.power1,
                                                                     self.power2,
                                                                     self.power1,
                                                                     self.power2,
                                                                     self.power1,
                                                                     self.power2,
                                                                     self.power1,
                                                                     self.power2,
                                                                     ])
        samples_df['selected_scenario'] = True
        self.exec_eng.load_study_from_input_dict(dict_values)
        ## flatten_subprocess
        # ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        # self.assertEqual(ms_sub_disc_names, scenario_names)
        for sc in scenario_names:
            for var in scenario_vars:
                self.assertEqual(self.exec_eng.dm.get_value('MyCase.multi_scenarios.' + sc + '.' + var),
                                 samples_df[samples_df['scenario_name'] == sc].iloc[0][var])

        # configure the reference values...
        private_values = {}

        for scenario in scenario_names:
            private_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            private_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            private_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
        self.exec_eng.load_study_from_input_dict(private_values)

        self.exec_eng.execute()
        for _, sc_row in samples_df.iterrows():
            scenario_name = sc_row['scenario_name']
            b = sc_row['Disc1.b']
            z = sc_row['z']
            power = sc_row['Disc3.power']
            y, o = (self.a1 * self.x1 + b, self.constant + z ** power)
            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.' + scenario_name + '.y'), y)
            self.assertEqual(self.exec_eng.dm.get_value(
                'MyCase.multi_scenarios.' + scenario_name + '.o'), o)

    def test_07_dump_and_load_after_execute_with_2_trade_vars(self):
        # # simple 2-disc process NOT USING nested scatters
        repo_name = self.repo + ".tests_driver_eval.multi"
        proc_name = 'test_multi_driver_sample_generator_simple'
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # get the sample generator inputs
        self.setUp_cp()
        # setup the driver and the sample generator jointly
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.with_sample_generator'] = True
        dict_values[f'{self.study_name}.SampleGenerator.sampling_method'] = 'cartesian_product'
        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values[f'{self.study_name}.multi_scenarios.eval_inputs_cp'] = self.input_selection_cp_b_z
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.samples_df')['scenario_name'].values.tolist(), ['scenario_1',
                                                                                    'scenario_2',
                                                                                    'scenario_3',
                                                                                    'scenario_4'])

        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_4']
        dict_values = {}
        for scenario in scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
        # activate some of the scenarios, deactivated by default
        samples_df = self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.samples_df')
        samples_df['selected_scenario'] = [True, True, False, True]
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
        y3, o3 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z1 ** self.power)
        y4, o4 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z2 ** self.power)

        dump_dir = join(self.root_dir, self.namespace)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.namespace)
        builders = exec_eng2.factory.get_builder_from_process(
            repo_name, proc_name)
        exec_eng2.factory.set_builders_to_coupling_builder(builders)
        exec_eng2.configure()

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.y'), y4)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.o'), o4)

        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.y'), y4)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_4.o'), o4)

        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(
            join(self.root_dir, self.namespace))


if '__main__' == __name__:
    cls = TestMultiScenario()
    cls.setUp()
    cls.test_05_dump_and_load_after_execute_with_2_trade_vars()
