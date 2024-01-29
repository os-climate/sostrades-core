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
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
import pandas as pd
from logging import Handler

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestDriverDataIntegrity(unittest.TestCase):
    """
    Driver Data Integrity test class
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
        self.my_handler = UnitTestHandler()
        self.exec_eng.logger.addHandler(self.my_handler)

        # reference var values
        self.x1 = 2.
        self.a1 = 3
        self.constant = 3
        self.power = 2
        self.b1 = 4
        self.b2 = 2
        self.z1 = 1.2
        self.z2 = 1.5

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def test_01_two_scenarios_with_same_name(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', False, 0], ['scenario_1', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ Reference Scenario\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'

        # Logging only
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        data_integrity_error_message = 'Two scenarios have same names in the samples_df, check the scenario_name column'
        self.assertEqual(check_integrity_msg, data_integrity_error_message)

        runtime_error_message = 'Variable MyCase.multi_scenarios.samples_df : Two scenarios have same names in the samples_df, check the scenario_name column'
        # data integrity Exception
        with self.assertRaises(ValueError) as cm:
            self.exec_eng.execute()
        self.assertTrue(runtime_error_message in str(cm.exception))

    def test_02_two_scenarios_with_same_name_on_2nd_config(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', False, 0], ['scenario_2', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        samples_df['scenario_name'].iloc[2] = 'scenario_1'
        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ Reference Scenario\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'
        # Logging only
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        data_integrity_error_message = 'Two scenarios have same names in the samples_df, check the scenario_name column'
        self.assertEqual(check_integrity_msg, data_integrity_error_message)

        runtime_error_message = 'Variable MyCase.multi_scenarios.samples_df : Two scenarios have same names in the samples_df, check the scenario_name column'
        # data integrity Exception
        with self.assertRaises(ValueError) as cm:
            self.exec_eng.execute()
        self.assertTrue(runtime_error_message in str(cm.exception))

    def test_03_no_scenario_selected(self):
        '''
        Check if no scenario are selected that data integrity is not OK
        Check if empty scenario_df that data integrity is not OK
        '''
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame({'scenario_name': ['scenario_1', 'scenario_2'],
                                   'selected_scenario': False})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        # confgiure
        self.exec_eng.load_study_from_input_dict(dict_values)

        # Still reference scenario because new scenario_df is not ok with data integrity
        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ Reference Scenario\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        data_integrity_error_message = 'You need to select at least one scenario to execute your driver'
        self.assertEqual(check_integrity_msg, data_integrity_error_message)

        ### And now empty dataframe
        samples_df = pd.DataFrame({}, columns=['scenario_name', 'selected_scenario'])
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        # confgiure
        self.exec_eng.load_study_from_input_dict(dict_values)

        # Still reference scenario because new scenario_df is not ok with data integrity
        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ Reference Scenario\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        data_integrity_error_message = 'Your samples_df is empty, the driver cannot be configured'
        self.assertEqual(check_integrity_msg, data_integrity_error_message)

        runtime_error_message = f'Variable MyCase.multi_scenarios.samples_df : {data_integrity_error_message}'
        # data integrity Exception
        with self.assertRaises(ValueError) as cm:
            self.exec_eng.execute()
        self.assertTrue(runtime_error_message in str(cm.exception))

    def test_04_multi_integrity_samples_df_empty(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(columns=['scenario_name', 'selected_scenario'])

        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = 'Your samples_df is empty, the driver cannot be configured'
  
        self.assertEqual(check_integrity_msg, integrity_error_message)

    def test_04_mono_integrity_samples_df_empty(self):
        proc_name = 'test_mono_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(columns=['scenario_name', 'selected_scenario'])

        dict_values = {f'{self.study_name}.Eval.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df',
                                                        'check_integrity_msg')
        check_integrity_msg = check_integrity_msg.split('\n')[0]
        integrity_error_message = 'Your samples_df is empty, the driver cannot be configured'
  
        self.assertEqual(check_integrity_msg, integrity_error_message)

    
    def test_05_mono_integrity_samples_df_no_input(self):
        proc_name = 'test_mono_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True], ['scenario_2', True]],
            columns=['scenario_name', 'selected_scenario'])

        dict_values = {f'{self.study_name}.Eval.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = 'There should be at least one trade variable column in samples_df'
  
        self.assertTrue(check_integrity_msg, integrity_error_message)
    
    def test_06_multi_integrity_samples_df_none_input(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, None]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = "There is a None in the samples_df, check the columns ['Disc1.b'] "
  
        self.assertEqual(check_integrity_msg, integrity_error_message)
    
    def test_06_mono_integrity_samples_df_none_input(self):
        proc_name = 'test_mono_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, None]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.Eval.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = "There is a None in the samples_df, check the columns ['Disc1.b'] "
  
        self.assertEqual(check_integrity_msg, integrity_error_message)

    def test_multi_07_integrity_samples_df_type_input(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, 'a']],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = 'Some value has wrong types in column Disc1.b, the subprocess variable is of type float and all variables in the column should be the same'
  
        self.assertEqual(check_integrity_msg, integrity_error_message)
    
    def test_07_mono_integrity_samples_df_type_input(self):
        proc_name = 'test_mono_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, 'a']],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.Eval.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = 'Some value has wrong types in column Disc1.b, the subprocess variable is of type float and all variables in the column should be the same'
  
        self.assertEqual(check_integrity_msg, integrity_error_message)
    
    def test_08_mono_integrity_no_gather_output(self):
        proc_name = 'test_mono_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.mono"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, '2']],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.Eval.samples_df': samples_df}
        dict_values[f'{self.study_name}.Eval.gather_outputs'] = pd.DataFrame(
            [[False, 'Disc1.b', None]],
            columns=['selected_output', 'full_name', 'output_name'])
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.Eval.gather_outputs',
                                                        'check_integrity_msg')
        integrity_error_message = 'There should be at least one selected output'
  
        self.assertEqual(check_integrity_msg, integrity_error_message)
    
    def test_09_integrity_samples_df_unknown_input(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, '2']],
            columns=['scenario_name', 'selected_scenario', 'Disc1.wrong_input'])
        
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        check_integrity_msg = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.samples_df',
                                                        'check_integrity_msg')
        integrity_error_message = 'The variable Disc1.wrong_input is not in the subprocess eval input values: It cannot be a column of the samples_df '
  
        self.assertEqual(check_integrity_msg, integrity_error_message)

    def test_10_input_not_editable(self):
        proc_name = 'test_multi_driver_simple'
        repo_name = self.repo + ".tests_driver_eval.multi"
        builders = self.exec_eng.factory.get_builder_from_process(repo_name,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        samples_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, '2']],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        # get editable state of Disc1.b variables
        sc1_b_editable = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.scenario_1.Disc1.b', 'editable')
        self.assertFalse(sc1_b_editable, 'Trades variable should be not editable')
        sc2_b_editable = self.exec_eng.dm.get_data(f'{self.study_name}.multi_scenarios.scenario_2.Disc1.b', 'editable')
        self.assertFalse(sc2_b_editable, 'Trades variable should be not editable')

if '__main__' == __name__:
    cls = TestDriverDataIntegrity()
    cls.setUp()
    cls.test_09_integrity_samples_df_unknown_input()
