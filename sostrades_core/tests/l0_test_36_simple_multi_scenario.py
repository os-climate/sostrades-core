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
import numpy as np
from logging import Handler

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager


class UnitTestHandler(Handler):
    """
    Logging handler for UnitTest
    """

    def __init__(self):
        Handler.__init__(self)
        self.msg_list = []

    def emit(self, record):
        self.msg_list.append(record.msg)


class TestSimpleMultiScenario(unittest.TestCase):
    """
    SoSSimpleMultiScenario test class
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

    # NEW EEV4 TESTS
    def test_01_multi_instance_configuration_from_df_without_reference_scenario(self):
        # # simple 2-disc process NOT USING nested scatters
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(self.repo,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # build the scenarios
        dict_values = {}
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        # dict_values[f'{self.study_name}.Eval.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # configure the scenarios
        scenario_list = ['scenario_1', 'scenario_2']

        for scenario in scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power

        self.exec_eng.load_study_from_input_dict(dict_values)

        # configure b from a dataframe
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2'],
                                    'Disc1.b': [self.b1, 1e6, self.b2],
                                    'z': [self.z1, 1e6, self.z2]})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_df')['scenario_name'].values.tolist(), ['scenario_1',
                                                                                     'scenario_W',
                                                                                     'scenario_2'])
        ms_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.multi_scenarios')[0]
        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]

        self.assertEqual(ms_sub_disc_names, ['scenario_1',
                                             'scenario_2'])

        y1 = self.a1 * self.x1 + self.b1
        y2 = self.a1 * self.x1 + self.b2
        o1 = self.constant + self.z1 ** self.power
        o2 = self.constant + self.z2 ** self.power

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

    def test_02_multi_instance_configuration_from_df_with_reference_scenario(self):
        # # simple 2-disc process NOT USING nested scatters
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(self.repo,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        # build the scenarios
        dict_values = {}
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        dict_values[f'{self.study_name}.multi_scenarios.builder_mode'] = 'multi_instance'
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = True
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'linked_mode'
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # reference var values
        self.x = 2.
        self.a = 3
        self.constant = 3
        self.power = 2
        self.b = 8
        self.z = 12

        # configure the Reference scenario
        # Non-trade variables (to propagate)
        dict_values[f'{self.study_name}.multi_scenarios.ReferenceScenario.a'] = self.a
        dict_values[f'{self.study_name}.multi_scenarios.ReferenceScenario.x'] = self.x
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

        # Configure b and z of others scenarios (trade variables)
        scenario_df = pd.DataFrame({'selected_scenario': [True, False, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2'],
                                    'Disc1.b': [self.b1, 1e6, self.b2],
                                    'z': [self.z1, 1e6, self.z2]})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        self.exec_eng.load_study_from_input_dict(dict_values)

        # Check trades variables are ok and that non-trade variables have been
        # propagated to other scenarios
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.Disc1.b'), self.b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.Disc1.b'), self.b2)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.z'), self.z1)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.z'), self.z2)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.x'), self.x)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.x'), self.x)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.constant'), self.constant)
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.power'), self.power)
        self.exec_eng.execute()

        # Change non-trade variable value from reference and check it has
        # induced a reconfiguration and re-propagation
        new_constant_ref = 23
        dict_values[self.study_name +
                    '.multi_scenarios.ReferenceScenario.Disc3.constant'] = new_constant_ref
        self.exec_eng.load_study_from_input_dict(dict_values)
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.constant'), new_constant_ref)

        self.exec_eng.execute()

        # Now, check that, since we are in LINKED_MODE, that the non-trade variables from non-reference scenarios have
        # 'editable' in False.
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), False)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), False)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), False)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), False)

        self.assertEqual(
            self.exec_eng.dm.get_value('MyCase.multi_scenarios.scenario_df')['scenario_name'].values.tolist(),
            ['scenario_1', 'scenario_W', 'scenario_2'])
        ms_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.multi_scenarios')[0]
        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        for sc in ['scenario_1', 'scenario_2']:
            assert sc in ms_sub_disc_names

        # Now, change to REFERENCE_MODE to COPY_MODE and check that the non-trade variables from non-reference scenarios have
        # 'editable' in True.
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'copy_mode'
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), True)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), True)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), True)

        # Now check that after un-instantiating the reference from linked mode,
        # the non-trade variables go back to True.
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'linked_mode'
        self.exec_eng.load_study_from_input_dict(dict_values)
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), True)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), True)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), True)

        # Now check that, having an instantiated reference in linked mode, adding a new scenario, we will find the new
        # scenario with proper propagation and editability state.
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = True
        dict_values[f'{self.study_name}.multi_scenarios.reference_mode'] = 'linked_mode'
        scenario_df = pd.DataFrame({'selected_scenario': [True, True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_W',
                                                      'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        self.exec_eng.load_study_from_input_dict(dict_values)
        # Value propagation
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_W.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.a'), self.a)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_1.x'), self.x)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_W.x'), self.x)
        self.assertEqual(self.exec_eng.dm.get_value(
            self.study_name + '.multi_scenarios.scenario_2.x'), self.x)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_W']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.constant'), new_constant_ref)
            self.assertEqual(self.exec_eng.dm.get_value(self.study_name + '.multi_scenarios.' +
                                                        scenario + '.Disc3.power'), self.power)
        # Editability propagation
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_1.a', 'editable'),
                         False)
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_W.a', 'editable'),
                         False)
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_2.a', 'editable'),
                         False)
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_1.x', 'editable'),
                         False)
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_W.x', 'editable'),
                         False)
        self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.scenario_2.x', 'editable'),
                         False)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_W']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), False)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), False)

        # Now check that un_instantiating the reference from linked_mode, the new added scenario_W comes back to the
        # proper editability state
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_W.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.a', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_1.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_W.x', 'editable'), True)
        self.assertEqual(self.exec_eng.dm.get_data(
            self.study_name + '.multi_scenarios.scenario_2.x', 'editable'), True)
        scenario_list = ['scenario_1', 'scenario_2', 'scenario_W']
        for scenario in scenario_list:
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.constant', 'editable'), True)
            self.assertEqual(self.exec_eng.dm.get_data(self.study_name + '.multi_scenarios.' +
                                                       scenario + '.Disc3.power', 'editable'), True)

    def test_03_consecutive_configure(self):
        # # simple 2-disc process NOT USING nested scatters
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(self.repo,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1]], columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        dict_values = {f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance',
                       f'{self.study_name}.multi_scenarios.scenario_df': scenario_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        ms_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.multi_scenarios')[0]
        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        self.assertEqual(ms_sub_disc_names, ['scenario_1'])

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        self.assertEqual(ms_sub_disc_names, ['scenario_1',
                                             'scenario_2'])

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', False, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        self.assertEqual(ms_sub_disc_names, ['scenario_1'])

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values[self.study_name +
                    '.multi_scenarios.scenario_df'] = scenario_df

        self.exec_eng.load_study_from_input_dict(dict_values)
        ms_sub_disc_names = [d.sos_name for d in ms_disc.proxy_disciplines]
        self.assertEqual(ms_sub_disc_names, ['scenario_1',
                                             'scenario_2'])

        # manually configure the scenarios non-varying values (~reference)
        private_val = {}
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.z'] = self.z1
        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_1.Disc1.b'), self.b1)
        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_2.Disc1.b'), self.b2)

        y1, o1 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z1 ** self.power)
        y2, o2 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z1 ** self.power)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

    def test_04_dump_and_load_after_execute_with_2_trade_vars(self):
        # # simple 2-disc process NOT USING nested scatters
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(
            self.repo, proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()
        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1, self.z1], ['scenario_2', True, self.b2, self.z2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b', 'z'])

        dict_values = {f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance',
                       f'{self.study_name}.multi_scenarios.scenario_df': scenario_df}

        self.exec_eng.load_study_from_input_dict(dict_values)

        # manually configure the scenarios non-varying values (~reference)
        private_val = {}
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
            private_val[f'{self.study_name}.multi_scenarios.{scenario}.z'] = self.z1
        self.exec_eng.load_study_from_input_dict(private_val)
        self.exec_eng.execute()

        y1, o1 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z1 ** self.power)
        y2, o2 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z1 ** self.power)
        dump_dir = join(self.root_dir, self.namespace)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.namespace)
        builders = exec_eng2.factory.get_builder_from_process(
            self.repo, proc_name)
        exec_eng2.factory.set_builders_to_coupling_builder(builders)
        exec_eng2.configure()

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(exec_eng2.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(
            join(self.root_dir, self.namespace))

    def test_08_changing_trade_variables_by_adding_df_column(self):
        # # simple 2-disc process NOT USING nested scatters
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(
            self.repo, proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()
        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values = {f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance',
                       f'{self.study_name}.multi_scenarios.scenario_df': scenario_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        # manually configure the scenarios non-varying values (~reference)
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = self.a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = self.x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = self.constant
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = self.power
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.z'] = self.z1
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()
        y1, o1 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z1 ** self.power)
        y2, o2 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z1 ** self.power)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1, self.z2], ['scenario_2', True, self.b2, self.z2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b', 'z'])
        dict_values[f'{self.study_name}.multi_scenarios.scenario_df'] = scenario_df
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()
        y1, o1 = (self.a1 * self.x1 + self.b1,
                  self.constant + self.z2 ** self.power)
        y2, o2 = (self.a1 * self.x1 + self.b2,
                  self.constant + self.z2 ** self.power)

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.o'), o1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.o'), o2)

    def test_09_two_scenarios_with_same_name(self):
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(self.repo,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', False, 0], ['scenario_1', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])
        dict_values = {f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance',
                       f'{self.study_name}.multi_scenarios.scenario_df': scenario_df}

        error_message = 'Cannot activate several scenarios with the same name (scenario_1).'
        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios'

        # Exception
        # with self.assertRaises(Exception) as cm:
        #     self.exec_eng.load_study_from_input_dict(dict_values)
        # self.assertEqual(str(cm.exception), error_message)

        # Logging only
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())
        self.assertIn(error_message, self.my_handler.msg_list)

        runtime_error_message = 'Variable MyCase.multi_scenarios.scenario_df : ' \
                                'Cannot activate several scenarios with the same name (scenario_1).'
        # data integrity Exception
        with self.assertRaises(ValueError) as cm:
            self.exec_eng.execute()
        self.assertIn(runtime_error_message, str(cm.exception))

    def test_10_two_scenarios_with_same_name_on_2nd_config(self):
        proc_name = 'test_multi_instance_basic'
        builders = self.exec_eng.factory.get_builder_from_process(self.repo,
                                                                  proc_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(builders)
        self.exec_eng.configure()

        scenario_df = pd.DataFrame(
            [['scenario_1', True, self.b1], ['scenario_2', False, 0], ['scenario_2', True, self.b2]],
            columns=['scenario_name', 'selected_scenario', 'Disc1.b'])

        dict_values = {f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance',
                       f'{self.study_name}.multi_scenarios.scenario_df': scenario_df}
        self.exec_eng.load_study_from_input_dict(dict_values)

        scenario_df['scenario_name'].iloc[2] = 'scenario_1'
        self.exec_eng.load_study_from_input_dict(dict_values)

        error_message = 'Cannot activate several scenarios with the same name (scenario_1).'
        exp_tv = 'Nodes representation for Treeview MyCase\n' \
                 '|_ MyCase\n' \
                 '\t|_ multi_scenarios\n' \
                 '\t\t|_ scenario_1\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3\n' \
                 '\t\t|_ scenario_2\n' \
                 '\t\t\t|_ Disc1\n' \
                 '\t\t\t|_ Disc3'

        # Exception
        # with self.assertRaises(Exception) as cm:
        #     self.exec_eng.load_study_from_input_dict(dict_values)
        # self.assertEqual(str(cm.exception), error_message)

        # Logging only
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.assertEqual(exp_tv, self.exec_eng.display_treeview_nodes())
        self.assertIn(error_message, self.my_handler.msg_list)

        runtime_error_message = 'Variable MyCase.multi_scenarios.scenario_df : ' \
                                'Cannot activate several scenarios with the same name (scenario_1).'
        # data integrity Exception
        with self.assertRaises(ValueError) as cm:
            self.exec_eng.execute()
        self.assertIn(runtime_error_message, str(cm.exception))

    # EEV3 TESTS #TODO: cleanup when nested scatter exists

if '__main__' == __name__:
    cls = TestSimpleMultiScenario()
    cls.setUp()
    cls.test_06_trade_on_name_list()
