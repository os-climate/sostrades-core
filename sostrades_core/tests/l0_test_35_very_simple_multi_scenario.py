'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/12-2023/11/03 Copyright 2023 Capgemini

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

import pandas as pd
from sostrades_core.sos_processes.test.tests_driver_eval.multi.test_multi_driver.usecase_scatter import \
    Study as study_scatter

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from time import sleep
from shutil import rmtree
from pathlib import Path
from os.path import join
import os

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
# from sos_trades_core.execution_engine.sos_very_simple_multi_scenario import SoSVerySimpleMultiScenario
# from sostrades_core.execution_engine.scatter_data import SoSScatterData
from tempfile import gettempdir
from sostrades_core.tools.rw.load_dump_dm_data import DirectLoadDump
from sostrades_core.study_manager.base_study_manager import BaseStudyManager
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.proxy_coupling import ProxyCoupling
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class TestVerySimpleMultiScenario(unittest.TestCase):
    """
    SoSVerySimpleMultiScenario test class
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

    def tearDown(self):

        for dir_to_del in self.dirs_to_del:
            sleep(0.5)
            if Path(dir_to_del).is_dir():
                rmtree(dir_to_del)
        sleep(0.5)

    def get_simple_multiscenario_process_configured(self, exec_eng):
        # shared namespace
        exec_eng.ns_manager.add_ns(
            'ns_disc3', 'MyCase')
        exec_eng.ns_manager.add_ns(
            'ns_out_disc3', 'MyCase')

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', builder_list)
        exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        exec_eng.configure()

    def test_01_multi_scenario_driver(self):

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)
        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', 'MyCase.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', 'MyCase')

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', builder_list)

        # add post-processing on 'Post-processing' node by loading a module
        # with implemented graphs
        self.exec_eng.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_post_proc', 'sostrades_core.sos_wrapping.test_discs.chart_post_proc_multi_scenario')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        dict_values[f'{self.study_name}.multi_scenarios.instance_reference'] = False
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = 3
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.samples_df')['scenario_name'].values.tolist(), ['scenario_1',
                                                                                    'scenario_2'])

        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_2.y'), y2)

    def test_04_consecutive_configure(self):

        self.get_simple_multiscenario_process_configured(self.exec_eng)
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # # # check tree view with scenario_1 and scenario_2
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        dict_values = {}
        dict_values[self.study_name + '.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           False],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # # check tree view after scenario_2 deletion to validate cleaning
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys()
             if 'scenario_2' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN],
            [])

        dict_values[self.study_name +
                    '.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                         True,
                                                                                         True],
                                                                   'scenario_name': ['scenario_1',
                                                                                     'scenario_2',
                                                                                     'scenario_3']})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        dict_values[self.study_name +
                    '.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [],
                                                                   'scenario_name': []})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        dict_values[self.study_name +
                    '.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                         True],
                                                                   'scenario_name': ['scenario_A',
                                                                                     'scenario_B']})

        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if
             'scenario_1' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])
        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if
             'scenario_2' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])
        self.assertListEqual(
            [key for key in self.exec_eng.dm.data_id_map.keys() if
             'scenario_3' in key and key.split('.')[-1] not in ProxyDiscipline.NUM_DESC_IN and
             key.split('.')[-1] not in ProxyCoupling.DESC_IN], [])

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_list = ['scenario_A', 'scenario_B']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = 3
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_A.Disc1.b'] = b1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_B.Disc1.b'] = b2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_A.z'] = 1.2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_B.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

    def _test_05_dump_and_load_after_execute(self):

        self.get_simple_multiscenario_process_configured(self.exec_eng)

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = 3
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_2.y'), y2)

        dump_dir = join(self.root_dir, self.namespace)

        BaseStudyManager.static_dump_data(
            dump_dir, self.exec_eng, DirectLoadDump())

        exec_eng2 = ExecutionEngine(self.namespace)
        self.get_simple_multiscenario_process_configured(exec_eng2)

        BaseStudyManager.static_load_data(
            dump_dir, exec_eng2, DirectLoadDump())

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.scenario_1.y'), y2)
        # Clean the dump folder at the end of the test
        self.dirs_to_del.append(
            join(self.root_dir, self.namespace))

    def test_05_get_samples_after_crash(self):
        # scatter build map
        self.get_simple_multiscenario_process_configured(self.exec_eng)

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})
        # dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
        #     'scenario_1', 'scenario_2']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[self.study_name + '.a'] = a1

            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.constant'] = 3
            dict_values[self.study_name + '.multi_scenarios.' +
                        scenario + '.Disc3.power'] = 2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[self.study_name +
                    '.multi_scenarios.scenario_2.Disc3.z'] = 1.5
        # missing input x1
        #         dict_values[self.study_name + '.x'] = x1

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        try:
            # execute fails because of missing input x1 => ValueError
            self.exec_eng.execute()
        except ValueError:
            # check that all variables have been loaded on dm at the end of the
            # execution
            for var in self.exec_eng.dm.data_id_map.keys():
                dm_value = self.exec_eng.dm.get_value(var)
                if var == f'{self.study_name}.multi_scenarios.scenario_list' or \
                        var == f'{self.study_name}.multi_scenarios.generated_samples' or \
                        var == f'{self.study_name}.multi_scenarios_gather.eval_outputs':
                    # this variable is an exception because it is forced by the value of another variable during setup
                    continue

                if var not in dict_values:  # default inputs and all outputs
                    if isinstance(dm_value, pd.DataFrame):
                        self.assertListEqual(self.exec_eng.dm.get_data(
                            var, 'default').values.tolist(), dm_value.values.tolist())
                    else:
                        self.assertEqual(self.exec_eng.dm.get_data(
                            var, 'default'), dm_value)
                else:  # user defined inputs
                    if isinstance(dm_value, pd.DataFrame):
                        self.assertListEqual(
                            dict_values[var].values.tolist(), dm_value.values.tolist())
                    else:
                        self.assertEqual(dict_values[var], dm_value)

    def _test_06_scatter_node_namespace_removal_and_change_builder_mode_multi_to_mono(self):
        # deactivated after mono/multi split
        '''
        At this stage, the initial namespace values of the subprocess are cleaned once the multi-instance is activated
        and namespace values are updated. Therefore, once the mono-instance is activated from the multi-instance,
        the initial namespace cannot be recovered to update back the namespace value. An initial_ns_value_dict could
        be introduced. however, it would add complexity and an on-going dev is currently (26-oct-2023) refactoring the code to
        deal with multi-instance and mono-instance in separate class. This refactoring should also deal with this case.
        For those reasons, it is decided to deactivate this test
        '''
        # scatter build map
        self.get_simple_multiscenario_process_configured(self.exec_eng)

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # # check tree view with scenario_1 and scenario_2
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # dict_values[self.study_name +
        #             '.multi_scenarios.scenario_list'] = ['scenario_1']

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes(display_variables=True)

        # check tree view after all namespaces are under subprocess name
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_07_scatter_node_namespace_removal_and_change_builder_mode_mono_to_multi(self):
        # deactivated after mono/multi split
        self.get_simple_multiscenario_process_configured(self.exec_eng)

        dict_values = {}
        self.exec_eng.load_study_from_input_dict(dict_values)

        # -- set up disciplines
        private_values = {
            self.study_name + '.x': 10.,
            self.study_name + '.a': 5.,
            self.study_name + '.multi_scenarios.subprocess.Disc1.b': 25431.,
            self.study_name + '.multi_scenarios.subprocess.Disc3.constant': 3.1416,
            self.study_name + '.multi_scenarios.Disc3.z': 1,
            self.study_name + '.multi_scenarios.subprocess.Disc3.power': 2}
        self.exec_eng.load_study_from_input_dict(private_values)

        input_selection_x_b = {'selected_input': [True, True],
                               'full_name': ['x', 'subprocess.Disc1.b']}
        input_selection_x_b = pd.DataFrame(input_selection_x_b)

        output_selection_y_ind = {'selected_output': [True, True],
                                  'full_name': ['y', 'subprocess.Disc1.indicator']}
        output_selection_y_ind = pd.DataFrame(output_selection_y_ind)

        io_dict = {
            f'{self.study_name}.multi_scenarios.eval_inputs': input_selection_x_b,
            f'{self.study_name}.multi_scenarios.eval_outputs': output_selection_y_ind,
        }
        self.exec_eng.load_study_from_input_dict(io_dict)
        self.exec_eng.display_treeview_nodes(display_variables=True)

        # check the mono-instance dynamic outputs are created
        output_data_names = self.exec_eng.root_process.get_output_data_names()
        self.assertIn('MyCase.multi_scenarios.samples_inputs_df', output_data_names)
        self.assertIn('MyCase.multi_scenarios.y_dict', output_data_names)
        self.assertIn('MyCase.multi_scenarios.subprocess.Disc1.indicator_dict', output_data_names)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        print(exp_tv_str)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc3']
        exp_tv_str = '\n'.join(exp_tv_list)
        print(exp_tv_str)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        # # check tree view with scenario_1 and scenario_2
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc3', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        # check that the mono-instance dynamic inputs have been cleared
        output_data_names = self.exec_eng.root_process.get_output_data_names()
        self.assertNotIn('MyCase.multi_scenarios.samples_inputs_df', output_data_names)
        self.assertNotIn('MyCase.y_dict', output_data_names)
        self.assertNotIn('MyCase.multi_scenarios.subprocess.Disc1.indicator_dict', output_data_names)

    def _test_08_check_double_prepare_execution_and_status(self):

        study = study_scatter()
        study.load_data()
        dict_values = {f'{study.study_name}.cache_type': 'SimpleCache'}
        study.load_data(from_input_dict=dict_values)

        status_dict_all_done = {'<study_ph>': {'ProxyCoupling': 'DONE'},
                                '<study_ph>.multi_scenarios': {'ProxyDriverEvaluator': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_1': {'ProxyCoupling': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_1.Disc1': {'ProxyDiscipline': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_1.Disc2': {'ProxyDiscipline': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_2': {'ProxyCoupling': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_2.Disc1': {'ProxyDiscipline': 'DONE'},
                                '<study_ph>.multi_scenarios.scenario_2.Disc2': {'ProxyDiscipline': 'DONE'}}

        # run with dump cache_map
        study.run()

        # check sosmdachain of execution and sosmdachain under proxycouplings
        proxy_coupling1 = study.execution_engine.root_process.proxy_disciplines[
            0].proxy_disciplines[1]
        mdachain_under_proxy_coupling = proxy_coupling1.mdo_discipline_wrapp.mdo_discipline
        mdodisciplinedriver = study.execution_engine.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[
            0]
        mdachain_under_mdodisciplinedriver = mdodisciplinedriver.disciplines[0]
        self.assertEqual(mdachain_under_proxy_coupling.__hash__,
                         mdachain_under_mdodisciplinedriver.__hash__)
        self.assertDictEqual(
            status_dict_all_done, study.execution_engine.get_anonimated_disciplines_status_dict())
        study.run()

        # check sosmdachain of execution and sosmdachain under proxycouplings
        proxy_coupling1 = study.execution_engine.root_process.proxy_disciplines[
            0].proxy_disciplines[1]
        mdachain_under_proxy_coupling = proxy_coupling1.mdo_discipline_wrapp.mdo_discipline
        mdodisciplinedriver = study.execution_engine.root_process.mdo_discipline_wrapp.mdo_discipline.disciplines[
            0]
        mdachain_under_mdodisciplinedriver = mdodisciplinedriver.disciplines[0]
        self.assertEqual(mdachain_under_proxy_coupling.__hash__,
                         mdachain_under_mdodisciplinedriver.__hash__)
        self.assertDictEqual(
            status_dict_all_done, study.execution_engine.get_anonimated_disciplines_status_dict())

    def test_09_multi_scenario_scatter_flatten_subprocess(self):

        builder_list = self.factory.get_builder_from_process(repo=self.repo,
                                                             mod_id='test_disc1_scenario')

        self.exec_eng.ns_manager.add_ns(
            'ns_disc3', 'MyCase.Disc3')
        self.exec_eng.ns_manager.add_ns(
            'ns_out_disc3', 'MyCase')

        mod_list = f'{self.base_path}.disc3_scenario.Disc3'
        disc3_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc3', mod_list)
        builder_list.append(disc3_builder)

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', builder_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)
        self.exec_eng.configure()

        dict_values = {}
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = samples_df
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes(exec_display=True, display_variables=True)

        scenario_list = ['scenario_1', 'scenario_2']
        for scenario in scenario_list:
            x1 = 2.
            x2 = 4.
            a1 = 3
            b1 = 4
            a2 = 6
            b2 = 2

            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.a'] = a1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.x'] = x1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.constant'] = 3
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.Disc3.power'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc1.b'] = b1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc1.b'] = b2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.Disc3.z'] = 1.2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.Disc3.z'] = 1.5

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        # check that all disciplines are at root node with flatten_subprocess option
        disc_list_in_root_coupling = [disc.get_disc_full_name() for disc in
                                      self.exec_eng.root_process.proxy_disciplines]
        disc_list_in_root_coupling_th = ['MyCase.multi_scenarios',
                                         'MyCase.multi_scenarios_gather',
                                         'MyCase.multi_scenarios.scenario_1.Disc1',
                                         'MyCase.multi_scenarios.scenario_1.Disc3',
                                         'MyCase.multi_scenarios.scenario_2.Disc1',
                                         'MyCase.multi_scenarios.scenario_2.Disc3']
        self.assertEqual(disc_list_in_root_coupling, disc_list_in_root_coupling_th)
        self.exec_eng.execute()

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.samples_df')['scenario_name'].values.tolist(), ['scenario_1',
                                                                                    'scenario_2'])

        y1 = a1 * x1 + b1
        y2 = a1 * x1 + b2

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.y'), y1)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.y'), y2)


if '__main__' == __name__:
    cls = TestVerySimpleMultiScenario()
    cls.setUp()
    cls.test_01_multi_scenario_driver()
    cls.tearDown()
