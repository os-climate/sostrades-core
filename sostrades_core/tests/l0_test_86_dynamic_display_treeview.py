'''
Copyright 2022 Airbus SAS
Modifications on 2023/10/03-2024/05/16 Copyright 2023 Capgemini

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

import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestConfigDependencyDiscs(unittest.TestCase):
    """
    Tool building test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.study_name = 'MyCase'
        self.exec_eng = ExecutionEngine(self.study_name)
        self.factory = self.exec_eng.factory

        self.repo = 'sostrades_core.sos_processes.test'

    def test_01_display_existing_variable_ns(self):
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        display_value = f'{self.exec_eng.study_name}.new_display'
        self.exec_eng.ns_manager.add_ns(
            'ns_ac', self.exec_eng.study_name, display_value=display_value)

        ns_ac = self.exec_eng.ns_manager.get_ns_in_shared_ns_dict('ns_ac')

        self.assertEqual(ns_ac.get_display_value(), display_value)
        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes()
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ Disc1',
                       '\t|_ Disc2',
                       '\t|_ new_display', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        treeview_display_var = self.exec_eng.display_treeview_nodes(
            display_variables=True)

        assert '\n\t\t-> x' == treeview_display_var.split('new_display')[-1]

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ Disc1',
                       '\t|_ Disc2', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

    def test_02_display_existing_disc_ns(self):
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1_builder, disc2_builder])

        self.exec_eng.ns_manager.add_ns(
            'ns_ac', self.exec_eng.study_name)
        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.study_name}.New_ns_disc')

        self.exec_eng.configure()

        self.exec_eng.display_treeview_nodes(display_variables=True)
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ New_ns_disc',
                       '\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.study_name}.New_ns_disc_new')
        self.exec_eng.dm.treeview = None
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ New_ns_disc_new',
                       '\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc2_builder, f'{self.study_name}.New_ns_disc2')
        self.exec_eng.dm.treeview = None
        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ New_ns_disc_new',
                       '\t|_ New_ns_disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.study_name}')
        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc2_builder, f'{self.study_name}')
        self.exec_eng.dm.treeview = None

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ Disc1',
                       '\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

    def test_03_display_on_single_instance_evaluator(self):
        self.repo = 'sostrades_core.sos_processes.test'

        my_namespace = {'ns_barrierr': self.exec_eng.study_name,
                        'ns_ac': f'{self.exec_eng.study_name}.Disc1'}

        # instantiate factory by getting builder from process
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.ns_manager.add_ns_def(my_namespace)
        multi_scenarios = self.exec_eng.factory.create_mono_instance_driver(
            'multi_scenarios', [disc1_builder, disc2_builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(multi_scenarios)

        self.exec_eng.configure()
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})

        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ subprocess',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_04_display_on_multi_instance_evaluator(self):
        self.repo = 'sostrades_core.sos_processes.test'

        # instantiate factory by getting builder from process
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.ns_manager.add_ns(
            'ns_ac', f'{self.exec_eng.study_name}', display_value=f'{self.exec_eng.study_name}.Disc1')

        #         self.exec_eng.ns_manager.add_display_ns_to_builder(
        #             disc1_builder, f'{self.exec_eng.study_name}.Disc1')
        #         self.exec_eng.ns_manager.add_display_ns_to_builder(
        #             disc2_builder, f'{self.exec_eng.study_name}.Disc2')
        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios',
                                                                             [disc1_builder, disc2_builder])
        self.exec_eng.ns_manager.add_display_ns_to_builder(
            multi_scenarios[0], f'{self.exec_eng.study_name}')
        self.exec_eng.factory.set_builders_to_coupling_builder(multi_scenarios)
        self.exec_eng.configure()
        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                           True],
                                                                                     'scenario_name': ['scenario_1',
                                                                                                       'scenario_2']})
        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2',
                       '\t|_ multi_scenarios_gather',]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ scenario_1',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2',
                       '\t|_ scenario_2',
                       '\t\t|_ Disc1',
                       '\t\t|_ Disc2',
                       '\t|_ Disc1']

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_05_display_on_multi_instance_evaluator_hide_coupling(self):
        self.repo = 'sostrades_core.sos_processes.test'

        # instantiate factory by getting builder from process
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc1.Disc1'
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc2.Disc2'
        disc2_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc2', mod_list)

        self.exec_eng.ns_manager.add_ns(
            'ns_ac', f'{self.exec_eng.study_name}', display_value=f'{self.exec_eng.study_name}.Disc1')

        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc1_builder, f'{self.exec_eng.study_name}.Disc1')
        self.exec_eng.ns_manager.add_display_ns_to_builder(
            disc2_builder, f'{self.exec_eng.study_name}.Disc2')
        driver_name = 'multi_scenarios'
        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver(driver_name,
                                                                             [disc1_builder, disc2_builder])
        self.exec_eng.ns_manager.add_display_ns_to_builder(
            multi_scenarios[0], f'{self.exec_eng.study_name}')
        self.exec_eng.factory.set_builders_to_coupling_builder(multi_scenarios)
        self.exec_eng.configure()
        dict_values = {}
        dict_values[f'{self.study_name}.{driver_name}.samples_df'] = pd.DataFrame({'selected_scenario': [True,
                                                                                                         True],
                                                                                   'scenario_name': ['scenario_1',
                                                                                                     'scenario_2']})

        dict_values[f'{self.study_name}.{driver_name}.display_options'] = {'hide_coupling_in_driver': True}
        self.exec_eng.load_study_from_input_dict(dict_values)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       f'\t|_ {driver_name}',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Disc1',
                       '\t\t\t|_ Disc2',
                       f'\t|_ {driver_name}_gather',]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes(
            exec_display=True)

        exp_tv_list = [f'Nodes representation for Treeview {self.study_name}',
                       f'|_ {self.study_name}',
                       '\t|_ Disc1',
                       '\t|_ Disc2', ]

        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestConfigDependencyDiscs()
    cls.setUp()
    cls.test_05_display_on_multi_instance_evaluator_hide_coupling()
    cls.tearDown()
