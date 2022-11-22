'''
Copyright 2022 Remy SAS

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
import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from tempfile import gettempdir


class TestMultiScenarioArchiBuilder(unittest.TestCase):
    """
    Multi scenario of architecture builder test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.root_dir = gettempdir()

    def test_01_very_simple_multi_scenario_of_simple_architecture(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy', 'Remy'],
             'Current': ['Remy', 'CAPEX', 'OPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),  ('standard'),  ('standard')],
             'Activation': [True, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'scatter_ns': 'ns_scenario'}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_scatter_driver_with_tool(
            'multi_scenarios',  [builder], map_name='scenario_list')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def(
            {'ns_scatter_scenario': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()
        scenario_df = pd.DataFrame({'selected_scenario': [True, True],
                                    'scenario_name': ['scenario_1',
                                                      'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.scenario_df': scenario_df,
                       f'{self.study_name}.multi_scenarios.builder_mode': 'multi_instance'}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ OPEX',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ OPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'OPEX': [False]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'OPEX': [True]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ OPEX']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [False],
                                        'OPEX': [True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [False],
                                        'OPEX': [False]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ OPEX',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_02_very_simple_multi_scenario_of_architecture_scatter_of_scatter(self):

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scenario',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        # subsystem scatter map dict
        subsystem_services_map_dict = {'input_name': 'subsystems_list',
                                       'input_type': 'string_list',
                                       'input_ns': 'ns_scenario',
                                       'output_name': 'subsystem',
                                       'scatter_ns': 'ns_subsystem',
                                       #   'gather_ns': 'ns_services_subsystem',
                                       'gather_ns': 'ns_ac_subsystem',
                                       'gather_ns_out': 'ns_ac',
                                       # add scatter name to this namespace
                                       # , 'ns_ac']
                                       }

        # add subsystem map
        self.exec_eng.smaps_manager.add_build_map(
            'subsystems_list', subsystem_services_map_dict)

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Boeing', 'Services', 'Services'],
             'Current': ['Remy', 'Boeing', 'CAPEX', 'Services', 'CAPEX', 'FHS', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'),
                        ('scatter', 'AC_list', ('scatter', 'subsystems_list', 'ValueBlockDiscipline')), ('scatter', 'AC_list', ('scatter', 'subsystems_list', 'ValueBlockDiscipline'))],
             'Activation': [True, True, False, False, False, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_ac', 'ns_services', 'ns_coc', 'ns_rc', 'ns_nrc', 'ns_market']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': f'{self.study_name}.multi_scenarios',
                                             'ns_services': f'{self.study_name}.multi_scenarios',
                                             'ns_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_coc': f'{self.study_name}.multi_scenarios',
                                             'ns_data_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_business_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_rc': f'{self.study_name}.multi_scenarios',
                                             'ns_nrc': f'{self.study_name}.multi_scenarios',
                                             'ns_market': f'{self.study_name}.multi_scenarios',
                                             'ns_market_in': f'{self.study_name}.multi_scenarios',
                                             'ns_scenario': f'{self.study_name}.multi_scenarios',
                                             'ns_services_ac': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_list':  [
            'scenario_1', 'scenario_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ Boeing',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ Boeing',
                       '\t\t\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy', 'Remy', 'Boeing', 'Boeing'],
                                        'AC_list': ['AC1', 'AC2', 'AC3', 'AC4'],
                                        'subsystems_list': ['Airframe', 'Propulsion', 'Propulsion', 'Propulsion', ],
                                        'CAPEX': [True, True, True, True],
                                        'Services': [True, True, False, False],
                                        'FHS': [True, True, False, False],
                                        'OSS': [True, True, False, False]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy', 'Remy', 'Boeing', 'Boeing'],
                                        'AC_list': ['AC1', 'AC2', 'AC3', 'AC4', 'AC5'],
                                        'subsystems_list': ['Airframe', 'Propulsion', 'Propulsion', 'Propulsion', 'Propulsion'],
                                        'CAPEX': [True, True, True, True, True],
                                        'Services': [True, False, True, False, False],
                                        'FHS': [True, False, False, False, False],
                                        'OSS': [True, False, True, False, False]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Boeing',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC3',
                       '\t\t\t\t\t\t|_ AC4',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ AC2',
                       '\t\t\t\t\t\t|_ AC3',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ FHS',
                       '\t\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t\t|_ AC3',
                       '\t\t\t\t\t\t\t\t|_ Propulsion',
                       '\t\t\t\t|_ Boeing',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC4',
                       '\t\t\t\t\t\t|_ AC5']
        exp_tv_str = '\n'.join(exp_tv_list)

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_03_multi_scenario_of_architecture(self):

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scenario',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),  ('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_ac', 'ns_services', 'ns_coc', 'ns_rc', 'ns_nrc', 'ns_market']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': f'{self.study_name}.multi_scenarios',
                                             'ns_services': f'{self.study_name}.multi_scenarios',
                                             'ns_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_coc': f'{self.study_name}.multi_scenarios',
                                             'ns_data_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_business_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_rc': f'{self.study_name}.multi_scenarios',
                                             'ns_nrc': f'{self.study_name}.multi_scenarios',
                                             'ns_market': f'{self.study_name}.multi_scenarios',
                                             'ns_scenario': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()

        activation_df_1 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['AC1', 'AC2'],
                                        'CAPEX': [True, True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['AC3', 'AC4'],
                                        'CAPEX': [True, True]})

        dict_values = {f'{self.study_name}.multi_scenarios.activation_df_trade': [activation_df_1, activation_df_2],
                       f'{self.study_name}.multi_scenarios.trade_variables': {'activation_df': 'dataframe'}}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC1',
                       '\t\t\t\t\t\t|_ AC2',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ AC3',
                       '\t\t\t\t\t\t|_ AC4']
        exp_tv_str = '\n'.join(exp_tv_list)

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_04_very_simple_multi_scenario_with_sub_architecture(self):

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scenario',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        vb_builder_name = 'Business'

        subarchitecture_df = pd.DataFrame(
            {'Parent': ['CAPEX', 'CAPEX'],
             'Current': ['Propu', 'Airframe'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'),  ('standard')],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),  ('architecture', subarchitecture_df)],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_ac', 'ns_services', 'ns_coc', 'ns_rc', 'ns_nrc', 'ns_market']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': f'{self.study_name}.multi_scenarios',
                                             'ns_services': f'{self.study_name}.multi_scenarios',
                                             'ns_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_coc': f'{self.study_name}.multi_scenarios',
                                             'ns_data_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_business_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_rc': f'{self.study_name}.multi_scenarios',
                                             'ns_nrc': f'{self.study_name}.multi_scenarios',
                                             'ns_market': f'{self.study_name}.multi_scenarios',
                                             'ns_scenario': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_list':  [
            'scenario_1', 'scenario_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Propu',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Propu',
                       '\t\t\t\t\t\t|_ Airframe']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'Propu': [True],
                                        'Airframe': [True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'Propu': [True],
                                        'Airframe': [False]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Propu',
                       '\t\t\t\t\t\t|_ Airframe',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Propu']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_05_very_simple_multi_scenario_with_scatter_architecture(self):

        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scenario',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.smaps_manager.add_build_map('AC_list', mydict)

        vb_builder_name = 'Business'

        subarchitecture_df = pd.DataFrame(
            {'Parent': ['CAPEX', 'CAPEX'],
             'Current': ['Propu', 'Airframe'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'),  ('standard')],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),  ('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', subarchitecture_df)],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_ac', 'ns_services', 'ns_coc', 'ns_rc', 'ns_nrc', 'ns_market']}

        self.exec_eng.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.exec_eng.ns_manager.add_ns(
            'ns_scatter_scenario', 'MyCase.multi_scenarios')

        multi_scenarios = self.exec_eng.factory.create_very_simple_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def({'ns_public': f'{self.study_name}.multi_scenarios',
                                             'ns_services': f'{self.study_name}.multi_scenarios',
                                             'ns_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_coc': f'{self.study_name}.multi_scenarios',
                                             'ns_data_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_business_ac': f'{self.study_name}.multi_scenarios',
                                             'ns_rc': f'{self.study_name}.multi_scenarios',
                                             'ns_nrc': f'{self.study_name}.multi_scenarios',
                                             'ns_market': f'{self.study_name}.multi_scenarios',
                                             'ns_scenario': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_list':  [
            'scenario_1', 'scenario_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['A1', 'A2'],
                                        'CAPEX': [True, True],
                                        'Propu': [True, True],
                                        'Airframe': [True, True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['A3', 'A4'],
                                        'CAPEX': [True, False],
                                        'Propu': [False, True],
                                        'Airframe': [True, True]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       f'\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ A1',
                       '\t\t\t\t\t\t\t|_ Propu',
                       '\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t\t\t\t\t|_ A2',
                       '\t\t\t\t\t\t\t|_ Propu',
                       '\t\t\t\t\t\t\t|_ Airframe',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ A3',
                       '\t\t\t\t\t\t\t|_ Airframe', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestMultiScenarioArchiBuilder()
    cls.setUp()
    cls.test_01_very_simple_multi_scenario_of_simple_architecture()
