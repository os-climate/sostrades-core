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
from tempfile import gettempdir

import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestMultiScenarioArchiBuilder(unittest.TestCase):
    """Multi scenario of architecture builder test class"""

    def setUp(self):
        '''Initialize third data needed for testing'''
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
             'Action': [('standard'), ('standard'), ('standard')],
             'Activation': [True, False, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)
        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', [builder])

        self.exec_eng.factory.set_builders_to_coupling_builder(
            multi_scenarios)

        self.exec_eng.ns_manager.add_ns_def(
            {'ns_scatter_scenario': f'{self.study_name}.multi_scenarios'})

        self.exec_eng.configure()
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
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
                       '\t|_ multi_scenarios',
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
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ OPEX',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_02_multi_scenario_of_architecture(self):
        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline')],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', [builder])

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
                                        'AC_list': ['Product1', 'Product2'],
                                        'CAPEX': [True, True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['Product3', 'Product4'],
                                        'CAPEX': [True, True]})
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        self.exec_eng.load_study_from_input_dict(dict_values)

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t|_ Product2',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product3',
                       '\t\t\t\t\t\t|_ Product4']
        exp_tv_str = '\n'.join(exp_tv_list)

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_03_very_simple_multi_scenario_with_sub_architecture(self):
        vb_builder_name = 'Business'

        subarchitecture_df = pd.DataFrame(
            {'Parent': ['CAPEX', 'CAPEX'],
             'Current': ['Materials', 'Factory'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard')],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'), ('architecture', subarchitecture_df)],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', [builder])

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
        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t|_ Factory',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t|_ Factory']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'Materials': [True],
                                        'Factory': [True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy'],
                                        'CAPEX': [True],
                                        'Materials': [True],
                                        'Factory': [False]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t|_ Factory',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Materials']
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def test_04_very_simple_multi_scenario_with_scatter_architecture(self):
        vb_builder_name = 'Business'

        subarchitecture_df = pd.DataFrame(
            {'Parent': ['CAPEX', 'CAPEX'],
             'Current': ['Materials', 'Factory'],
             'Type': ['ValueBlockDiscipline', 'ValueBlockDiscipline'],
             'Action': [('standard'), ('standard')],
             'Activation': [False, False]})

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy'],
             'Current': ['Remy', 'CAPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),
                        ('scatter_architecture', 'AC_list', 'SumValueBlockDiscipline', subarchitecture_df)],
             'Activation': [True, False]})

        builder = self.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        multi_scenarios = self.exec_eng.factory.create_multi_instance_driver('multi_scenarios', [builder])

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

        samples_df = pd.DataFrame({'selected_scenario': [True, True],
                                   'scenario_name': ['scenario_1',
                                                     'scenario_2']})
        dict_values = {f'{self.study_name}.multi_scenarios.samples_df': samples_df}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
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
                                        'Materials': [True, True],
                                        'Factory': [True, True]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy'],
                                        'AC_list': ['A3', 'A4'],
                                        'CAPEX': [True, False],
                                        'Materials': [True, True],
                                        'Factory': [True, True]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ A1',
                       '\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t\t|_ Factory',
                       '\t\t\t\t\t\t|_ A2',
                       '\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t\t|_ Factory',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ A3',
                       '\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t\t|_ Factory', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

    def _test_02_very_simple_multi_scenario_of_architecture_scatter_of_scatter(self):
        mydict = {'input_name': 'AC_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scenario',
                  'output_name': 'AC_name',
                  'scatter_ns': 'ns_ac'}
        self.exec_eng.scattermap_manager.add_build_map('AC_list', mydict)

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
        self.exec_eng.scattermap_manager.add_build_map(
            'subsystems_list', subsystem_services_map_dict)

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Business', 'Remy', 'Remy', 'Competitor', 'Services', 'Services'],
             'Current': ['Remy', 'Competitor', 'CAPEX', 'Services', 'CAPEX', 'SAV', 'OSS'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline',
                      'SumValueBlockDiscipline'],
             'Action': [('standard'), ('standard'), ('scatter', 'AC_list', 'ValueBlockDiscipline'), ('standard'),
                        ('scatter', 'AC_list', 'ValueBlockDiscipline'),
                        ('scatter', 'AC_list', ('scatter', 'subsystems_list', 'ValueBlockDiscipline')),
                        ('scatter', 'AC_list', ('scatter', 'subsystems_list', 'ValueBlockDiscipline'))],
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

        self.exec_eng.scattermap_manager.add_build_map(
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

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_list': [
            'scenario_1', 'scenario_2']}

        self.exec_eng.load_study_from_input_dict(dict_values)
        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ SAV',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ Competitor',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t|_ scenario_2',
                       '\t\t\t|_ Business',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ SAV',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t|_ Competitor',
                       '\t\t\t\t\t|_ CAPEX', ]
        exp_tv_str = '\n'.join(exp_tv_list)
        assert exp_tv_str == self.exec_eng.display_treeview_nodes()

        activation_df_1 = pd.DataFrame({'Business': ['Remy', 'Remy', 'Competitor', 'Competitor'],
                                        'AC_list': ['Product1', 'Product2', 'Product3', 'Product4'],
                                        'subsystems_list': ['Factory', 'Materials', 'Materials', 'Materials', ],
                                        'CAPEX': [True, True, True, True],
                                        'Services': [True, True, False, False],
                                        'SAV': [True, True, False, False],
                                        'OSS': [True, True, False, False]})

        activation_df_2 = pd.DataFrame({'Business': ['Remy', 'Remy', 'Remy', 'Competitor', 'Competitor'],
                                        'AC_list': ['Product1', 'Product2', 'Product3', 'Product4', 'AC5'],
                                        'subsystems_list': ['Factory', 'Materials', 'Materials', 'Materials',
                                                            'Materials'],
                                        'CAPEX': [True, True, True, True, True],
                                        'Services': [True, False, True, False, False],
                                        'SAV': [True, False, False, False, False],
                                        'OSS': [True, False, True, False, False]})

        dict_values = {f'{self.study_name}.multi_scenarios.scenario_1.Business.activation_df': activation_df_1,
                       f'{self.study_name}.multi_scenarios.scenario_2.Business.activation_df': activation_df_2, }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()

        exp_tv_list = [f'Nodes representation for Treeview {self.namespace}',
                       f'|_ {self.namespace}',
                       '\t|_ multi_scenarios',
                       '\t\t|_ scenario_1',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t|_ Product2',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ SAV',
                       '\t\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t\t\t|_ Factory',
                       '\t\t\t\t\t\t\t|_ Product2',
                       '\t\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t\t\t|_ Factory',
                       '\t\t\t\t\t\t\t|_ Product2',
                       '\t\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t|_ Competitor',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product3',
                       '\t\t\t\t\t\t|_ Product4',
                       '\t\t|_ scenario_2',
                       f'\t\t\t|_ {vb_builder_name}',
                       '\t\t\t\t|_ Remy',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t|_ Product2',
                       '\t\t\t\t\t\t|_ Product3',
                       '\t\t\t\t\t|_ Services',
                       '\t\t\t\t\t\t|_ SAV',
                       '\t\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t\t\t|_ Factory',
                       '\t\t\t\t\t\t|_ OSS',
                       '\t\t\t\t\t\t\t|_ Product1',
                       '\t\t\t\t\t\t\t\t|_ Factory',
                       '\t\t\t\t\t\t\t|_ Product3',
                       '\t\t\t\t\t\t\t\t|_ Materials',
                       '\t\t\t\t|_ Competitor',
                       '\t\t\t\t\t|_ CAPEX',
                       '\t\t\t\t\t\t|_ Product4',
                       '\t\t\t\t\t\t|_ AC5']
        exp_tv_str = '\n'.join(exp_tv_list)

        assert exp_tv_str == self.exec_eng.display_treeview_nodes()


if '__main__' == __name__:
    cls = TestMultiScenarioArchiBuilder()
    cls.setUp()
    cls.test_02_multi_scenario_of_architecture()
