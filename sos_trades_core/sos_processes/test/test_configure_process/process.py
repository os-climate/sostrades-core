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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- Generate test 2 process
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    def get_builders(self):

        repo = 'sos_trades_core.sos_processes.test'
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        # scatter build map
        ac_map = {'input_name': 'name_list',
                  'input_type': 'string_list',
                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'gather_ns': 'ns_scenario',
                  'ns_to_update': ['ns_data_ac']}

        self.ee.smaps_manager.add_build_map('name_list', ac_map)

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'gather_ns': 'ns_scatter_scenario',
                        'ns_to_update': ['ns_barrier']}

        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)

        # gather data build map
        mydict_u = {'input_name': 'Disc2.u',
                    'input_type': 'float',
                    'output_name': 'u_dict',
                    'output_type': 'dict',
                    'output_ns': 'ns_scatter_scenario',
                    'scatter_var_name': 'scenario_list'}
        self.ee.smaps_manager.add_data_map('gather_u', mydict_u)

        # scatter data build map
        mydict_z = {'input_name': 'z_dict',
                    'input_type': 'dict',
                    'output_name': 'z',
                    'output_type': 'float',
                    'output_ns': 'ns_scatter_scenario',
                    'scatter_var_name': 'scenario_list'}
        self.ee.smaps_manager.add_data_map('scatter_z', mydict_z)

        # shared namespace
        self.ee.ns_manager.add_ns('ns_barrier', self.ee.study_name)
        self.ee.ns_manager.add_ns(
            'ns_scatter_scenario', f'{self.ee.study_name}.multi_scenarios')
        self.ee.ns_manager.add_ns(
            'ns_data_ac', self.ee.study_name)

        builder_list = self.ee.factory.get_builder_from_process(repo=repo,
                                                                mod_id='test_disc1_scenario')

        scatter_list = self.ee.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=True)

        mod_path = f'{base_path}.disc2_scenario.Disc2'
        disc2_builder = self.ee.factory.get_builder_from_module(
            'Disc2', mod_path)
        scatter_list.append(disc2_builder)

        gather_data_u = self.ee.factory.create_gather_data_builder(
            'gather_data_u', 'gather_u')
        scatter_data_z = self.ee.factory.create_scatter_data_builder(
            'scatter_data_z', 'scatter_z')

        multi_scenarios = self.ee.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', scatter_list, autogather=True)

        multi_scenarios.append(gather_data_u)
        multi_scenarios.append(scatter_data_z)

        return multi_scenarios
