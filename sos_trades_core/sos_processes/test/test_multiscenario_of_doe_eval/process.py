'''
Copyright 2022 Airbus SA

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
    '''This test_multiscenario_of_DoE_Eval specify an example of a multiscenario
     of Sensitivity Analysis process.
    It uses the 2 wrapped disciplines : disc1_scenario.Disc1 (orchectrated by the
     test_disc1_scenario process) and disc3_scenario.Disc3.
    '''
    # ontology information
    _ontology_data = {
        'label': 'Core Test Multiscenario Process OF DoE_Eval',
        'description': '',
        'category': '',
        'version': '',
    }
    def get_builders(self):
        # This process is based on test 38
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
                        'ns_to_update': ['ns_disc3', 'ns_doe_eval', 'ns_barrierr', 'ns_out_disc3']}
        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        # shared namespace
        self.ee.ns_manager.add_ns('ns_barrierr', self.ee.study_name)
        self.ee.ns_manager.add_ns(
            'ns_scatter_scenario', f'{self.ee.study_name}.multi_scenarios')
        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{self.ee.study_name}.multi_scenarios.Disc3')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{self.ee.study_name}.multi_scenarios')
        self.ee.ns_manager.add_ns(
            'ns_data_ac', self.ee.study_name)
        self.ee.ns_manager.add_ns(
            'ns_doe_eval', f'{self.ee.study_name}.multi_scenarios.DoE_Eval')
        # instantiate factory # get instantiator from Discipline class
        builder_list = self.ee.factory.get_builder_from_process(repo=repo,
                                                           mod_id='test_disc1_scenario')
        scatter_list = self.ee.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list, autogather=False)
        mod_path = f'{base_path}.disc3_scenario.Disc3'
        disc3_builder = self.ee.factory.get_builder_from_module(
            'Disc3', mod_path)
        scatter_list.append(disc3_builder)
        doe_eval_builder = self.ee.factory.create_evaluator_builder(
            'DoE_Eval', 'doe_eval', scatter_list)
        multi_scenarios = self.ee.factory.create_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', [doe_eval_builder], autogather=False)
        return multi_scenarios
