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
#-- Generate test 1 process
from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Disc1 Disc3 Very Simple Multi Scenario Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):
        '''
        Subprocess for multiscenario 
        '''
        # scatter build map
        ac_map = {'input_name': 'name_list',

                  'input_ns': 'ns_scatter_scenario',
                  'output_name': 'ac_name',
                  'scatter_ns': 'ns_ac',
                  'ns_to_update': ['ns_data_ac']}

        self.ee.smaps_manager.add_build_map('name_list', ac_map)

        # instantiate factory # get instantiator from Discipline class

        builder_list = self.ee.factory.get_builder_from_process(repo='sos_trades_core.sos_processes.test',
                                                                mod_id='test_disc1_scenario')

        scatter_list = self.ee.factory.create_multi_scatter_builder_from_list(
            'name_list', builder_list=builder_list)

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc3_scenario.Disc3'
        disc3_builder = self.create_builder_list({'Disc3': mod_list}, ns_dict={'ns_disc3': f'{self.ee.study_name}.Disc3',
                                                                               'ns_out_disc3': f'{self.ee.study_name}'}
                                                 )
        scatter_list.extend(disc3_builder)
        '''
        End of subprocess
        '''

        # scenario build map
        scenario_map = {'input_name': 'scenario_list',
                        'ns_to_update': ['ns_disc3', 'ns_out_disc3']}

        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)

#         # shared namespace
        self.ee.ns_manager.add_ns(
            'ns_scatter_scenario', f'{self.ee.study_name}')

        multi_scenario_name = 'multi_scenarios'
        multi_scenarios = self.ee.factory.create_very_simple_multi_scenario_builder(
            multi_scenario_name, 'scenario_list', scatter_list)

        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_ns=multi_scenario_name, after_name=self.ee.study_name)

        return multi_scenarios
