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
    def get_builders(self):

        repo = 'sos_trades_core.sos_processes.test'

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scenario',
                        'output_name': 'scenario_name',
                        'scatter_ns': 'ns_scenario',
                        'ns_to_update': ['ns_ac']}

        self.ee.smaps_manager.add_build_map(
            'scenario_list', scenario_map)
        self.ee.ns_manager.add_ns(
            'ns_scenario', self.ee.study_name)

        builder_list = self.ee.factory.get_builder_from_process(repo=repo,
                                                                mod_id='test_morphological_matrix_with_setup')
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            'multi_scenarios', after_name=self.ee.study_name)

        multi_scenarios = self.ee.factory.create_very_simple_multi_scenario_builder(
            'multi_scenarios', 'scenario_list', builder_list)

        return multi_scenarios
