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
        'label': 'Core Test (scatter of Disc1, Disc3) Very Simple Multi Scenario Process (from proc)',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        # 1. Define scatter_list from sub_proc
        repo = 'sos_trades_core.sos_processes.test'
        sub_proc = 'test_scatter-disc1_disc3_from_proc'
        scatter_list = self.ee.factory.get_builder_from_process(
            repo=repo, mod_id=sub_proc)

        # 2. scenario build map
        scenario_map_name = 'scenario_list'
        input_ns = 'ns_scatter_scenario'  # same as input_ns of scatter
        output_name = 'scenario_name'
        scatter_ns = 'ns_scenario'      # same as gather_ns of scatter
        scenario_map = {'input_name': scenario_map_name,
                        'input_ns': input_ns,
                        'output_name': output_name,
                        'scatter_ns': scatter_ns,
                        'gather_ns': input_ns,
                        'ns_to_update': ['ns_disc3', 'ns_out_disc3']}
        self.ee.smaps_manager.add_build_map(
            scenario_map_name, scenario_map)
        # driver name
        driver_name = 'vs_MS'
        root = f'{self.ee.study_name}'
        driver_root = f'{root}.{driver_name}'
        # shared namespace :
        self.ee.ns_manager.add_ns(
            input_ns, f'{driver_root}')
        # shared namespace : shifted by nested operation
        self.ee.ns_manager.add_ns(
            'ns_disc3', f'{driver_root}.Disc3')
        self.ee.ns_manager.add_ns(
            'ns_out_disc3', f'{driver_root}')
        # remark : 'ns_scenario' set to {self.ee.study_name} in subprocess not
        # needed !

        # 3. add multi_scenario
        multi_scenarios = self.ee.factory.create_very_simple_multi_scenario_builder(
            driver_name, scenario_map_name, scatter_list, autogather=False, gather_node='Post-processing')

        return multi_scenarios
