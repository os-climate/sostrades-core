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
        'label': 'Process vs_MS (very simple Multi Scenarios) driver creation',
        'description': 'Process to instantiate the vs_MS driver (very simple Multi Scenarios) without any nested builder or by specifiying the nested builder from a process.py python file',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        driver_name = 'vs_MS'
        autogather = False
        gather_node = 'Post-processing'
        business_post_proc = False

        # 1. empty nested process and associated map
        builder_list = []
        scenario_map_name = ''

        # 2. add multi_scenario
        multi_scenarios = (
            self.ee.factory.create_build_very_simple_multi_scenario_builder(
                driver_name,
                scenario_map_name,
                builder_list,
                autogather=autogather,
                gather_node=gather_node,
                business_post_proc=business_post_proc,
            )
        )
        
        return multi_scenarios
