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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
#-- Generate test 2 process

import pandas as pd
from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    # ontology information
    _ontology_data = {
        'label': 'Core Test Multi Scenario Architecture Process',
        'description': '',
        'category': '',
        'version': '',
    }

    def get_builders(self):

        vb_builder_name = 'Business'

        architecture_df = pd.DataFrame(
            {'Parent': ['Business', 'Remy', 'Remy'],
             'Current': ['Remy', 'CAPEX', 'OPEX'],
             'Type': ['SumValueBlockDiscipline', 'SumValueBlockDiscipline', 'SumValueBlockDiscipline'],
             'Action': [('standard'),  ('standard'),  ('standard')],
             'Activation': [True, False, False]})

        builder = self.ee.factory.create_architecture_builder(
            vb_builder_name, architecture_df)

        scenario_map = {'input_name': 'scenario_list',
                        'input_type': 'string_list',
                        'input_ns': 'ns_scatter_scenario',
                        'scatter_ns': 'ns_scenario'}

        self.ee.scattermap_manager.add_build_map(
            'scenario_list', scenario_map)
        self.ee.ns_manager.add_ns_def(
            {'ns_scatter_scenario': f'{self.ee.study_name}.multi_scenarios'})

        multi_scenarios = self.ee.factory.create_driver(
            'multi_scenarios',  [builder], map_name='scenario_list')

        return multi_scenarios
