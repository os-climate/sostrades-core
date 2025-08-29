'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2024/06/10 Copyright 2023 Capgemini

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

from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):
    """Process builder for PuLP optimization test case."""

    # ontology information
    _ontology_data = {
        'label': 'PuLP Optimization Process',
        'description': 'Process for testing PuLP linear programming optimization',
        'category': 'Test',
        'version': '1.0.0',
    }

    def get_builders(self):
        """Get the builders for the process."""
        
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.pulp_optim.'
        
        builder_list = []
        
        # Add the linear programming discipline
        builder_list.append(
            self.ee.factory.get_builder_from_module(
                'LinearProgrammingDisc', disc_dir + 'linear_programming_disc'
            )
        )
        
        return builder_list
