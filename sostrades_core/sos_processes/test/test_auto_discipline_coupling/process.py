'''
Copyright 2024 Capgemini
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
from sostrades_core.tools.discipline_decorator.function_discipline_decorator import AUTO_DISCIPLINE_NAMESPACE


class ProcessBuilder(BaseProcessBuilder):
    """Process builder for auto-discipline coupling demonstration."""

    # ontology information
    _ontology_data = {
        'label': 'Auto Discipline Coupling Process',
        'description': 'Demonstration of auto_sos_discipline decorator with coupling between disciplines',
        'category': 'Test',
        'version': '1.0',
    }

    def get_builders(self):
        """Build the process with two coupled auto-generated disciplines."""
        disc_dir = 'sostrades_core.sos_wrapping.test_discs.auto_disciplines.'
        mods_dict = {
            'TempConverter': disc_dir + 'temp_converter.TempConverter',
            'EnergyCalc': disc_dir + 'energy_calculator.EnergyCalc',
        }
        builder_list = self.create_builder_list(mods_dict, ns_dict={AUTO_DISCIPLINE_NAMESPACE: self.ee.study_name})

        return builder_list
