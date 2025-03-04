'''
Copyright 2022 Airbus SAS
Modifications on 2025/02/18-2025/02/18 Copyright 2025 Capgemini

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
from sostrades_core.sos_wrapping.valueblock_discipline import ValueBlockDiscipline


class FakeValueBlockDiscipline(ValueBlockDiscipline):
    """Fake value block discipline to test architecture builder functionalities"""

    # ontology information
    _ontology_data = {
        'label': 'Fake Value Block Discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Research'

    DESC_OUT = {'output': {'type': 'dict'}}

    def run(self):

        self.store_sos_outputs_values({'output': {'out': [0, 1, 2],
                                                  'out2': [2, 3, 4], }})
