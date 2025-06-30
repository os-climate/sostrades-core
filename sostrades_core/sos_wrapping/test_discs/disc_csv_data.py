'''
Copyright 2022 Airbus SAS
Modifications on 27/06/25 Copyright 2025 Capgemini

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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class DiscCsvData(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_csv_data',
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
    # Test disc to validate saving based on csv file transfer from GUI
    _maturity = 'Fake'

    DESC_IN = {
        'dict_mix_types': {'type': 'dict',  'namespace': 'ns_test'},
        'array_mix_types': {'type': 'array',  'namespace': 'ns_test'},
        'dataframe_mix_types': {'type': 'dataframe',  'namespace': 'ns_test'},
        'dict_as_dict_dataframe': {'type': 'dict', 'subtype_descriptor': {'dict': 'dataframe'},  'namespace': 'ns_test'}
    }

    DESC_OUT = {'z': {'type': 'float'}}

    def run(self):
        # Fake run not used
        inputs = self.get_sosdisc_inputs()
        z = 42

        self.store_sos_outputs_values({'z': z})
