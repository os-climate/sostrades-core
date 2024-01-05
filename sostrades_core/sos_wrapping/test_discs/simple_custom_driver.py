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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class SimpleCustomDriver(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.simple_custom_driver',
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
    _maturity = 'Fake'

    DESC_IN = {
        'output_full_name': {'type': 'string'}
    }
    DESC_OUT = {
        'output_squared': {'type': 'float'}
    }

    def run(self):
        input_data = {}
        input_data_names = self.attributes['sub_mdo_disciplines'][0].input_grammar.names
        if len(input_data_names) > 0:
            input_data = self.get_sosdisc_inputs(keys=input_data_names, in_dict=True, full_name_keys=True)
        sub_disc_local_data = self.attributes['sub_mdo_disciplines'][0].execute(input_data)
        output_val = sub_disc_local_data.get(self.get_sosdisc_inputs('output_full_name'))
        dict_values = {'output_squared': output_val ** 2}
        self.store_sos_outputs_values(dict_values)