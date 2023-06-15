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
from numpy import array


# Discipline with dataframe


class Disc6(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc6',
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
        'df': {'type': 'dataframe', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected',
               'dataframe_descriptor': {'years': ('float', [0., 3000.], True),
                                        'year': ('float', [0., 3000.], True),
                                        'c1': ('float', [-1e4, 1e4], True),
                                        'c2': ('float', None, True)}},
        'dict_df': {'type': 'dict', 'subtype_descriptor': {'dict': 'dataframe'},
                    'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    DESC_OUT = {
        'h': {'type': 'array', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    def run(self):
        df = self.get_sosdisc_inputs('df')
        dict_df = self.get_sosdisc_inputs('dict_df')
        key1 = df['c1'][0]
        key2 = df['c2'][0]
        h = array([0.5 * (key1 + 1. / (2 * key1)),
                   0.5 * (key2 + 1. / (2 * key2))])
        dict_values = {'h': h}
        self.store_sos_outputs_values(dict_values)
