'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
import pandas
from numpy import array

from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

# Discipline with dataframe


class Disc7(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc7',
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
        'h': {'type': 'array', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'},
    }

    DESC_OUT = {
        'df': {'type': 'dataframe', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'},
        'dict_df': {'type': 'dict', 'subtype_descriptor': {'dict': 'dataframe'}, 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    def run(self):
        h = self.get_sosdisc_inputs('h')
        df = pandas.DataFrame(
            array([[(h[0] + h[1]) / 2, (h[0] + h[1]) / 2]]), columns=['c1', 'c2'])
        dict_df = {'key_1': df, 'key_2': df}
        dict_values = {'df': df, 'dict_df': dict_df}
        self.store_sos_outputs_values(dict_values)
