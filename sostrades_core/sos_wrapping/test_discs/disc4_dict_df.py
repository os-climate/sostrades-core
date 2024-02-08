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
from pandas import DataFrame


class Disc4(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc4_dict_df',
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
        'h': {'type': 'dict', 'subtype_descriptor':{'dict':'dataframe'}, 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'mydict': {'type': 'dict', 'subtype_descriptor':{'dict':'array'}}
    }

    DESC_OUT = {
        'z': {'type': 'array', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):
        # get unused dict
        self.get_sosdisc_inputs('mydict')

        # get dict of df
        h = self.get_sosdisc_inputs('h')
        df = h['dataframe']
        h = df['col1']

        # compute values
        val1 = (h[0] + h[1]) / 2.
        val2 = (h[0] + h[1]) / 2.

        # set dict of dicts/df
        df_data = {'col1': [1, 2], 'col2': [3, val2]}
        df = DataFrame(data=df_data)
        dict_out = {'key1': {'key11': val1, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
                    'key2': 10.}

        # set z
        z = array([h[0], 2 * h[1]])

        # store outputs
        dict_values = {'z': z,
                       'dict_out': dict_out}
        self.store_sos_outputs_values(dict_values)
