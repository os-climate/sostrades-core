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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from numpy import array
from pandas import DataFrame


class Disc4EmptyDf(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc4_dict_empty_df',
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
        'h': {'type': 'dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    DESC_OUT = {
        'z': {'type': 'array', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_empty_df': {'type': 'dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_empty_list': {'type': 'dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'empty_df': {'type': 'dataframe', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):

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
        empty_df = DataFrame(columns=['col1', 'col2'])
        dict_empty_df = {'key1': {'key11': empty_df, 'key12': empty_df, 'key14': {'key141': empty_df}},
                         'key2': empty_df}
        empty_list = []
        dict_empty_list = {'key1': {'key11': empty_list, 'key12': empty_list, 'key14': {'key141': empty_list}},
                           'key2': empty_list}
        # store outputs
        dict_values = {'z': z,
                       'dict_out': dict_out,
                       'dict_empty_df': dict_empty_df,
                       'dict_empty_list': dict_empty_list,
                       'empty_df': empty_df}
        self.store_sos_outputs_values(dict_values)
