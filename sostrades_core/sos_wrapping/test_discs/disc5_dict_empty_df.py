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
from sostrades_core.execution_engine.SoSWrapp import SoSWrapp
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class Disc5EmptyDf(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc5_disc_empty_df',
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
        'z': {'type': 'array', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_empty_df': {'type': 'dict', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_empty_list': {'type': 'dict', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'empty_df': {'type': 'dataframe', 'visibility':  ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}}

    DESC_OUT = {
        'is_df_empty': {'type': 'bool'},
        'is_dict_empty_df_empty': {'type': 'bool'},
        'is_dict_empty_list_empty': {'type': 'bool'}
    }

    def run(self):
        dict_out = self.get_sosdisc_inputs('dict_out')
        key1 = dict_out['key1']
        key11 = key1['key11']
        key14 = key1['key14']
        df = key14['key141']
        val = df['col2'][1]
        z = self.get_sosdisc_inputs('z')

        dict_empty_df = self.get_sosdisc_inputs('dict_empty_df')
        dict_empty_list = self.get_sosdisc_inputs('dict_empty_list')
        empty_df = self.get_sosdisc_inputs('empty_df')

        dict_empty_df_th = {'key1': {'key11': empty_df, 'key12': empty_df, 'key14': {'key141': empty_df}},
                            'key2': empty_df}

        empty_list = []
        dict_empty_list_th = {'key1': {'key11': empty_list, 'key12': empty_list, 'key14': {'key141': empty_list}},
                              'key2': empty_list}

        dict_values = {'is_df_empty': empty_df.empty,
                       'is_dict_empty_df_empty': dict_empty_df_th.keys() == dict_empty_df.keys(),
                       'is_dict_empty_list_empty': dict_empty_list_th == dict_empty_list}
        self.store_sos_outputs_values(dict_values)
