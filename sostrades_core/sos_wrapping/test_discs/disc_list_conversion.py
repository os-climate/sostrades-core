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
import pandas as pd
import numpy as np

from sostrades_core.execution_engine.discipline_proxy import DisciplineProxy


class Disc(DisciplineProxy):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_list_conversion',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }

    _maturity = 'Fake'
    dict_data = {'col1': 1., 'col2': 2., 'col3': 3., 'col4': 4.}
    df_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 0.5]})
    dict_df_data = {f'key{i}': pd.DataFrame({'col1': [1, 2], 'col2': [3, 0.5]}) for i in range(1, 6)}
    list_dict_df = [dict_df_data, dict_df_data, dict_df_data]
    dict_list_dict_df = {'key1': list_dict_df, 'key2': list_dict_df, 'key3': list_dict_df, 'key4': list_dict_df,
                         'key5': list_dict_df}
    float_list = [1.5, 2., 4., 5.5]
    dict_list = [dict_data, dict_data, dict_data, dict_data]
    array_list = [np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]),
                  np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5, 6])]
    dict_list_array = {'key1': array_list, 'key2': array_list, 'key3': array_list, 'key4': array_list,
                       'key5': array_list}
    list_dict_list_array = [dict_list_array, dict_list_array, dict_list_array]
    list_array_list = [array_list, array_list, array_list, array_list]
    df_list = [df_data, df_data, df_data, df_data, df_data]
    dict_dict_array = {f'key{i}': {'key1': np.array([1, 2, 3]), 'key2': np.array([1, 2, 3, 4]),
                                   'key3': np.array([1, 2, 3, 4, 5]),
                                   'key4': np.array([1, 2, 3]), 'key5': np.array([1, 2, 3, 4, 5, 6])} for i in
                       range(1, 6)}

    DESC_IN = {
        'list_float': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                       'default': [1.5, 2.5, 3.5, 4.5, 5], 'subtype_descriptor': {'list': 'float'}},
        'list_array': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                       'default': [np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]),
                                   np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5, 6])],
                       'subtype_descriptor': {'list': 'array'}},

        'list_dataframe': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                           'default': [df_data, df_data, df_data, df_data, df_data],
                           'subtype_descriptor': {'list': 'dataframe'}},
        'list_dict_float': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                            'default': [dict_data, dict_data, dict_data, dict_data, dict_data],
                            'subtype_descriptor': {'list': {'dict': 'float'}}},
        'list_list_float': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                            'default': [float_list, float_list, float_list, float_list, float_list],
                            'subtype_descriptor': {'list': {'list': 'float'}}},
        'list_list_dict_float': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                 'default': [dict_list, dict_list, dict_list, dict_list, dict_list],
                                 'subtype_descriptor': {'list': {'list': {'dict': 'float'}}}},
        'list_list_dataframe': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                'default': [df_list, df_list, df_list],
                                'subtype_descriptor': {'list': {'list': 'dataframe'}}},
        'list_list_list_array': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                 'default': [list_array_list, list_array_list, list_array_list, list_array_list,
                                             list_array_list],
                                 'subtype_descriptor': {'list': {'list': {'list': 'array'}}}},
        'dict_float': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                       'default': {'key1': 0.5, 'key2': 0.5, 'key3': 1., 'key4': 2.5, 'key5': 1.5},
                       'subtype_descriptor': {'dict': 'float'}},
        'dict_array': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                       'default': {'key1': np.array([1, 2, 3]), 'key2': np.array([1, 2, 3, 4]),
                                   'key3': np.array([1, 2, 3, 4, 5]),
                                   'key4': np.array([1, 2, 3]), 'key5': np.array([1, 2, 3, 4, 5, 6])},
                       'subtype_descriptor': {'dict': 'array'}},
        'dict_dataframe': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                           'default': {'key1': df_data, 'key2': df_data,
                                       'key3': df_data,
                                       'key4': df_data, 'key5': df_data},
                           'subtype_descriptor': {'dict': 'dataframe'}},
        'dict_dataframe_array': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                 'default': {'key1': df_data, 'key2': df_data,
                                             'key3': df_data,
                                             'key4': df_data, 'key5': pd.DataFrame(
                                         data={'col1': [1, np.array([0.70710678, 0.70710678])]})},
                                 'subtype_descriptor': {'dict': 'dataframe'}},
        'large_dict': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                       'default': {'key1': df_data, 'key2': df_data,
                                   'key3': df_data,
                                   'key4': df_data, 'key5': df_data},
                       'subtype_descriptor': {'dict': 'dataframe'}},
        'dict_dict_float': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                            'default': {f'key{i}': {'1': i + 1, '2': i + 2,
                                                    '3': i + 3, '4': i + 4} for i in range(500)},
                            'subtype_descriptor': {'dict': {'dict': 'float'}}},

        'dict_dict_dataframe': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                'default': {
                                    f'key{i}': {f'key{i}': pd.DataFrame({'col1': [1, 2], 'col2': [3, 0.5]}) for i in
                                                range(1, 6)} for i in range(1, 6)},
                                'subtype_descriptor': {'dict': {'dict': 'dataframe'}}},

        'dict_dict_dict_array': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                 'default': {f'key{i}': {
                                     f'key{i}': {'key1': np.array([1, 2, 3]), 'key2': np.array([1, 2, 3, 4]),
                                                 'key3': np.array([1, 2, 3, 4, 5]),
                                                 'key4': np.array([1, 2, 3]), 'key5': np.array([1, 2, 3, 4, 5, 6])} for
                                     i in range(1, 6)} for i in range(1, 6)},
                                 'subtype_descriptor': {'dict': {'dict': {'dict': 'array'}}}},
        'dict_list_float': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                            'default': {f'key{i}': [1.5, 2.5, 3.5, 4.5, 5] for i in range(1, 6)},
                            'subtype_descriptor': {'dict': {'list': 'float'}}},
        'dict_list_list_dataframe': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                     'default': {'key1': [df_list, df_list, df_list],
                                                 'key2': [df_list, df_list, df_list],
                                                 'key3': [df_list, df_list, df_list],
                                                 'key4': [df_list, df_list, df_list],
                                                 'key5': [df_list, df_list, df_list]},
                                     'subtype_descriptor': {'dict': {'list': {'list': 'dataframe'}}}},

        'list_dict_list_array': {'type': 'list', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                 'default': list_dict_list_array,
                                 'subtype_descriptor': {'list': {'dict': {'list': 'array'}}}},
        'dict_list_dict_dataframe': {'type': 'dict', 'visibility': DisciplineProxy.LOCAL_VISIBILITY,
                                     'default': dict_list_dict_df,
                                     'subtype_descriptor': {'dict': {'list': {'dict': 'dataframe'}}}},

    }
    DESC_OUT = {
        'indicator': {'type': 'float'}
    }

    def run(self):
        dict_values = {'indicator': 1}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
