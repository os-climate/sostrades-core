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

from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class Disc(SoSDiscipline):
    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc_list_conversion',
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
    float_list = [1.5, 2., 4., 5.5]
    dict_list = [dict_data, dict_data, dict_data, dict_data]
    array_list = [np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]),
                  np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5, 6])]
    list_array_list = [array_list, array_list, array_list, array_list]
    df_list = [df_data, df_data, df_data, df_data, df_data]

    DESC_IN = {
        'list_float': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                       'default': [1.5, 2.5, 3.5, 4.5, 5], 'subtype_descriptor': {'list': 'float'}},
        'list_array': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                       'default': [np.array([1, 2, 3]), np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4, 5]),
                                   np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5, 6])],
                       'subtype_descriptor': {'list': 'array'}},

        'list_dataframe': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                           'default': [df_data, df_data, df_data, df_data, df_data],
                           'subtype_descriptor': {'list': 'dataframe'}},
        'list_dict': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                      'default': [dict_data, dict_data, dict_data, dict_data, dict_data],
                      'subtype_descriptor': {'list': 'dict'}},
        'list_list_float': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                            'default': [float_list, float_list, float_list, float_list, float_list],
                            'subtype_descriptor': {'list': {'list': 'float'}}},
        'list_list_dict': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                           'default': [dict_list, dict_list, dict_list, dict_list, dict_list],
                           'subtype_descriptor': {'list': {'list': 'dict'}}},
        'list_list_dataframe': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                                'default': [df_list, df_list, df_list],
                                'subtype_descriptor': {'list': {'list': 'dataframe'}}},
        'list_list_list_array': {'type': 'list', 'visibility': SoSDiscipline.LOCAL_VISIBILITY,
                                 'default': [list_array_list, list_array_list, list_array_list, list_array_list,
                                             list_array_list],
                                 'subtype_descriptor': {'list': {'list': {'list': 'array'}}}},

    }
    DESC_OUT = {
        'indicator': {'type': 'float'}
    }

    def run(self):
        dict_values = {'indicator': 1}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
