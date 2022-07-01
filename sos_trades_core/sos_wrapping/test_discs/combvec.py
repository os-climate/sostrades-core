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
import itertools
import numpy as np
import pandas as pd


class Combvec(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'Combvec discipline',
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

    default_my_dict_of_vec = {}
    default_my_dict_of_vec['stat_A'] = [2, 7]
    default_my_dict_of_vec['stat_B'] = [2]
    default_my_dict_of_vec['stat_C'] = [3, 4, 8]

    DESC_IN = {
        'my_dict_of_vec': {'type': 'dict', 'subtype_descriptor': {'dict': {'list':'int'}}, 'default': default_my_dict_of_vec, 'unit': '-', 'visibility': SoSDiscipline.LOCAL_VISIBILITY}
    }
    DESC_OUT = {
        'custom_samples_df': {'type': 'dataframe', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY,
                              'namespace': 'ns_doe_eval'}
    }

    def run(self):

        my_dict_of_vec = self.get_sosdisc_inputs('my_dict_of_vec')
        my_df = self.comb_dict_vect2df(my_dict_of_vec)
        dict_values = {'custom_samples_df': my_df}
        self.store_sos_outputs_values(dict_values)

    def combvec(self, vect_list):
        my_sample = list(itertools.product(*vect_list))
        return my_sample

    def comb_dict_vect2df(self, my_dict_of_vec):
        vect_list = [my_dict_of_vec[elem] for elem in my_dict_of_vec.keys()]
        my_res = self.combvec(vect_list)
        my_res = np.array(my_res)
        my_df = pd.DataFrame(my_res, columns=my_dict_of_vec.keys())
        return my_df
