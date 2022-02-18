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
import pandas
# Discipline with dataframe


class Disc7(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc7',
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
        'h': {'type': 'array', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'},
    }

    DESC_OUT = {
        'df': {'type': 'dataframe', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'},
        'dict_df': {'type': 'df_dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    def run(self):
        h = self.get_sosdisc_inputs('h')
        df = pandas.DataFrame(
            array([[(h[0] + h[1]) / 2, (h[0] + h[1]) / 2]]), columns=['c1', 'c2'])
        dict_df = {'key_1': df, 'key_2': df}
        dict_values = {'df': df, 'dict_df': dict_df}
        self.store_sos_outputs_values(dict_values)
