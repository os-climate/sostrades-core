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
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
import pandas as pd


class Disc1(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc1_grid',
        'type': '',
        'source': '',
        'validated': '',
        'validated_by': '',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'x': {'type': 'float', 'default': 0.},
        'd': {'type': 'float', 'default': 0.},
        'f': {'type': 'float', 'default': 0.},
        'g': {'type': 'float', 'default': 0.},
        'h': {'type': 'float', 'default': 0.},
        'j': {'type': 'float', 'default': 0.},
        'a': {'type': 'int', 'default': 0},
        'b': {'type': 'int', 'possible_values': [0, 2, 5]},
        'name': {'type': 'string', 'possible_values': ['A1', 'A2', 'A3']},
        'x_dict': {'type': 'dict', 'default': {}},
        'di_dict': {'type': 'dict', 'default': {}, 'namespace': 'ns_test'},
        'dd_df': {'type': 'dataframe', 'default': pd.DataFrame(), 'namespace': 'ns_test'}
    }
    DESC_OUT = {
        'indicator': {'type': 'int'},
        'y': {'type': 'float'},
        'val_sum_dict': {'type': 'dict'},
        'val_sum_dict2': {'type': 'dict'},
        'y_dict2': {'type': 'dict'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        name = self.get_sosdisc_inputs('name')
        x_dict = self.get_sosdisc_inputs('x_dict')
        di_dict = self.get_sosdisc_inputs('di_dict')
        dd_df = self.get_sosdisc_inputs('dd_df')

        y_dict = {}
        val_sum_dict = {}
        val_sum_dict2 = {}
        for name_i, x_i in x_dict.items():
            y_dict[name_i] = a * x_i + b

        for col in dd_df:
            if dd_df[col].dtype == float:
                val_sum_dict[col] = dd_df[col].cumsum().values[-1]

        for key in di_dict:
            if isinstance(di_dict[key], float):
                val_sum_dict2[key] = di_dict[key]

        dict_values = {'indicator': a * b, 'y': a *
                       x + b, 'y_dict2': y_dict, 'val_sum_dict': val_sum_dict, 'val_sum_dict2': val_sum_dict2}

        self.store_sos_outputs_values(dict_values)
