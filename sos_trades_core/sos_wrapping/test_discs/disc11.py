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
from sos_trades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sos_trades_core.tools.post_processing.charts.chart_filter import ChartFilter


class Disc11(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc11',
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
    DESC_IN = {
        'x': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'test_df': {'type': 'dataframe', 'unit': '-', 'is_formula': True},
        'c_dict': {'type': 'dict', 'unit': '-', 'is_formula': True},
        'test_string': {'type': 'string', 'unit': '-'}
    }
    DESC_OUT = {
        'indicator': {'type': 'float', 'unit': '-'},
        'y': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'out_string': {'type': 'string', 'unit': '-'},
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        c_dict = self.get_sosdisc_inputs('c_dict')
        c = c_dict['c']
        test_df = self.get_sosdisc_inputs('test_df')
        a = test_df['a'].values
        b = test_df['b'].values
        test_string = self.get_sosdisc_inputs('test_string')
        dict_values = {'indicator': a * b, 'y': a *
                       x + b + c, 'out_string': test_string}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
