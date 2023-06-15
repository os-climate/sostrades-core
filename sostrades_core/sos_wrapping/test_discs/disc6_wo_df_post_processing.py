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
import numpy as np
from scipy.sparse import diags
# post processing
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import InstanciatedSeries, TwoAxesInstanciatedChart
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter


class Disc6(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df',
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
        'x': {'type': 'array', 'visibility':  SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    DESC_OUT = {
        'h': {'type': 'array', 'visibility':  SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')

        h = np.array([0.5 * (x[0] + 1. / (2 * x[0])),
                      0.5 * (x[1] + 1. / (2 * x[1]))])
        dict_values = {'h': h}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        x = self.get_sosdisc_inputs('x')
        grad = [0.5 * (1.0 - 0.5 / x[0] ** 2), 0.5 * (1.0 - 0.5 / x[1] ** 2)]
        value = diags(grad) / 2
        self.set_partial_derivative('h', 'x', value)

    def get_chart_filter_list(self):

        chart_filters = []

        chart_list = ['h vs x']

        chart_filters.append(ChartFilter(
            'Charts', chart_list, chart_list, 'graphs'))

        return chart_filters

    def get_post_processing_list(self, filters=None):

        instanciated_charts = []

        # Overload default value with chart filter
        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'graphs':
                    charts_list = chart_filter.selected_values

        if 'h vs x' in charts_list:

            chart_name = 'h vs x'

            h = list(self.get_sosdisc_outputs('h'))
            x = list(self.get_sosdisc_inputs('x') * np.array([0., 1.]))

            new_chart = TwoAxesInstanciatedChart('x (-)', 'h (-)',
                                                 chart_name=chart_name)
            serie = InstanciatedSeries(
                x, h, '', 'lines')

            new_chart.series.append(serie)

            instanciated_charts.append(new_chart)

        return instanciated_charts
